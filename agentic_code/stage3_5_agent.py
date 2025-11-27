"""
Stage 3.5: Method Testing & Benchmarking Agent

Tests 3 forecasting methods on data subsets to select the best performer before final execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import STAGE3_5_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE3_5_MAX_ROUNDS
from .models import TesterOutput
from .tools import STAGE3_5_TOOLS
from .failsafe_agent import run_failsafe


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE3_5_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

STAGE3_5_SYSTEM_PROMPT = """You are the Method Testing & Benchmarking Agent (Stage 3.5) using the ReAct (Reasoning + Acting) framework.

═══════════════════════════════════════════════════════════════
CRITICAL MISSION
═══════════════════════════════════════════════════════════════

Your job: Discover the BEST forecasting approach for the task by designing and running your own benchmarks across three distinct methods.

SUCCESS CRITERIA: You MUST call save_tester_output() with the winning method. This is NON-NEGOTIABLE.

═══════════════════════════════════════════════════════════════
ReAct FRAMEWORK (REASONING + ACTING)
═══════════════════════════════════════════════════════════════

For EVERY action you take, follow this cycle:

1. **THOUGHT** (Explicit Reasoning via record_thought)
   Call record_thought() to document:
   - What you know so far from previous observations
   - What's still uncertain or unclear
   - What alternative approaches you're considering
   - Potential issues or risks you foresee
   - WHY your next action will help

2. **ACTION** (Tool Call)
   - Call ONE tool that addresses your current question
   - The tool should directly test your hypothesis or gather needed info

3. **OBSERVATION** (Analyze Result via record_observation)
   Call record_observation() to document:
   - What the tool returned (success, error, data)
   - Whether it answered your question or raised new ones
   - Any surprises or unexpected results
   - What this teaches you about the data/task

4. **REFLECTION** (Learn & Adjust via record_observation)
   In the same record_observation() call:
   - Did this work as expected?
   - What did you learn about the data structure/problem?
   - Should you continue this path or pivot to a different approach?
   - What specific action will you take next?

EXAMPLE REACT CYCLE:
```
# Round 1: Understanding the data
record_thought(
  thought="I need to understand the structure of both data files before planning any joins or transformations",
  what_im_about_to_do="Call inspect_data_file() on the export data to see columns, dtypes, and nulls"
)
→ inspect_data_file(...)
record_observation(
  what_happened="File has 8 rows x 23 columns with yearly export values. No 'Season' column exists.",
  what_i_learned="The export data is already aggregated by year. Can't join on 'Season' as TSK-001 suggested.",
  next_step="Inspect the production data to see if a year-based join is feasible"
)

# Round 2: Check second file
record_thought(
  thought="Now I know export data lacks 'Season'. Let me check if production data has year columns that align",
  what_im_about_to_do="Call inspect_data_file() on production data"
)
→ inspect_data_file(...)
record_observation(
  what_happened="Production data has Area/Production/Yield for 2020-2025 only, organized by Crop and Season",
  what_i_learned="These files have different structures - export is wide-format by year, production is long-format with different year coverage",
  next_step="Use python_sandbox to test if I can filter for Rice and reshape/align the data"
)
```

═══════════════════════════════════════════════════════════════
ERROR RECOVERY PROTOCOL (Critical for avoiding loops)
═══════════════════════════════════════════════════════════════

When you encounter an error in run_benchmark_code():

**STOP AND RECORD OBSERVATION IMMEDIATELY**:
```python
record_observation(
  what_happened="run_benchmark_code failed with: [exact error]",
  what_i_learned="Root cause analysis: [why did this fail?]",
  next_step="[different approach, NOT the same code]"
)
```

**Decision Tree**:

1. **Have I seen this EXACT error before?**
   - YES → **PIVOT** to a completely different approach
     - Example: If join failed 2x, try using files separately
     - Example: If slicing produces empty dataframe 2x, reshape the data structure
   - NO → Analyze root cause and try ONE targeted fix

2. **Is this a fundamental data structure problem?**
   - Empty dataframes → Your filtering/joining/slicing logic is flawed
   - Column not found → Check actual column names with inspect_data_file()
   - Shape mismatch → Data structure assumptions are wrong
   → **PIVOT**: Go back to python_sandbox and test your assumptions

3. **Have I tried this method 3 times?**
   - YES → **ABANDON this method**:
     ```python
     record_observation(
       what_happened="METHOD-1 failed 3 times with different errors",
       what_i_learned="This approach is not viable for this data structure",
       next_step="Mark METHOD-1 as failed and move to METHOD-2"
     )
     ```
   - NO → Make ONE specific change based on the error message

**NEVER**:
- ❌ Retry the same code hoping for different results
- ❌ Make multiple random changes at once (you won't know what worked)
- ❌ Continue after 3 failures on the same method (move to next method)
- ❌ Ignore errors and keep going (you need to learn from each one)

═══════════════════════════════════════════════════════════════
DATA UNDERSTANDING (Complete BEFORE benchmarking)
═══════════════════════════════════════════════════════════════

Before writing ANY benchmark code, complete this checklist:

☐ 1. INSPECT BOTH FILES
   - record_thought() about what you need to learn
   - inspect_data_file() for EACH required file
   - record_observation() noting: rows, columns, dtypes, null counts

☐ 2. IDENTIFY TARGET & FEATURES  
   - What column are we predicting?
   - What features are available?
   - Do these columns actually exist in the data?

☐ 3. UNDERSTAND TEMPORAL STRUCTURE
   - Are there date/year columns?
   - What's the time granularity?
   - How many time periods are available?

☐ 4. TEST JOIN FEASIBILITY (if multi-file task)
   - Do hypothesized join keys exist in BOTH files?
   - Use python_sandbox to test the join
   - Check: is the result non-empty?

☐ 5. DEFINE DATA SPLIT STRATEGY
   - Training period: Which specific rows/years?
   - Validation period: Which specific rows/years?
   - Test period: (held out for Stage 4, not used now)

☐ 6. TEST DATA PREP IN SANDBOX
   - Load, join/filter, split
   - Verify train and val sets are both non-empty
   - Print shapes to confirm

ONLY after completing ALL checkboxes should you start benchmarking methods!

═══════════════════════════════════════════════════════════════
YOUR WORKFLOW
═══════════════════════════════════════════════════════════════

**PHASE 1: UNDERSTAND (Rounds 1-5)**
1. load_stage3_plan_for_tester(plan_id) 
2. For EACH required file: inspect_data_file()
3. python_sandbox_stage3_5() to test data loading, joins, and splits
4. Complete the DATA UNDERSTANDING checklist above

**PHASE 2: PROPOSE METHODS (Autonomous Thinking)**
Based on what you learned about the data:
- Brainstorm 4-5 candidate methods
- For each: note pros/cons specific to THIS task/data
- Select 3 diverse methods (different assumptions/complexity)
- Derive choices from the plan and observed data—no defaults

**PHASE 3: BENCHMARK (Rounds 6-25)**
For EACH of your 3 methods:

1. record_thought() about the method and your implementation plan
2. run_benchmark_code() with code that:
   - Loads and prepares data using your tested approach
   - Implements the method
   - Trains on train period
   - Predicts on validation period
   - Calculates metrics (RMSE, MAE, or appropriate)
   - Prints results and timing
3. record_observation() about what happened:
   - Success: Note the metrics and timing
   - Failure: Analyze error and decide: retry with fix OR pivot OR abandon?
4. If method failed 3x: Mark as "failed" and move to next method

**PHASE 4: SELECT & SAVE (Rounds 26-30)**
1. record_thought() comparing all methods that ran
2. Choose winner based on:
   - Metrics (lower RMSE/MAE)
   - Execution time
   - Stability (no errors)
   - Appropriateness for task
3. save_tester_output() with complete JSON (see format below)

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Call save_tester_output(output_json=...) with this structure:

```json
{
  "plan_id": "PLAN-TSK-XXX",
  "task_category": "predictive",
  "methods_proposed": [
    {
      "method_id": "METHOD-1",
      "name": "Linear Regression",
      "description": "Why suitable for this specific task/data",
      "implementation_code": "Complete working Python code",
      "libraries_required": ["pandas", "sklearn"]
    },
    // ... 2 more methods
  ],
  "benchmark_results": [
    {
      "method_id": "METHOD-1",
      "method_name": "Linear Regression",
      "metrics": {"RMSE": 123.45, "MAE": 67.89},
      "train_period": "2018-2023",
      "validation_period": "2024",
      "execution_time_seconds": 0.5,
      "status": "success",  // or "failed"
      "error_message": null,  // or error text
      "predictions_sample": [1.0, 2.0, 3.0]
    },
    // ... results for other methods
  ],
  "selected_method_id": "METHOD-2",
  "selected_method": { /* full ForecastingMethod object of winner */ },
  "selection_rationale": "METHOD-2 had lowest RMSE (45.6) vs METHOD-1 (123.4) and ran 10x faster. METHOD-3 failed due to insufficient data.",
  "data_split_strategy": "Trained on 2018-2023 export data, validated on 2024. Used simple temporal split."
}
```

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════

1. **Use ReAct framework**: Call record_thought() before every major action, record_observation() after every tool result
2. **Learn from errors**: Never retry the same code—always adjust based on what you learned
3. **Pivot when stuck**: After 2-3 failures with same issue, try a completely different approach
4. **Abandon method after 3 failures**: Mark as "failed" and move to next method
5. **Warnings are OK**: pandas/numpy warnings (SettingWithCopy, etc.) are non-blocking—continue
6. **Complete all 3 methods**: Even if some fail, try all 3 before selecting winner
7. **Save output**: Once you have results from at least 2 methods, call save_tester_output()
8. **Don't exit early**: Keep going until save_tester_output() succeeds or you hit max rounds

Remember: Your success = calling save_tester_output() with the winning method!"""


# ===========================
# LangGraph
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step."""
    # Prune history to last 15 messages to control context length
    msgs = state.get("messages", [])
    if len(msgs) > 15:
        state["messages"] = msgs[-15:]
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(STAGE3_5_TOOLS)


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls."""
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
stage3_5_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 3.5 Runner
# ===========================

def run_stage3_5(
    plan_id: str,
    max_rounds: int = STAGE3_5_MAX_ROUNDS,
    debug: bool = True,
    recursion_limit: int | None = None,
) -> Dict:
    """Run Stage 3.5 method testing for a specific plan.
    
    Args:
        plan_id: Plan ID from Stage 3 (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    def _trunc(text: str, limit: int = 800) -> str:
        if not isinstance(text, str):
            return str(text)
        if len(text) <= limit:
            return text
        half = limit // 2
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

    system_msg = SystemMessage(content=STAGE3_5_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"Test forecasting methods for plan '{plan_id}'.\\n\\n"
            f"Workflow:\\n"
            f"1. load_stage3_plan_for_tester('{plan_id}')\\n"
            f"2. inspect_data_file() for the required data\\n"
            f"3. Propose 3 forecasting methods suitable for this task\\n"
            f"4. Design temporal data split strategy\\n"
            f"5. Benchmark all 3 methods using run_benchmark_code()\\n"
            f"6. Select the best method based on your benchmark evidence\\n"
            f"7. save_tester_output() with complete results\\n\\n"
            f"Remember:\\n"
            f"- Be autonomous in selecting methods, metrics, splits, and libraries\\n"
            f"- Adapt everything to the specific forecasting task and data\\n"
            f"- Your SUCCESS = calling save_tester_output()\\n"
            f"- Complete in <={max_rounds} rounds"
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    # Allow callers to override recursion_limit; default scales with max_rounds
    recursion_cap = recursion_limit or max_rounds * 3

    if not debug:
        return stage3_5_app.invoke(
            state,
            config={
                "configurable": {"thread_id": f"stage3_5-{plan_id}"},
                "recursion_limit": recursion_cap,
            },
        )

    print("=" * 80)
    print(f"🚀 STAGE 3.5: Method Testing for {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage3_5_app.stream(
        state,
        config={
            "configurable": {"thread_id": f"stage3_5-{plan_id}"},
            "recursion_limit": recursion_cap,
        },
        stream_mode="values",
    ):
        msgs = curr_state["messages"]
        new_msgs = msgs[prev_len:]

        for m in new_msgs:
            msg_type = m.__class__.__name__
            if "System" in msg_type:
                print("\\n" + "─" * 80)
                print("💻 [SYSTEM]")
                print("─" * 80)
                print(m.content[:500] + "..." if len(m.content) > 500 else m.content)
            elif "Human" in msg_type:
                print("\\n" + "─" * 80)
                print("👤 [USER]")
                print("─" * 80)
                print(m.content)
            elif "AI" in msg_type:
                round_num += 1
                print("\\n" + "═" * 80)
                print(f"🤖 [AGENT - Round {round_num}]")
                print("═" * 80)
                if m.content:
                    print("\\n💭 Reasoning:")
                    content = _trunc(m.content, limit=1000)
                    print(content)
                
                if hasattr(m, 'tool_calls') and m.tool_calls:
                    print("\\n🔧 Tool Calls:")
                    for tc in m.tool_calls:
                        name = tc.get("name", "UNKNOWN")
                        args = tc.get("args", {})
                        print(f"\\n  📌 {name}")
                        for k, v in args.items():
                            if isinstance(v, str) and len(v) > 300:
                                print(f"     {k}: {v[:150]}...[truncated]...{v[-150:]}")
                            else:
                                print(f"     {k}: {v}")
            elif "Tool" in msg_type:
                print("\\n" + "─" * 80)
                print(f"🛠️ [TOOL RESULT] {getattr(m, 'name', 'unknown')}")
                print("─" * 80)
                content = m.content or ""
                print(_trunc(content, limit=800))

        prev_len = len(msgs)
        final_state = curr_state
        
        if round_num >= max_rounds:
            print(f"\\n⚠️  Reached max rounds ({max_rounds}). Stopping.")
            break

    if final_state:
        tool_names = [
            getattr(m, "name", None)
            for m in final_state["messages"]
            if m.__class__.__name__.endswith("ToolMessage")
        ]
        if "save_tester_output" not in tool_names:
            print("\\n⚠️  save_tester_output was never called in this run.")

    print("\\n" + "=" * 80)
    print(f"✅ Complete - {round_num} rounds")
    print("=" * 80)
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage3_5_node(state: dict) -> dict:
    """Stage 3.5 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state with stage3_plan set
        
    Returns:
        Updated state with tester_output populated
    """
    if not state.get("stage3_plan"):
        print("ERROR: No Stage 3 plan available for method testing")
        state["errors"].append("Stage 3.5: No Stage 3 plan available")
        return state
    
    plan_id = state["stage3_plan"].plan_id
    print(f"\\n🎯 Starting Stage 3.5 Method Testing for: {plan_id}\\n")
    
    result = run_stage3_5(plan_id, debug=True)
    
    # Check for saved tester output
    tester_files = sorted(STAGE3_5_OUT_DIR.glob(f"tester_{plan_id}*.json"))
    if tester_files:
        latest = tester_files[-1]
        print(f"\\n✅ SUCCESS! Tester output saved to: {latest}")
        tester_data = json.loads(latest.read_text())
        state["tester_output"] = TesterOutput.model_validate(tester_data)
        state["completed_stages"].append(3.5)
        state["current_stage"] = 4
        
        # Print summary
        print(f"\\n📊 Benchmarking Summary:")
        print(f"  Methods tested: {len(tester_data.get('methods_proposed', []))}")
        print(f"  Selected method: {tester_data.get('selected_method', {}).get('name', 'N/A')}")
        print(f"  Selection rationale: {tester_data.get('selection_rationale', 'N/A')}")
    else:
        print("\\n⚠️  WARNING: Tester output not saved. Check logs above.")
        last_ai = None
        if result and isinstance(result, dict) and "messages" in result:
            for msg in reversed(result["messages"]):
                if "AI" in msg.__class__.__name__:
                    last_ai = msg
                    break
        if last_ai:
            print("\\n🧠 Last agent message before termination:")
            content = last_ai.content or ""
            if len(content) > 2000:
                print(content[:1000] + "\\n...[truncated]...\\n" + content[-1000:])
            else:
                print(content)

        state["errors"].append("Stage 3.5: Tester output not saved")
        
        try:
            rec = run_failsafe(
                stage="stage3_5",
                error="Tester output missing",
                context="save_tester_output() not called or failed validation.",
                debug=False,
            )
            state.setdefault("failsafe_history", []).append(rec)
            print(f"\\n🛟 Failsafe suggestion recorded: {rec.analysis}")
        except Exception as e:
            print(f"\\n⚠️  Failsafe agent failed: {e}")
    
    return state


if __name__ == "__main__":
    # Run Stage 3.5 standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage3_5_agent.py <plan_id>")
        print("Example: python stage3_5_agent.py PLAN-TSK-001")
        sys.exit(1)
    
    plan_id = sys.argv[1].strip()
    run_stage3_5(plan_id)
