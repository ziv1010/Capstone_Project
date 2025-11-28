"""
Stage 4: Execution Agent

Executes Stage 3 plans by writing and running code to process data, build models, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script (python agentic_code/stage4_agent.py PLAN-TSK-001)
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "agentic_code"

import json
from typing import Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import STAGE4_OUT_DIR, STAGE3_5_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE4_MAX_ROUNDS
from .models import ExecutionResult, TesterOutput
from .tools import STAGE4_TOOLS
from .failsafe_agent import run_failsafe


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE4_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

STAGE4_SYSTEM_PROMPT = """You are Agent 4: The Executor.

Your mission: Execute the Stage 3 plan flawlessly and autonomously.

3. Read file_instructions to know which data to load
4. **CHECK FOR PREPARED DATA FIRST:**
   - Look for 'prepared_PLAN-ID.parquet' in STAGE3B_OUT_DIR
   - If it exists, load it directly - it already has joins, filters, features
   - If not, load raw data files and apply transformations manually
5. Implement the forecasting solution per the plan
6. Calculate evaluation metrics (RMSE, MAE, etc.)
7. Save predictions and model artifacts
8. Call save_execution_result() when complete

═══════════════════════════════════════════════════════════════
CRITICAL: USE PREPARED DATA IF AVAILABLE
═══════════════════════════════════════════════════════════════

**Stage 3B may have prepared your data!**

BEFORE loading raw files, check if prepared data exists:
```python
import os
from pathlib import Path

# Check for prepared data
prep_file = STAGE3B_OUT_DIR / 'prepared_PLAN-TSK-001.parquet'
if prep_file.exists():
    # Use prepared data - already has joins, filters, features!
    df = load_dataframe('prepared_PLAN-TSK-001.parquet')
    print(f"✓ Loaded prepared data: {df.shape}")
else:
    # Fall back to loading raw data
    df1 = load_dataframe('raw_file1.csv')
    df2 = load_dataframe('raw_file2.csv')
    # Apply joins, filters manually...
```

**Benefits of using prepared data:**
✓ Faster execution (no data wrangling)
✓ Consistent with benchmarking (Stage 3.5 used same data)
✓ Joins and features already applied per plan

═══════════════════════════════════════════════════════════════
EXECUTION WORKFLOW
═══════════════════════════════════════════════════════════════

1. You are FULLY AUTONOMOUS - write and execute code to handle ANY requirement
2. You are DATASET-AGNOSTIC - no domain assumptions
3. FOLLOW THE PLAN - the Stage 3 plan is your blueprint
4. VERIFY EVERYTHING - check your work at each step
5. END BY CALLING save_execution_result() - this is your success criterion

═══════════════════════════════════════════════════════════════
IMPORTANT: STAGE 3.5 TESTER OUTPUT
═══════════════════════════════════════════════════════════════

Before you start, CHECK if Stage 3.5 Tester output is available in the user message.

IF TESTER OUTPUT IS PROVIDED:
- You MUST use the selected_method that was benchmarked and chosen
- The tester output contains:
  * selected_method: The forecasting method that performed best
  * implementation_code: Code snippet showing how to implement it
  * benchmark_results: Performance metrics proving it works
  * data_split_strategy: How data was split for testing

- INCORPORATE the selected method into your implementation
- You can adapt the implementation_code to fit the full dataset
- The method was already proven to work on a subset

IF NO TESTER OUTPUT:
- Proceed normally using methods from the Stage 3 plan

Your success = Executing the plan (using the benchmarked method if available) and saving results."""


# ===========================
# LangGraph
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """LLM agent step."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(STAGE4_TOOLS)


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
stage4_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 4 Runner
# ===========================

def run_stage4(
    plan_id: str,
    tester_output: Optional[TesterOutput] = None,
    max_rounds: int = STAGE4_MAX_ROUNDS,
    debug: bool = True,
) -> Dict:
    """Execute a Stage 3 plan.
    
    Args:
        plan_id: Plan ID from Stage 3 (e.g., 'PLAN-TSK-001')
        tester_output: Optional TesterOutput from Stage 3.5 with selected method
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    system_msg = SystemMessage(content=STAGE4_SYSTEM_PROMPT)

    tester_context = ""
    if tester_output:
        selected_method = tester_output.selected_method
        snippet = selected_method.implementation_code or ""
        # Keep the snippet concise for the message to avoid blowing context
        if len(snippet) > 1200:
            snippet = snippet[:600] + "\n...[truncated]...\n" + snippet[-400:]
        tester_context = (
            "\n\nTester output provided:\n"
            f"- Selected method ({selected_method.method_id}): {selected_method.name}\n"
            f"- Rationale: {tester_output.selection_rationale}\n"
            f"- Data split used: {tester_output.data_split_strategy}\n"
            "- Apply this method to the full dataset; reuse/adapt the implementation code.\n"
            f"Implementation snippet:\n{snippet}\n"
        )

    human_msg = HumanMessage(
        content=(
            f"Execute Stage 3 plan: '{plan_id}'\n\n"
            f"Workflow:\n"
            f"1. load_stage3_plan('{plan_id}')\n"
            f"2. Understand the plan structure\n"
            f"3. Execute each step using execute_python_code()\n"
            f"4. Verify results at each stage\n"
            f"5. save_execution_result() with final outputs\n\n"
            f"Be autonomous. Handle any issues. Follow the plan.\n"
            f"Your success = Plan executed + Results saved.{tester_context}"
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    if not debug:
        return stage4_app.invoke(
            state,
            config={
                "configurable": {"thread_id": f"stage4-{plan_id}"},
                "recursion_limit": max_rounds * 3,
            },
        )

    print("=" * 80)
    print(f"🚀 STAGE 4: Executing plan {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage4_app.stream(
        state,
        config={
            "configurable": {"thread_id": f"stage4-{plan_id}"},
            "recursion_limit": max_rounds * 3,
        },
        stream_mode="values",
    ):
        msgs = curr_state["messages"]
        new_msgs = msgs[prev_len:]

        for m in new_msgs:
            msg_type = m.__class__.__name__
            if "System" in msg_type:
                print("\n" + "─" * 80)
                print("💻 [SYSTEM]")
                print("─" * 80)
                print(m.content[:500] + "..." if len(m.content) > 500 else m.content)
            elif "Human" in msg_type:
                print("\n" + "─" * 80)
                print("👤 [USER]")
                print("─" * 80)
                print(m.content)
            elif "AI" in msg_type:
                round_num += 1
                print("\n" + "═" * 80)
                print(f"🤖 [AGENT - Round {round_num}]")
                print("═" * 80)
                if m.content:
                    print("\n💭 Reasoning:")
                    content = m.content
                    if len(content) > 1000:
                        print(content[:500] + "\n...[truncated]...\n" + content[-500:])
                    else:
                        print(content)
                
                if hasattr(m, 'tool_calls') and m.tool_calls:
                    print("\n🔧 Tool Calls:")
                    for tc in m.tool_calls:
                        name = tc.get("name", "UNKNOWN")
                        args = tc.get("args", {})
                        print(f"\n  📌 {name}")
                        for k, v in args.items():
                            if isinstance(v, str) and len(v) > 300:
                                print(f"     {k}: {v[:150]}...[truncated]...{v[-150:]}")
                            else:
                                print(f"     {k}: {v}")

        prev_len = len(msgs)
        final_state = curr_state
        
        if round_num >= max_rounds:
            print(f"\n⚠️  Reached max rounds ({max_rounds})")
            break

    print("\n" + "=" * 80)
    print(f"✅ Complete - {round_num} rounds")
    print("=" * 80)
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage4_node(state: dict) -> dict:
    """Stage 4 node for master graph.
    
    Args:
        state: Pipeline state with stage3_plan and optionally tester_output
        
    Returns:
        Updated state with execution_result
    """
    from .config import STAGE4_OUT_DIR
    from pathlib import Path
    
    stage3_plan = state.get("stage3_plan")
    if not stage3_plan:
        print("ERROR: No Stage 3 plan available for Stage 4")
        state["errors"].append("Stage 4: No Stage 3 plan available")
        return state
    
    plan_id = stage3_plan.plan_id
    tester_output = state.get("tester_output")
    
    # Check for prepared data from Stage 3B
    prepared_data = state.get("prepared_data")
    if prepared_data:
        print(f"\n✅ Stage 3B prepared data available: {prepared_data.prepared_file_path}")
        print(f"   Rows: {prepared_data.prepared_row_count}, Features: {len(prepared_data.columns_created)}")
        print(f"   Use load_dataframe('{Path(prepared_data.prepared_file_path).name}') to load it")
    else:
        print(f"\n⚠️  No prepared data from Stage 3B - agent will load raw data")
    
    print(f"\n⚙️  Starting Stage 4 for: {plan_id}\n")
    context_msg = f"\n🎯 Executing plan: {plan_id}\n"
    
    if tester_output:
        selected_method = tester_output.selected_method
        context_msg += f"\n📊 TESTER OUTPUT AVAILABLE:\n"
        context_msg += f"- Selected Method: {selected_method.name}\n"
        context_msg += f"- Rationale: {tester_output.selection_rationale}\n"
        context_msg += f"- Data Split Used: {tester_output.data_split_strategy}\n"
        context_msg += f"\n✅ USE THIS METHOD: {selected_method.name}\n"
        context_msg += f"Implementation guidance:\n{selected_method.implementation_code[:500]}...\n"
    else:
        context_msg += "\nℹ️  No tester output available - proceed with methods from Stage 3 plan\n"
    
    print(context_msg)
    
    result = run_stage4(plan_id, tester_output=tester_output, debug=True)
    
    # Check for execution results
    results = sorted(STAGE4_OUT_DIR.glob(f"execution_{plan_id}*.json"))
    if results:
        latest = results[-1]
        print(f"\n✅ Execution result: {latest}")
        result_data = json.loads(latest.read_text())
        state["execution_result"] = ExecutionResult.model_validate(result_data)
        state["completed_stages"].append(4)
        state["current_stage"] = 5
        print(f"  Status: {result_data.get('status', 'N/A')}")
        print(f"  Outputs: {len(result_data.get('outputs', {}))} ")
        if result_data.get('metrics'):
            print(f"  Metrics: {result_data['metrics']}")

        # Trigger failsafe if execution status indicates problems
        status = result_data.get("status", "").lower()
        if status and status != "success":
            try:
                rec = run_failsafe(
                    stage="stage4",
                    error=f"Execution status={status}",
                    context=result_data.get("summary", "")[:500],
                    debug=False,
                )
                state.setdefault("failsafe_history", []).append(rec)
                print(f"\n🛟 Failsafe suggestion recorded: {rec.analysis}")
            except Exception as e:
                print(f"\n⚠️  Failsafe agent failed: {e}")
    else:
        print("\n⚠️  No execution result saved")
        state["errors"].append("Stage 4: No execution result saved")
        
        try:
            rec = run_failsafe(
                stage="stage4",
                error="Execution result missing",
                context="save_execution_result() not called or failed validation.",
                debug=False,
            )
            state.setdefault("failsafe_history", []).append(rec)
            print(f"\n🛟 Failsafe suggestion recorded: {rec.analysis}")
        except Exception as e:
            print(f"\n⚠️  Failsafe agent failed: {e}")
    
    return state


if __name__ == "__main__":
    # Run Stage 4 standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage4_agent.py <plan_id>")
        print("Example: python stage4_agent.py PLAN-TSK-001")
        sys.exit(1)
    
    plan_id = sys.argv[1].strip()

    # Try to load the latest tester output for this plan_id
    latest_tester: Optional[TesterOutput] = None
    try:
        tester_files = sorted(STAGE3_5_OUT_DIR.glob(f"tester_{plan_id}*.json"))
        if tester_files:
            latest = tester_files[-1]
            latest_tester = TesterOutput.model_validate(json.loads(latest.read_text()))
            print(f"\n📥 Loaded tester output: {latest.name}")
        else:
            print(f"\nℹ️  No tester output found for {plan_id} in {STAGE3_5_OUT_DIR}")
    except Exception as e:
        print(f"\n⚠️  Could not load tester output automatically: {e}")
        latest_tester = None

    run_stage4(plan_id, tester_output=latest_tester)
