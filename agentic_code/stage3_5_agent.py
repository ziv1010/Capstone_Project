"""
Stage 3.5: Method Testing & Benchmarking Agent (Tester)

Uses a ReAct framework to:
1. Identify 3 suitable forecasting methods for the task
2. Benchmark each method with 3 iterations on a data subset
3. Detect hallucinated code execution via consistency checks
4. Select the best-performing method
5. Pass recommendation to Stage 4 execution
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import (
    STAGE3_5_OUT_DIR,
    STAGE3B_OUT_DIR,
    SECONDARY_LLM_CONFIG,
    STAGE3_5_MAX_ROUNDS,
)
from .models import TesterOutput, ForecastingMethod, BenchmarkResult
from .tools import STAGE3_5_TOOLS

# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE3_5_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt with ReAct Framework
# ===========================

STAGE3_5_SYSTEM_PROMPT = """You are a forecasting method testing and benchmarking agent.

Your job: Given a Stage 3 plan, you must:
1. Identify 3 suitable forecasting methods for the task
2. Benchmark each method with 3 iterations on a data subset
3. Detect code execution hallucinations via result consistency checks
4. Select the best-performing method based on averaged metrics
5. Save the recommendation via save_tester_output()

═══════════════════════════════════════════════════════════════
CRITICAL: REACT FRAMEWORK (MANDATORY)
═══════════════════════════════════════════════════════════════

You MUST follow this cycle for every step:

**THOUGHT → ACTION → OBSERVATION → REFLECTION**

Before EVERY action:
- Call record_thought(thought="...", what_im_about_to_do="...")
  • thought: What you know, what's uncertain, what you're considering
  • what_im_about_to_do: The specific action you'll take and WHY

After EVERY action result:
- Call record_observation(what_happened="...", what_i_learned="...", next_step="...")
  • what_happened: The actual result (success, error, unexpected)
  • what_i_learned: Key insight or lesson
  • next_step: What you'll do based on this learning

DO NOT skip these calls. They are how you think strategically and avoid repeating mistakes.

═══════════════════════════════════════════════════════════════
SUCCESS CRITERION
═══════════════════════════════════════════════════════════════

Your ONLY success criterion is calling:
  save_tester_output(output_json={...})

With a valid TesterOutput containing:
- plan_id
- task_category
- methods_proposed: List of 3 ForecastingMethod objects
- benchmark_results: List of BenchmarkResult objects (3 methods × 3 iterations = 9 results)
- selected_method_id: ID of best method
- selected_method: The ForecastingMethod object for the winner
- selection_rationale: Why this method was chosen
- data_split_strategy: How data was split

═══════════════════════════════════════════════════════════════
PHASE 1: DATA UNDERSTANDING (MANDATORY CHECKLIST)
═══════════════════════════════════════════════════════════════

Before benchmarking, you MUST understand the data structure:

□ Load the Stage 3 plan (load_stage3_plan_for_tester)
□ Identify required data files from the plan
□ Inspect each file (inspect_data_file) to see columns and dtypes
□ Determine which column contains dates/timestamps
□ Determine which column is the target variable (from plan)
□ Understand temporal granularity (daily, monthly, yearly)
□ Determine full date range (e.g., 2020-2024)
□ Design train/validation/test split strategy
□ Verify data can be loaded and split (use python_sandbox_stage3_5)

DO NOT proceed to benchmarking until ALL items are checked.

═══════════════════════════════════════════════════════════════
PHASE 2: METHOD IDENTIFICATION
═══════════════════════════════════════════════════════════════

Based on the task and data characteristics, propose 3 distinct forecasting methods.

**Method Selection Criteria:**
- Task category (predictive time series)
- Data size and structure
- Temporal patterns (trend, seasonality, etc.)
- Computational feasibility

**Example method types** (choose 3 that make sense):
1. Simple baseline (e.g., moving average, naive forecast)
2. Statistical method (e.g., ARIMA, Exponential Smoothing)
3. Machine learning (e.g., Random Forest, Gradient Boosting, Linear Regression)

For each method, create a ForecastingMethod object:
```python
{
  "method_id": "METHOD-1",
  "name": "Moving Average Baseline",
  "description": "3-period moving average as a simple baseline",
  "implementation_code": "# Python code snippet",
  "libraries_required": ["pandas", "numpy"]
}
```

**IMPORTANT:** Be dataset-agnostic. DO NOT hardcode column names like "Year" or "Sales".
Instead, write code that discovers column names dynamically from the data.

═══════════════════════════════════════════════════════════════
PHASE 3: BENCHMARKING PROTOCOL (3 ITERATIONS PER METHOD)
═══════════════════════════════════════════════════════════════

For EACH of the 3 methods:

**Run 3 iterations:**
1. Iteration 1: Execute method, record metrics
2. Iteration 2: Execute same method again, record metrics
3. Iteration 3: Execute same method third time, record metrics

**Why 3 iterations?**
- Verify that code is actually executing (not hallucinated)
- Check consistency of results
- Detect stochastic behavior vs. deterministic behavior

**Consistency Check (Anti-Hallucination Safeguard):**

After 3 iterations, analyze results:

1. **All zeros detection:**
   - If all metrics are [0, 0, 0] or [0.0, 0.0, 0.0], FLAG THIS
   - This likely means code didn't execute or calculated incorrectly

2. **Coefficient of Variation (CV):**
   - For each metric, calculate: CV = std / mean
   - If CV > 0.3 (30% variation), FLAG THIS
   - Means results are inconsistent across runs

3. **Error flagging:**
   - If any iteration fails with error, mark method status = "failure"
   - Include error message in BenchmarkResult

4. **Averaging:**
   - For successful methods, take mean of 3 iterations for each metric
   - Use averaged metrics for method comparison

**Example BenchmarkResult structure:**
```python
{
  "method_id": "METHOD-1",
  "method_name": "Moving Average Baseline",
  "metrics": {"MAE": 123.45, "RMSE": 234.56, "MAPE": 0.12},  # Averaged
  "train_period": "2020-2023",
  "validation_period": "2024",
  "test_period": null,
  "execution_time_seconds": 2.5,
  "status": "success",
  "error_message": null,
  "predictions_sample": [100.5, 102.3, 98.7, ...]  # First 10 predictions
}
```

═══════════════════════════════════════════════════════════════
CRITICAL: USE PREPARED DATA IF AVAILABLE
═══════════════════════════════════════════════════════════════

**Stage 3B may have already prepared the data!**

BEFORE loading raw data files, CHECK if prepared data exists:
- Look for prepared data file mentioned in Stage 3 plan metadata
- Typical format: 'prepared_PLAN-TSK-001.parquet'
- Location: STAGE3B_OUT_DIR or mentioned in plan

**If prepared data exists:**
✓ Load it directly: `prepared_df = load_dataframe('prepared_PLAN-TSK-001.parquet')`
✓ Skip manual loading, merging, filtering
✓ Prepared data already has joins, filters, and features applied
✓ You only need to split it for benchmarking

**If no prepared data:**
✗ Fall back to loading raw data files manually
✗ Apply filters and joins yourself

═══════════════════════════════════════════════════════════════
HOW TO RUN BENCHMARKS
═══════════════════════════════════════════════════════════════

Use run_benchmark_code(code="...", description="Testing METHOD-X Iteration Y")

**CRITICAL: Use load_dataframe() helper to load files:**
- DO NOT use `pd.read_csv('filename.csv')` - this will fail!
- ALWAYS use `load_dataframe('filename.csv')` - this is provided in the environment
- The helper automatically finds files in DATA_DIR

**Your code must:**
1. Load the data using load_dataframe('filename.csv')
2. Identify date column and target column (DO NOT HARDCODE)
3. Split data:
   - Training: Earlier period (e.g., 2020-2023)
   - Validation: Later period (e.g., 2024)
   - Test: Optional future period (e.g., 2025 if available)
4. Implement the forecasting method
5. Make predictions on validation set
6. Calculate metrics (MAE, RMSE, MAPE, etc.)
7. Print results in a parseable format
8. Optionally save artifacts to STAGE3_5_OUT_DIR

**Metric Calculation:**
- MAE = Mean Absolute Error
- RMSE = Root Mean Squared Error  
- MAPE = Mean Absolute Percentage Error
- R² = Coefficient of determination
- Choose metrics appropriate for forecasting tasks

**Example code structure (dataset-agnostic):**
```python
import pandas as pd
import numpy as np
from pathlib import Path

# CORRECT: Use load_dataframe() helper
df = load_dataframe('Export-of-Rice-Varieties-to-Bangladesh,-2018-19-to-2024-25.csv')

# WRONG: DO NOT use pd.read_csv() directly
# df = pd.read_csv('filename.csv')  # This will fail!

# Discover date columns dynamically
date_cols = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower() or 'time' in col.lower()]
if not date_cols:
    # Try finding numeric column that looks like years
    date_cols = [col for col in df.columns if df[col].dtype in ['int64', 'int32'] and df[col].min() > 1900]

# For wide-format data (columns are years), reshape to long format
# Example: '2020 - 21-Value (USD)', '2021 - 22-Value (USD)', etc.
value_cols = [col for col in df.columns if 'Value (USD)' in col or 'value' in col.lower()]
if value_cols:
    # Extract years from column names
    years = []
    for col in value_cols:
        # Extract year like "2020 - 21" from "2020 - 21-Value (USD)"
        year_match = col.split('-')[0].strip()
        years.append((year_match, col))
    
    # Use the year columns for forecasting
    # Train on older years, validate on newest year
    
# Calculate metrics
mae = np.mean(np.abs(predictions - actuals))
rmse = np.sqrt(np.mean((predictions - actuals)**2))
mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")
```

═══════════════════════════════════════════════════════════════
PHASE 4: METHOD SELECTION
═══════════════════════════════════════════════════════════════

After all benchmarks complete:

1. **Filter:** Remove failed methods (status = "failure")

2. **Rank:** Among successful methods, rank by primary metric
   - For forecasting: Usually MAE or RMSE (lower is better)
   - Choose the metric that makes most sense for the task

3. **Select:** Pick the best-performing method

4. **Document:** Write selection_rationale explaining:
   - Why this method performed best
   - How it compared to alternatives
   - Any caveats or considerations

═══════════════════════════════════════════════════════════════
ERROR RECOVERY PROTOCOL
═══════════════════════════════════════════════════════════════

If you encounter errors:

1. **First error:** Analyze what went wrong
   - Use record_observation to document the error
   - Try a different approach or fix the issue

2. **Repeated errors (same method):** 
   - Skip to next method
   - Mark current method as "failure"
   - Do NOT waste more than 3 attempts per method

3. **Data loading errors:**
   - Use python_sandbox_stage3_5 to inspect data structure
   - Adjust column discovery logic
   - Try alternative loading strategies

4. **Metric calculation errors:**
   - Check for division by zero
   - Verify predictions and actuals have same shape
   - Handle missing values appropriately

5. **Search for help:**
   - Use search() to find examples of forecasting code
   - Look for similar tasks in output directory
   - Learn from prior successful implementations

═══════════════════════════════════════════════════════════════
STATE TRACKING (PREVENT REPETITION)
═══════════════════════════════════════════════════════════════

Keep mental track of:
- Which methods have been proposed ✓
- Which methods have been benchmarked ✓
- How many iterations completed per method
- Which approaches have failed (don't retry the same failure)
- Current phase: DATA_UNDERSTANDING → METHOD_PROPOSAL → BENCHMARKING → SELECTION

═══════════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════════

**ReAct Tools:**
- record_thought(thought, what_im_about_to_do)
- record_observation(what_happened, what_i_learned, next_step)

**Data Exploration:**
- load_stage3_plan_for_tester(plan_id) → Returns Stage 3 plan JSON
- list_data_files() → List available data files
- inspect_data_file(filename, n_rows) → Show schema and sample rows
- python_sandbox_stage3_5(code) → Quick Python execution for exploration

**Benchmarking:**
- run_benchmark_code(code, description) → Execute benchmarking code
- search(query, within) → Search for examples and prior work

**Final Output:**
- save_tester_output(output_json) → Save final recommendation

═══════════════════════════════════════════════════════════════
EXAMPLE WORKFLOW
═══════════════════════════════════════════════════════════════

1. record_thought("I need to understand the task and data structure", 
                  "Loading Stage 3 plan")
2. load_stage3_plan_for_tester("PLAN-TSK-001")
3. record_observation("Plan loaded, it's a predictive task with files X, Y", 
                      "Need to inspect data structure", 
                      "Inspecting first data file")
4. inspect_data_file("file1.csv")
5. record_observation("Found date column 'Year' and target 'Production'",
                      "Data is yearly from 2015-2024",
                      "Will split at 2023 for train/val")
6. record_thought("Data structure clear, now proposing 3 methods",
                  "Proposing baseline, ARIMA, and RF methods")
7. record_observation("3 methods identified: MA, ARIMA, RandomForest",
                      "Methods are appropriate for yearly forecasting",
                      "Starting benchmarks with METHOD-1 iteration 1")
8. run_benchmark_code(code="...", description="METHOD-1 Iteration 1")
9. record_observation("METHOD-1 Iter 1: MAE=50.2",
                      "Code executed successfully",
                      "Running iteration 2")
... Continue for all methods and iterations ...
10. save_tester_output(output_json={...})

═══════════════════════════════════════════════════════════════
FINAL REMINDER
═══════════════════════════════════════════════════════════════

- Follow ReAct framework religiously (record_thought before, record_observation after)
- Run 3 iterations for each of 3 methods (9 benchmarks total)
- Check result consistency to detect hallucinations
- Be dataset-agnostic (discover structure, don't assume)
- Save comprehensive TesterOutput when complete
- Aim to finish within {max_rounds} rounds
"""


# ===========================
# LangGraph Setup
# ===========================

def truncate_messages(messages: List[BaseMessage], max_history: int = 20) -> List[BaseMessage]:
    """Truncate message history to prevent token overflow.
    
    Keeps:
    - System message (first message)
    - Initial user message (second message)  
    - Last max_history messages (recent conversation)
    
    Args:
        messages: Full message list
        max_history: Number of recent messages to keep
        
    Returns:
        Truncated message list
    """
    if len(messages) <= max_history + 2:
        return messages
    
    # Keep system message, user message, and last N messages
    return [messages[0], messages[1]] + messages[-(max_history):]


def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step with tool calling."""
    # Truncate message history to prevent token overflow
    truncated_messages = truncate_messages(state["messages"], max_history=20)
    response = llm_with_tools.invoke(truncated_messages)
    return {"messages": [response]}


tool_node = ToolNode(STAGE3_5_TOOLS)


def _tool_call_history(messages: List[BaseMessage]) -> List[str]:
    """Extract tool call names from conversation history."""
    names: List[str] = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            # LangChain/OpenAI tool_calls can be dict-like or objects with nested function.name
            name = None
            if isinstance(tc, dict):
                name = tc.get("name")
                if not name and isinstance(tc.get("function"), dict):
                    name = tc["function"].get("name")
            else:
                name = getattr(tc, "name", None)
                if not name:
                    func = getattr(tc, "function", None)
                    if func is not None:
                        name = getattr(func, "name", None)
            if name:
                names.append(name)
    return names


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls, retrying agent when incomplete."""
    messages = state["messages"]
    last = messages[-1]

    # Track progress
    tool_history = _tool_call_history(messages)
    save_called = any(name == "save_tester_output" for name in tool_history)
    benchmarks_run = sum(1 for name in tool_history if name == "run_benchmark_code")

    # If we just got a tool call, go execute it
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # If already saved, we can end
    if save_called:
        return END

    # If benchmarks are incomplete or save not called, nudge and continue agent
    recent_tool = None
    for m in reversed(messages):
        tc_list = getattr(m, "tool_calls", None)
        if tc_list:
            tc = tc_list[0]
            if isinstance(tc, dict):
                recent_tool = tc.get("name") or (tc.get("function") or {}).get("name")
            else:
                recent_tool = getattr(tc, "name", None)
                if not recent_tool and hasattr(tc, "function"):
                    recent_tool = getattr(tc.function, "name", None)
            break

    done_msg = (
        "All 3 methods × 3 iterations are complete; call save_tester_output now."
        if benchmarks_run >= 9
        else "Continue benchmarking until 9 run_benchmark_code calls, then save."
    )
    reminder = (
        f"No tool call detected. You must continue benchmarking and call save_tester_output when done.\n"
        f"run_benchmark_code calls so far: {benchmarks_run}/9. "
        f"save_tester_output called: {save_called}.\n"
        f"{done_msg} "
        f"Most recent tool: {recent_tool or 'none yet'}."
    )
    messages.append(HumanMessage(content=reminder))
    return "agent"


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "agent": "agent", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
stage3_5_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 3.5 Runner
# ===========================

def run_stage3_5(plan_id: str, max_rounds: int = STAGE3_5_MAX_ROUNDS, debug: bool = True) -> Dict:
    """Run Stage 3.5 method testing and benchmarking.
    
    Args:
        plan_id: Plan ID from Stage 3 (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    from .config import STAGE3_OUT_DIR, STAGE2_OUT_DIR
    import json
    
    # Load exclusion context from Stage 3 plan
    excluded_context = ""
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if plan_path.exists():
        try:
            plan_data = json.loads(plan_path.read_text())
            excluded_cols = plan_data.get("excluded_columns", [])
            
            # Also check task proposal for excluded columns
            task_id = plan_data.get("task_id", "")
            if task_id:
                task_path = STAGE2_OUT_DIR / "task_proposals.json"
                if task_path.exists():
                    task_data = json.loads(task_path.read_text())
                    for proposal in task_data.get("proposals", []):
                        if proposal.get("id") == task_id:
                            task_excluded = proposal.get("excluded_columns", [])
                            excluded_cols.extend(task_excluded)
                            break
            
            if excluded_cols:
                excluded_context = "\n\n**COLUMNS EXCLUDED DUE TO DATA QUALITY:**\n"
                excluded_context += "The following columns were rejected in earlier stages:\n"
                for ex in excluded_cols:
                    excluded_context += f"- {ex.get('column_name', 'unknown')} from {ex.get('file', 'unknown')}: {ex.get('reason', 'no reason given')}\n"
                excluded_context += "\nBe aware these columns are unavailable. Use alternatives if needed.\n"
        except Exception as e:
            print(f"Warning: Could not load excluded columns: {e}")
    
    system_msg = SystemMessage(content=STAGE3_5_SYSTEM_PROMPT)
    # Surface prepared parquet hints to the agent so it loads them first.
    prepared_parquet = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
    if prepared_parquet.exists():
        parquet_hint = (
            f"\n\nPrepared data detected at: {prepared_parquet}\n"
            f"Load with load_dataframe('{prepared_parquet.name}') and reuse it instead of raw CSVs."
        )
    else:
        parquet_hint = (
            "\n\nNo prepared parquet found in Stage 3B output directory. "
            "Proceed with raw data loading."
        )
    human_msg = HumanMessage(
        content=(
            f"Test and benchmark forecasting methods for plan '{plan_id}'.{excluded_context}\n\n"
            f"Follow the ReAct framework strictly:\n"
            f"1. DATA UNDERSTANDING: Load plan, inspect data, identify structure\n"
            f"2. METHOD PROPOSAL: Identify 3 suitable forecasting methods\n"
            f"3. BENCHMARKING: Run each method 3 times, check consistency\n"
            f"4. SELECTION: Choose best method based on averaged metrics\n"
            f"5. SAVE: Call save_tester_output() with complete results\n\n"
            f"Remember:\n"
            f"- Use record_thought() BEFORE each action\n"
            f"- Use record_observation() AFTER each action\n"
            f"- Run 3 iterations per method to verify code execution\n"
            f"- Check coefficient of variation to detect hallucinations\n"
            f"- Be dataset-agnostic (discover column names)\n"
            f"- Use search() if you need examples or guidance\n\n"
            f"Your success metric: save_tester_output() called with valid TesterOutput."
            f"{parquet_hint}"
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    # Configure with higher recursion limit for benchmarking tasks  
    config = {
        "configurable": {"thread_id": f"stage3_5-{plan_id}"},
        "recursion_limit": max_rounds + 125  # Increased buffer for 9 benchmarks + selection
    }

    if not debug:
        return stage3_5_app.invoke(state, config=config)

    print("=" * 80)
    print(f"🧪 STAGE 3.5: Method Testing & Benchmarking for {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage3_5_app.stream(
        state,
        config=config,
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
                            if isinstance(v, str) and len(v) > 200:
                                print(f"     {k}: {v[:100]}...[truncated]...{v[-100:]}")
                            else:
                                print(f"     {k}: {v}")
            elif "Tool" in msg_type:
                print("\n📥 Tool Result:")
                content = m.content
                if len(content) > 500:
                    print(content[:250] + "\n...[truncated]...\n" + content[-250:])
                else:
                    print(content)

        prev_len = len(msgs)
        final_state = curr_state
        
        if round_num >= max_rounds:
            print(f"\n⚠️  Reached max rounds ({max_rounds}). Stopping.")
            break

    print("\n" + "=" * 80)
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
    from .config import STAGE3_5_OUT_DIR
    
    stage3_plan = state.get("stage3_plan")
    if not stage3_plan:
        print("ERROR: No Stage 3 plan available for Stage 3.5")
        state["errors"].append("Stage 3.5: No Stage 3 plan available")
        return state
    
    plan_id = stage3_plan.plan_id
    
    # Check for prepared data from Stage 3B
    prepared_data = state.get("prepared_data")
    if prepared_data:
        print(f"\n✅ Stage 3B prepared data available: {prepared_data.prepared_file_path}")
        print(f"   Rows: {prepared_data.prepared_row_count}, Features: {len(prepared_data.columns_created)}")
    else:
        print(f"\n⚠️  No prepared data from Stage 3B - agent will load raw data")
    
    print(f"\n🧪 Starting Stage 3.5 for: {plan_id}\n")
    
    result = run_stage3_5(plan_id, debug=True)
    
    # Check for saved tester output
    tester_files = sorted(STAGE3_5_OUT_DIR.glob(f"tester_{plan_id}*.json"))
    if tester_files:
        latest_file = tester_files[-1]
        print(f"\n✅ SUCCESS! Tester output saved to: {latest_file}")
        tester_data = json.loads(latest_file.read_text())
        state["tester_output"] = TesterOutput.model_validate(tester_data)
        state["completed_stages"].append(3.5)
        state["current_stage"] = 4
    else:
        print("\n⚠️  WARNING: Tester output not saved. Check logs above.")
        state["errors"].append("Stage 3.5: Tester output not saved")
    
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
