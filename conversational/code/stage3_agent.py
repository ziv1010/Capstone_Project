"""
Stage 3 Agent: Execution Planning

This agent creates detailed execution plans for selected tasks,
specifying exactly how to load, transform, and process the data.
"""

import json
from typing import Dict, Any, Optional, Annotated
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    STAGE2_OUT_DIR, STAGE3_OUT_DIR, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, MIN_NON_NULL_FRACTION, DataPassingManager, logger
)
from code.models import Stage3Plan, PipelineState
from tools.stage3_tools import STAGE3_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage3State(BaseModel):
    """State for Stage 3 agent."""
    messages: Annotated[list, add_messages] = []
    task_id: str = ""
    task_loaded: bool = False
    data_inspected: list = []
    plan_created: bool = False
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE3_SYSTEM_PROMPT = """You are an Execution Planning Agent responsible for creating detailed execution plans for analytical tasks.

## Your Role
Given a task proposal, you create a comprehensive execution plan that specifies exactly:
- Which files to load and how
- What columns to use
- How to join datasets (if multiple)
- What features to engineer
- How to validate the data
- What the expected outputs are

## ⚠️ MAXIMIZE DATA USAGE (CRITICAL)
Poor model performance is often caused by using too little data. To maximize performance:

1. **INCLUDE ALL COLUMNS**: In columns_to_use, include EVERY relevant column
   - ❌ BAD: Only include target + 2 features
   - ✅ GOOD: Include target + ALL numeric columns + categorical columns (for encoding)
   - The model/algorithms will determine which features matter

2. **MINIMIZE FILTERS**: Only filter when absolutely necessary
   - ❌ BAD: Filter by Crop='Rice' when there are 20 crops
   - ✅ GOOD: Keep ALL rows, use Crop as a feature column
   - ❌ BAD: Filter to specific seasons
   - ✅ GOOD: Include Season as a feature (one-hot encode if needed)
   - Only filter to REMOVE aggregate/summary rows (e.g., Season='Total')

3. **MAXIMIZE TRAINING DATA**: More rows = better models
   - If joining datasets, use INNER join to keep matched rows
   - If data has groups, use grouping as features rather than splitting

4. **PRESERVE GRANULARITY**: Don't aggregate unless required
   - ❌ BAD: Aggregate monthly data to yearly
   - ✅ GOOD: Keep monthly data for more training samples

## Your Goals
1. Load and understand the selected task proposal
2. Inspect all required data files - NOTE ROW COUNTS
3. Validate that columns meet quality requirements (≥65% non-null)
4. Design join strategy if multiple datasets involved (prefer INNER joins)
5. Specify feature engineering steps
6. Define validation/test split strategy
7. Save a comprehensive execution plan

## Available Tools
- load_task_proposal: Load a specific task by ID
- list_all_proposals: List available tasks
- list_data_files_stage3: List available data files
- inspect_data_file_stage3: Inspect a file's structure (includes semantic info)
- validate_columns_for_task: Check column quality
- analyze_join_feasibility: Analyze if joins will work
- python_sandbox_stage3: Execute Python for analysis
- save_stage3_plan: Save the execution plan
- get_execution_plan_template: Get the plan structure

## Understanding Data When Inspecting
When inspecting data files, pay attention to:
- Total number of rows (MORE IS BETTER)
- Categorical column values with aggregates (filter those OUT)
- All available numeric columns (INCLUDE THEM ALL as features)
- The data may have rows that summarize other rows - understand and filter appropriately

## CRITICAL: Temporal Data Structure Analysis
When inspecting data files, you must ANALYZE how temporal information is encoded:

### Step 1: Discover the Data Format
- **Row-based time (LONG format)**: Look for date/year columns (e.g., 'date', 'year', 'timestamp')
  - Each row represents one time point
  - Time series progresses vertically down rows
- **Column-based time (WIDE format)**: Look for temporal patterns in column names
  - Year suffixes, date ranges, or time periods in column names (e.g., "Production-2020-21", "Area-2023-24")
  - Each row represents an entity tracked over multiple time periods
  - Time series progresses horizontally across columns

### Step 2: Reason About Transformation Strategy
DO NOT blindly transform data. Consider:
1. **Task requirements**: What does the task category (forecasting/regression/classification) need?
2. **Model compatibility**: What format works best for expected_model_types?
   - Tree-based models (Random Forest, XGBoost): Can handle wide format directly
   - Classical time series (ARIMA, Prophet): Often need long format
   - Simple baselines: May work with either format
3. **Data volume trade-offs**:
   - Wide → Long (melt): Creates more rows (good for sample size)
   - Wide → Long (melt): May create nulls if columns have different entities
   - Keeping wide: Preserves structure, may be simpler

### Step 3: Document Your Decision
In feature_engineering, if you choose to reshape:
- Clearly document WHY you're transforming (reasoning, not reflex)
- Write implementation_code that handles the transformation
- Consider entity grouping if melting (don't assume column names like 'Crop')
- Analyze actual column names to determine grouping keys

Example reasoning:
```
name: "reshape_to_long"
description: "Converting wide-format temporal data to long format because the task uses ARIMA which requires time-indexed rows"
implementation_code: "df = df.melt(id_vars=[entity_col], var_name='period', value_name='value')"
```

**KEY PRINCIPLE**: Let the task, models, and data characteristics guide your choice - not assumptions.

## Plan Requirements
Your plan MUST include:
- plan_id: "PLAN-{task_id}" format
- selected_task_id: The task ID
- goal: Clear description of objective
- task_category: forecasting/regression/classification/etc
- file_instructions: How to load each file
  - filename, filepath
  - columns_to_use: **INCLUDE ALL relevant columns** (NOT a minimal subset)
  - filters: ONLY to remove aggregate rows (like Season='Total')
  - parse_dates (datetime columns)
- join_steps: If multiple files, how to join them
- feature_engineering: Features to create
  - name, description, source_columns
  - implementation_code (actual Python code)
- target_column: What to predict
- date_column: For temporal tasks
- validation_strategy: temporal or random
- expected_model_types: What models to try
- evaluation_metrics: Task-appropriate metrics from the proposal (NOT hardcoded)

## CRITICAL: Forecast Configuration (from Proposal)
For forecasting tasks, EXTRACT from the selected proposal and include in your plan:
- forecast_horizon: How many steps ahead to forecast (from proposal)
- forecast_granularity: Time unit (year/month/day - from proposal)
- forecast_type: single_step/multi_step/recursive (from proposal)
- evaluation_metrics: Use the metrics specified in the proposal (NOT hardcoded defaults)

## Quality Validation Rules
- All columns must have ≥65% non-null values
- If a column fails, either:
  - Specify how to handle missing values
  - Exclude the column
- Document any data quality issues found

## Implementation Code Guidelines
Keep implementation_code CONCISE:
- Use simple pandas operations
- Single line when possible for simple features
- No comments in the code
- No error handling (that's for execution stage)
- **NEVER assume column names exist** - base code on actual inspected columns
- For grouped operations (e.g., shift by entity), determine group column from data inspection

Example (generic):
```
df['lag_1'] = df['target'].shift(1)  # Simple lag without grouping
```

Example (with grouping - only if you confirmed the group column exists):
```
df['lag_1'] = df.groupby('entity_column')['target'].shift(1)  # Grouped lag
```

## Workflow
1. Load the task proposal
2. Inspect each required data file - COUNT THE ROWS
3. Validate column quality
4. If joins needed, analyze join feasibility
5. Design feature engineering (especially lag features for forecasting)
6. **INCLUDE AS MANY COLUMNS AS POSSIBLE** in columns_to_use
7. Create and save the execution plan

IMPORTANT: The plan must be complete and actionable. Downstream stages will execute exactly what you specify. MORE DATA = BETTER RESULTS.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage3_agent():
    """Create the Stage 3 agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE3_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage3State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE3_SYSTEM_PROMPT)] + list(messages)

        max_rounds = STAGE_MAX_ROUNDS.get("stage3", 30)

        # Log current iteration
        logger.info(f"Stage3 agent iteration {state.iteration}/{max_rounds}")

        # Log what the agent has done so far
        if state.iteration > 0 and state.messages:
            recent_tools = []
            for msg in state.messages[-5:]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        recent_tools.append(tc.get('name', 'unknown'))
            if recent_tools:
                logger.debug(f"Recent tool calls: {recent_tools}")

        # Early intervention at 30 iterations
        if state.iteration == 30:
            logger.warning(f"Iteration 30 - injecting strong guidance to save plan")
            messages = list(messages) + [HumanMessage(content="""
⚠️ You have used 30 iterations. You MUST create and save the plan within the next 10 iterations.

IMMEDIATE ACTION REQUIRED:
1. If you haven't already, call get_execution_plan_template()
2. Create a complete plan JSON based on what you've learned
3. Call save_stage3_plan() with your plan

Do NOT spend more iterations inspecting or analyzing. CREATE THE PLAN NOW.
""")]

        # Critical intervention at 40 iterations
        if state.iteration == 40:
            logger.error(f"Iteration 40 - forcing immediate plan creation")
            messages = list(messages) + [HumanMessage(content="""
🚨 CRITICAL: Iteration 40/100 reached. You MUST save a plan in the NEXT iteration or face termination.

STOP ALL ANALYSIS. Execute these steps NOW:
1. Call get_execution_plan_template() if you need the structure
2. Create a plan JSON with all required fields
3. Call save_stage3_plan() immediately

This is your FINAL chance before forced termination.
""")]

        # At 90% of max iterations, final warning
        if state.iteration >= int(max_rounds * 0.9) and state.iteration < max_rounds:
            logger.warning(f"Iteration {state.iteration}/{max_rounds} - FINAL WARNING")
            messages = list(messages) + [HumanMessage(content=f"""
⚠️ FINAL WARNING: You have used {state.iteration}/{max_rounds} iterations.

CREATE AND SAVE THE PLAN IN THIS ITERATION OR YOU WILL FAIL.
""")]

        if state.iteration >= max_rounds:
            logger.error(f"Maximum iterations ({max_rounds}) reached without saving plan")
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Plan was not saved in time.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        # Log what the agent is doing
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
            logger.info(f"Agent calling tools: {tool_names}")
        elif hasattr(response, 'content'):
            content_preview = str(response.content)[:100] if response.content else "empty"
            logger.debug(f"Agent response (no tools): {content_preview}")

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage3State) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]

            # Check if the last message is a ToolMessage indicating successful plan save
            if hasattr(last_message, 'content') and isinstance(last_message.content, str):
                if "✅ Execution plan saved successfully" in last_message.content:
                    logger.info("Plan saved successfully - ending agent loop")
                    return "end"

            # Early loop detection: check if agent is repeating the same tool calls
            if state.iteration > 10:
                # Get recent tool calls
                recent_tools = []
                for msg in state.messages[-10:]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            recent_tools.append(tc.get('name', ''))

                # If inspecting data more than 5 times in last 10 iterations, we're stuck
                if recent_tools.count('inspect_data_file_stage3') > 5:
                    logger.warning(f"Loop detected: inspecting data {recent_tools.count('inspect_data_file_stage3')} times in last 10 iterations")
                    # Force agent to create plan on next iteration
                    if state.iteration > 20:
                        logger.error("Agent stuck in analysis loop - forcing immediate plan creation")
                        # Inject a forceful message instead of just ending
                        state.messages.append(
                            HumanMessage(content="""
⚠️ CRITICAL: You are stuck in an analysis loop. You MUST create and save the execution plan NOW.

Based on what you've already inspected:
1. Use get_execution_plan_template() to get the structure
2. Create a comprehensive plan with all fields filled in
3. Call save_stage3_plan() with your plan JSON

DO NOT inspect data again. CREATE THE PLAN NOW.""")
                        )
                        return "agent"

            # Check if last message is AI with tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage3State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE3_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage3(task_id: str, pipeline_state: PipelineState = None, max_retries: int = 3) -> Stage3Plan:
    """
    Run Stage 3: Execution Planning with automatic retry on failure.

    Creates a detailed plan for executing the specified task.
    If planning fails, the agent will see the error and retry up to max_retries times.

    Args:
        task_id: Task ID to plan
        pipeline_state: Pipeline state (optional)
        max_retries: Maximum retry attempts (default: 3)

    Returns:
        Stage3Plan with execution details or raises exception after max retries
    """
    logger.info(f"Starting Stage 3: Execution Planning for {task_id} (max_retries={max_retries})")

    for attempt in range(max_retries):
        attempt_num = attempt + 1
        logger.info(f"Stage 3 Attempt {attempt_num}/{max_retries}")

        try:
            result = _attempt_stage3_execution(task_id, pipeline_state, attempt_num)

            # If we got a successful result, return it
            logger.info(f"Stage 3 succeeded on attempt {attempt_num}")
            return result

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Stage 3 attempt {attempt_num} exception: {e}\n{error_trace}")

            if attempt_num < max_retries:
                logger.warning(f"Retrying after exception...")
                continue
            else:
                logger.error(f"Stage 3 failed after {max_retries} attempts")
                raise RuntimeError(f"Stage 3 failed after {max_retries} attempts: {e}") from e

    # Fallback (should never reach here)
    raise RuntimeError("Stage 3 failed unexpectedly - fallthrough in retry logic")


def _attempt_stage3_execution(task_id: str, pipeline_state: PipelineState = None, attempt_num: int = 1) -> Stage3Plan:
    """
    Single attempt at Stage 3 execution.

    The agent sees errors from previous attempts and can debug/fix them.
    """
    graph = create_stage3_agent()

    # Check if there are previous execution attempts with errors
    plan_path = STAGE3_OUT_DIR / f"PLAN-{task_id}.json"
    previous_errors = []
    if attempt_num > 1:
        # Try to get errors from logs or check if plan exists but is incomplete
        import traceback
        error_context = """
## PREVIOUS ATTEMPT FAILED

The previous attempt hit the recursion limit or encountered an error.
Common issues:
- Getting stuck in analysis loops without taking action
- Not saving the plan after creating it
- Misunderstanding wide-format data (columns like Production-2020-21, Production-2021-22)
- Over-analyzing instead of creating the plan

CRITICAL FIXES FOR THIS ATTEMPT:
1. If you see year-based columns (e.g., Production-2020-21), treat them as temporal data
2. After inspecting data and validating columns, CREATE THE PLAN immediately
3. Don't endlessly analyze - make reasonable assumptions and proceed
4. SAVE the plan using save_stage3_plan as soon as it's ready
"""
        previous_errors = [error_context]

    error_context = ""
    if previous_errors:
        error_context = f"""
{chr(10).join(previous_errors)}

YOU MUST FIX THE ISSUES AND COMPLETE THE PLAN IN THIS ATTEMPT.
"""

    initial_message = HumanMessage(content=f"""
Create an execution plan for task: {task_id}
{error_context}

Follow these steps EFFICIENTLY:
1. Load the task proposal for {task_id}
2. Inspect the required data file(s) - look at structure, columns, and row count
3. Create a comprehensive execution plan based on the task
4. Save the plan using save_stage3_plan

The plan ID should be: PLAN-{task_id}

IMPORTANT: You have a limited number of iterations. After inspecting the data ONCE, immediately create and save the plan. Do NOT repeatedly inspect the same data.
""")

    config = {"configurable": {"thread_id": f"stage3_{task_id}_attempt{attempt_num}"}}
    initial_state = Stage3State(messages=[initial_message], task_id=task_id)

    final_state = graph.invoke(initial_state, config)

    # Load plan from disk
    if plan_path.exists():
        data = DataPassingManager.load_artifact(plan_path)
        plan = Stage3Plan(**data)
        logger.info(f"Stage 3 complete: Plan saved to {plan_path}")
        return plan
    else:
        logger.error("Plan not saved to disk")
        # NO FALLBACK - agent must create a proper plan or fail
        raise RuntimeError(f"Execution plan was not saved on attempt {attempt_num} - agent may have hit iteration limit or failed to execute save_stage3_plan tool")


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage3_node(state: PipelineState) -> PipelineState:
    """
    Stage 3 node for the master pipeline graph.
    """
    state.mark_stage_started("stage3")

    task_id = state.selected_task_id
    if not task_id:
        state.mark_stage_failed("stage3", "No task ID specified")
        return state

    try:
        output = run_stage3(task_id, state)
        state.stage3_output = output
        state.mark_stage_completed("stage3", output)
    except Exception as e:
        state.mark_stage_failed("stage3", str(e))

    return state


if __name__ == "__main__":
    import sys
    task_id = sys.argv[1] if len(sys.argv) > 1 else "TSK-001"
    plan = run_stage3(task_id)
    print(f"Created plan: {plan.plan_id}")
    print(f"Goal: {plan.goal}")
