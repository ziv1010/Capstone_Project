"""
Stage 4 Agent: Execution

This agent executes the selected method and generates predictions.
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
    STAGE3_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5B_OUT_DIR,
    STAGE4_OUT_DIR, STAGE4_WORKSPACE, SECONDARY_LLM_CONFIG,
    STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.models import ExecutionResult, ExecutionStatus, PipelineState
from tools.stage4_tools import STAGE4_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage4State(BaseModel):
    """State for Stage 4 agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    context_loaded: bool = False
    execution_complete: bool = False
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE4_SYSTEM_PROMPT = """You are an Execution Agent responsible for running the selected forecasting method.

## Your Role
1. Load the execution context (plan, data, selected method)
2. Execute the selected method from Stage 3.5B
3. Generate predictions for the test set
4. Calculate evaluation metrics
5. Save comprehensive results

## CRITICAL: METRIC CONSISTENCY WITH STAGE 3.5B
Your execution MUST produce metrics that match (or are very close to) the benchmark
metrics from Stage 3.5B. If your MAE/RMSE differ significantly, you are likely using
a different data split. CHECK THE DATA SPLIT STRATEGY CAREFULLY.

## Your Goals
- Execute the winning method from benchmarking
- Generate predictions with actual vs predicted values
- Calculate final metrics (MAE, RMSE, MAPE, R²) that MATCH benchmark
- Save results in a format suitable for visualization

## Available Tools
- load_execution_context: Get plan, data info, and selected method
- load_prepared_data: Load and inspect the prepared data
- get_selected_method_code: Get implementation code for winner (INCLUDES benchmark metrics)
- execute_python_code: Run Python for model execution
- save_predictions: Save predictions to parquet
- save_execution_result: Save execution metadata
- verify_execution: Verify outputs are correct
- list_stage4_results: List existing results

## Execution Workflow
1. Load execution context for the plan
2. **CRITICAL**: Call get_selected_method_code to get:
   - The winning method's implementation code
   - The EXACT data split strategy used in benchmarking
   - The benchmark metrics (your results should match these)
3. Load prepared data
4. **CRITICAL**: Split data according to the EXACT strategy from Stage 3.5B
5. Execute the method using the EXACT same code
6. Calculate metrics - they should match benchmark
7. Create results DataFrame with:
   - Date/index column
   - Actual values
   - Predicted values
   - Any relevant features
8. Save predictions and execution result
9. Verify the outputs

## DATA SPLIT STRATEGY (MUST FOLLOW EXACTLY)
The get_selected_method_code tool returns the data_split_strategy JSON.
You MUST use the EXACT same split to get matching metrics:

- strategy_type: "temporal" = row-based temporal split
- strategy_type: "temporal_column" = wide format (column-based, use all rows)
- train_size, validation_size, test_size: The exact percentages to use

## Results DataFrame Requirements
The saved predictions must include:
- date/index column (for time series plotting)
- 'actual' column (true values)
- 'predicted' column (model predictions)
- Keep original feature columns for context

## Execution Code Template
```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load prepared data
STAGE3B_OUT_DIR = Path('{STAGE3B_OUT_DIR}')
df = pd.read_parquet(STAGE3B_OUT_DIR / 'prepared_{{plan_id}}.parquet')

# Get column info from execution context
target_col = 'your_target_column'  # From get_selected_method_code
date_col = 'your_date_column'       # From get_selected_method_code

# CRITICAL: Use EXACT split strategy from Stage 3.5B
# These values should come from the data_split_strategy in get_selected_method_code:
train_size = 0.7   # From data_split_strategy.train_size
val_size = 0.15    # From data_split_strategy.validation_size
test_size = 0.15   # From data_split_strategy.test_size

# Apply split based on strategy_type
train_end = int(len(df) * train_size)
val_end = int(len(df) * (train_size + val_size))

train_df = df.iloc[:train_end].copy()
test_df = df.iloc[val_end:].copy()  # Skip validation, use test only

# Define the selected method (COPY EXACTLY from get_selected_method_code)
def predict_selected_method(train_df, test_df, target_col, date_col, **params):
    # ... implementation from Stage 3.5B winner ...
    pass

# Run prediction
predictions = predict_selected_method(train_df, test_df, target_col, date_col)

# Create results DataFrame
results_df = test_df.copy()
results_df['predicted'] = predictions['predicted'].values
results_df['actual'] = results_df[target_col]

# Calculate metrics
actual = results_df['actual'].values
predicted = results_df['predicted'].values

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
mask = actual != 0
mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else 0.0
r2 = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)) if len(actual) > 1 else 0.0

# VERIFY: Compare with benchmark metrics
# Expected MAE from Stage 3.5B: X.XXXX
# Your MAE should be close to this value!
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")
```

## Error Handling
- If execution fails, provide detailed error messages
- Do NOT use fallback methods - fail fast with clear diagnostics
- Log all errors to help debug issues

CRITICAL: If the winning method fails to execute, STOP and report the error.
Do NOT create fallback predictions. The user needs to see the real error.

IMPORTANT: The results_df must be saved as parquet for Stage 5 visualization.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage4_agent():
    """Create the Stage 4 agent graph."""

    llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(STAGE4_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage4State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE4_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage4", 100):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage4State) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage4State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE4_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage4(plan_id: str, pipeline_state: PipelineState = None, max_retries: int = 3) -> ExecutionResult:
    """
    Run Stage 4: Execution with automatic retry on failure.

    Executes the selected method and generates predictions.
    If execution fails, the agent will see the error and retry up to max_retries times.

    Args:
        plan_id: Plan ID to execute
        pipeline_state: Pipeline state (optional)
        max_retries: Maximum retry attempts (default: 3)

    Returns:
        ExecutionResult with predictions or detailed error information
    """
    logger.info(f"Starting Stage 4: Execution for {plan_id} (max_retries={max_retries})")

    for attempt in range(max_retries):
        attempt_num = attempt + 1
        logger.info(f"Stage 4 Attempt {attempt_num}/{max_retries}")

        try:
            result = _attempt_stage4_execution(plan_id, pipeline_state, attempt_num)

            # If we got a successful result, return it
            if result.status == ExecutionStatus.SUCCESS:
                logger.info(f"Stage 4 succeeded on attempt {attempt_num}")
                return result

            # If we got a failure, check if we should retry
            if attempt_num < max_retries:
                logger.warning(f"Stage 4 attempt {attempt_num} failed, retrying... Error: {result.summary}")
                continue
            else:
                logger.error(f"Stage 4 failed after {max_retries} attempts")
                return result

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Stage 4 attempt {attempt_num} exception: {e}\n{error_trace}")

            if attempt_num < max_retries:
                logger.warning(f"Retrying after exception...")
                continue
            else:
                return ExecutionResult(
                    plan_id=plan_id,
                    status=ExecutionStatus.FAILURE,
                    summary=f"Execution failed after {max_retries} attempts with exception: {e}",
                    errors=[str(e), error_trace]
                )

    # Fallback (should never reach here)
    return ExecutionResult(
        plan_id=plan_id,
        status=ExecutionStatus.FAILURE,
        summary="Execution failed unexpectedly",
        errors=["Unexpected fallthrough in retry logic"]
    )


def _attempt_stage4_execution(plan_id: str, pipeline_state: PipelineState = None, attempt_num: int = 1) -> ExecutionResult:
    """
    Single attempt at Stage 4 execution.

    The agent sees errors from previous attempts and can debug/fix them.
    """
    graph = create_stage4_agent()

    # Check if there are previous execution attempts with errors
    result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
    previous_errors = []
    if result_path.exists() and attempt_num > 1:
        try:
            prev_result = DataPassingManager.load_artifact(result_path)
            if prev_result.get('errors'):
                previous_errors = prev_result.get('errors', [])
        except:
            pass

    error_context = ""
    if previous_errors:
        error_context = f"""
## PREVIOUS ATTEMPT FAILED WITH ERRORS:
{chr(10).join(previous_errors[:3])}  # Show first 3 errors

You MUST analyze these errors and FIX the issue in this attempt.
Common fixes:
- If KeyError: Check column names are correct using load_prepared_data
- If NameError: Ensure all functions are defined before calling
- If shape mismatch: Verify data split creates correct DataFrame shapes
- If method execution failed: Check the winning method code syntax and required columns
"""

    initial_message = HumanMessage(content=f"""
Execute forecasting for plan: {plan_id} (Attempt {attempt_num})
{error_context}

Steps:
1. Load execution context (plan, data info, selected method)
2. Get the selected method's implementation code from Stage 3.5B
3. Load the prepared data and VERIFY column names
4. Split data according to EXACT strategy from Stage 3.5B (use get_selected_method_code)
5. Execute the selected method
6. Calculate final metrics (MAE, RMSE, MAPE, R²)
7. VERIFY metrics match Stage 3.5B benchmarks (within 10%)
8. Create results DataFrame with actual and predicted values
9. Save predictions parquet and execution result JSON using save_predictions tool
10. Verify outputs are correct

The results should include:
- Date/index column for time series plots (if applicable)
- 'actual' column with true values
- 'predicted' column with model predictions
- Original relevant columns for context

CRITICAL REQUIREMENTS:
- Use the EXACT same data split strategy as Stage 3.5B
- Your MAE should match the benchmark MAE from Stage 3.5B (±10%)
- If you get an error, ANALYZE it and FIX it - don't just try again blindly
- Use execute_python_code tool which provides detailed error diagnostics

IMPORTANT: You MUST use save_predictions tool to save the results.

Save outputs:
- Predictions: {STAGE4_OUT_DIR}/results_{plan_id}.parquet
- Metadata: {STAGE4_OUT_DIR}/execution_result_{plan_id}.json
""")

    config = {"configurable": {"thread_id": f"stage4_{plan_id}_attempt{attempt_num}"}}
    initial_state = Stage4State(messages=[initial_message], plan_id=plan_id)

    final_state = graph.invoke(initial_state, config)

    # Load execution result from disk
    predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"

    if result_path.exists():
        data = DataPassingManager.load_artifact(result_path)
        output = ExecutionResult(**data)
        logger.info(f"Stage 4 attempt {attempt_num} result: {output.status}")

        # Validate that predictions file exists
        if not predictions_path.exists():
            error_msg = f"Execution result exists but predictions file missing: {predictions_path}"
            logger.error(error_msg)
            # Return failure so retry logic can try again
            return ExecutionResult(
                plan_id=plan_id,
                status=ExecutionStatus.FAILURE,
                summary=error_msg,
                errors=[error_msg]
            )

        # Validate metrics are reasonable (not NaN, Inf, etc.)
        if output.metrics:
            mae = output.metrics.get('mae')
            if mae is None or mae != mae or mae == float('inf'):  # Check for NaN or Inf
                error_msg = f"Invalid MAE metric: {mae}"
                logger.error(error_msg)
                return ExecutionResult(
                    plan_id=plan_id,
                    status=ExecutionStatus.FAILURE,
                    summary=error_msg,
                    errors=[error_msg, "Metrics validation failed"]
                )

        return output
    else:
        # Agent failed to create execution result - raise error for retry
        error_details = []
        error_details.append(f"Agent failed to create execution result file on attempt {attempt_num}")
        error_details.append(f"Expected path: {result_path}")

        if predictions_path.exists():
            error_details.append("Note: Predictions file exists but result file is missing")
            error_details.append("This suggests the agent created predictions but failed to save metadata")
        else:
            error_details.append("Predictions file also missing - agent likely failed during execution")

        # Check what files the agent did create
        stage4_files = list(STAGE4_OUT_DIR.glob(f"*{plan_id}*"))
        if stage4_files:
            error_details.append(f"Files found in STAGE4_OUT_DIR: {[f.name for f in stage4_files]}")
        else:
            error_details.append("No output files found - agent may not have started execution")

        # Check if we can get more info from the agent's final state
        if hasattr(final_state, 'messages') and final_state.messages:
            last_message = final_state.messages[-1]
            if hasattr(last_message, 'content'):
                error_details.append(f"Last agent message: {last_message.content[:500]}")

        error_msg = "\n".join(error_details)
        logger.error(error_msg)

        # NO FALLBACK - raise error so retry mechanism can try again
        logger.error(f"Agent failed to save execution result. Attempt {attempt_num} must be retried.")
        raise RuntimeError(
            f"Stage 4 attempt {attempt_num} failed: Agent did not save execution result. "
            f"This may be due to max_tokens error or other agent failure. "
            f"Details: {error_msg}"
        )


# Removed _create_fallback_execution - no longer using fallback logic
# All execution must go through the LLM agent which has proper error handling


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage4_node(state: PipelineState) -> PipelineState:
    """
    Stage 4 node for the master pipeline graph.
    """
    state.mark_stage_started("stage4")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage4", "No plan ID available")
        return state

    try:
        output = run_stage4(plan_id, state)
        state.stage4_output = output
        state.mark_stage_completed("stage4", output)
    except Exception as e:
        state.mark_stage_failed("stage4", str(e))

    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage4(plan_id)
    print(f"Execution status: {output.status}")
    print(f"Metrics: {output.metrics}")
