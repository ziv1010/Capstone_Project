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
    STAGE4_OUT_DIR, STAGE4_WORKSPACE, SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS,
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
3. Generate predictions for the test set (for validation)
4. **FOR FORECASTING**: Generate future forecasts if forecast_horizon > 0
5. Calculate evaluation metrics
6. Save comprehensive results

## CRITICAL: TWO TYPES OF PREDICTIONS

### 1. TEST SET PREDICTIONS (Required for all tasks)
- Generate predictions on the test set
- Compare actual vs predicted values
- Calculate metrics (MAE, RMSE, MAPE, R²)
- Metrics MUST match Stage 3.5B benchmark

### 2. FUTURE FORECASTS (Required if forecast_horizon > 0)
- Check the plan for `forecast_horizon` and `forecast_type`
- If forecast_horizon > 0: Generate predictions for the NEXT N periods
- forecast_type can be:
  - "single_step": Predict next period only
  - "multi_step": Predict next N periods directly
  - "recursive": Use each prediction as input for the next

**IMPORTANT**: Future forecasts have NO actual values (they're in the future!)
Save them separately with a 'forecast_type' column to distinguish from test predictions.

## Your Goals
- Execute the winning method from benchmarking
- Generate test set predictions with actual vs predicted values
- Generate future forecasts if specified in the plan
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

## CRITICAL: File Path Variables (DO NOT HARDCODE PATHS)
When using execute_python_code, these variables are available in the namespace:
- STAGE3B_OUT_DIR: Where prepared data files are located
- STAGE4_OUT_DIR: Where to save execution results
- DATA_DIR: Raw data directory

**NEVER hardcode paths like '/scratch/.../stage3b_out/' or '/scratch/.../data/'**
**ALWAYS use the provided variables: STAGE3B_OUT_DIR, STAGE4_OUT_DIR, DATA_DIR**

Example (CORRECT):
```python
df = pd.read_parquet(STAGE3B_OUT_DIR / 'prepared_PLAN-TSK-001.parquet')
```

Example (WRONG - DO NOT DO THIS):
```python
df = pd.read_parquet('/scratch/ziv_baretto/llmserve/final_code/conversational/output/stage3b_out/prepared_PLAN-TSK-001.parquet')
```

## Execution Workflow
1. Load execution context for the plan
2. **CRITICAL**: Call get_selected_method_code to get:
   - The winning method's implementation code
   - The EXACT data split strategy used in benchmarking
   - The benchmark metrics (your results should match these)
3. Load prepared data and VERIFY the column names match what's expected
4. **CRITICAL**: Check if forecast_horizon > 0 in the plan
5. **STEP A: TEST SET PREDICTIONS** (validation)
   - **MUST USE**: Copy the EXACT winning_method_code from get_selected_method_code
   - **MUST USE**: The EXACT data split strategy from Stage 3.5B
   - **DO NOT MODIFY** the method code - use it verbatim!
   - Execute the method using the code exactly as provided
   - Calculate metrics - they MUST match benchmark (±5%)
   - Create test results DataFrame with actual vs predicted
6. **STEP B: FUTURE FORECASTS** (if forecast_horizon > 0)
   - Generate forecasts for the next N periods
   - Use recursive/multi-step approach as specified
   - Create forecast DataFrame with predicted values (no actuals)
   - Add 'forecast_type' column to distinguish from test predictions
7. **COMBINE RESULTS**:
   - Concatenate test predictions + future forecasts
   - Mark each row with 'prediction_type': 'test' or 'forecast'
8. Save predictions and execution result
9. Verify the outputs

**CRITICAL VALIDATION**:
- Your test MAE MUST be within ±5% of the Stage 3.5B benchmark MAE
- If metrics differ significantly, you are NOT using the same code or split strategy
- DO NOT proceed if validation fails - debug the issue first

## DATA SPLIT STRATEGY (MUST FOLLOW EXACTLY)
The get_selected_method_code tool returns the data_split_strategy JSON.
You MUST use the EXACT same split to get matching metrics:

- strategy_type: "temporal" = row-based temporal split
- strategy_type: "temporal_column" = wide format (column-based, use all rows)
- train_size, validation_size, test_size: The exact percentages to use

## Results DataFrame Requirements
The saved predictions must include:
- date/index column (for time series plotting)
- 'actual' column (true values for test set, NaN for forecasts)
- 'predicted' column (model predictions)
- 'prediction_type' column: 'test' or 'forecast'
- Keep original feature columns for context

## Execution Code Template
```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Load prepared data using PROVIDED VARIABLES (DO NOT HARDCODE PATHS)
df = pd.read_parquet(STAGE3B_OUT_DIR / 'prepared_{{plan_id}}.parquet')

# Get context from plan
target_col = 'your_target_column'   # From get_selected_method_code
date_col = 'your_date_column'       # From get_selected_method_code
forecast_horizon = 0                # From execution context (e.g., 5)
forecast_granularity = 'year'       # From execution context (e.g., 'year', 'month')

# === STEP A: TEST SET PREDICTIONS (for validation) ===
# CRITICAL: Use data_split_strategy from get_selected_method_code
# Example shows temporal_column strategy - ADAPT based on actual strategy!
split_strategy = {{"strategy_type": "temporal_column"}}  # From get_selected_method_code

# CRITICAL: Copy the EXACT winning_method_code from get_selected_method_code
# DO NOT modify it - paste it verbatim!
# Example (replace with actual code from get_selected_method_code):
def predict_selected_method(train_df, test_df, target_col, date_col, **params):
    # <<< PASTE EXACT CODE FROM get_selected_method_code HERE >>>
    # DO NOT modify the code at all!
    pass

# For temporal_column strategy: features are year columns, target is specific year
# This example assumes wide format data - adjust based on actual split_strategy
if split_strategy.get('strategy_type') == 'temporal_column':
    # Wide format: different years are different columns
    # Implementation depends on the exact winning_method_code
    # Just execute the method as-is from Stage 3.5B
    test_predictions = predict_selected_method(df, df, target_col, date_col)
else:
    # Row-based temporal split (long format)
    train_size = 0.7
    val_size = 0.15
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[val_end:].copy()
    test_predictions = predict_selected_method(train_df, test_df, target_col, date_col)

# Create test results
test_results = test_df.copy()
test_results['predicted'] = test_predictions['predicted'].values
test_results['actual'] = test_results[target_col]
test_results['prediction_type'] = 'test'

# Calculate metrics DYNAMICALLY based on plan.evaluation_metrics (should match Stage 3.5B)
actual = test_results['actual'].values
predicted = test_results['predicted'].values

# Get metrics from execution context (plan.evaluation_metrics)
# CRITICAL: Calculate ALL metrics specified in the plan, not just MAE/RMSE
metrics = {{}}
for metric_name in plan['evaluation_metrics']:
    metric_lower = metric_name.lower()

    if metric_lower == 'mae':
        metrics[metric_name] = float(np.mean(np.abs(actual - predicted)))
    elif metric_lower == 'rmse':
        metrics[metric_name] = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    elif metric_lower == 'mse':
        metrics[metric_name] = float(np.mean((actual - predicted) ** 2))
    elif metric_lower == 'mape':
        mask = actual != 0
        metrics[metric_name] = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    elif metric_lower == 'r2':
        from sklearn.metrics import r2_score
        metrics[metric_name] = float(r2_score(actual, predicted))
    elif metric_lower == 'accuracy':
        from sklearn.metrics import accuracy_score
        metrics[metric_name] = float(accuracy_score(actual, predicted))
    elif metric_lower in ['precision', 'recall', 'f1', 'f1_score']:
        # Classification metrics (if applicable)
        from sklearn.metrics import precision_score, recall_score, f1_score
        if metric_lower == 'precision':
            metrics[metric_name] = float(precision_score(actual, predicted, average='weighted'))
        elif metric_lower == 'recall':
            metrics[metric_name] = float(recall_score(actual, predicted, average='weighted'))
        else:
            metrics[metric_name] = float(f1_score(actual, predicted, average='weighted'))

# Print metrics
metric_str = ", ".join([f"{{k}}: {{v:.4f}}" for k, v in metrics.items()])
print(f"Test Set Metrics - {{metric_str}}")

# === STEP B: FUTURE FORECASTS (if forecast_horizon > 0) ===
if forecast_horizon > 0:
    print(f"\\nGenerating {forecast_horizon} future forecasts...")

    # Train on ALL available data for forecasting
    full_train_df = df.copy()

    # Create future date index
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        last_date = df[date_col].max()
        if forecast_granularity == 'year':
            future_dates = pd.date_range(last_date, periods=forecast_horizon+1, freq='Y')[1:]
        elif forecast_granularity == 'month':
            future_dates = pd.date_range(last_date, periods=forecast_horizon+1, freq='M')[1:]
        else:
            future_dates = pd.date_range(last_date, periods=forecast_horizon+1, freq='D')[1:]
    else:
        # No date column - use sequential index
        future_dates = range(len(df), len(df) + forecast_horizon)

    # Generate recursive forecasts
    forecast_values = []
    for i in range(forecast_horizon):
        # Predict next period using current data
        # For recursive: use previous predictions as features
        next_pred = predict_selected_method(full_train_df, full_train_df.iloc[[-1]], target_col, date_col)
        forecast_values.append(next_pred['predicted'].values[0])

        # Update training data with prediction for next iteration
        # (Implementation depends on method - may need to append predicted value)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        date_col: future_dates if date_col else future_dates,
        'predicted': forecast_values,
        'actual': [np.nan] * forecast_horizon,  # No actuals for future
        'prediction_type': ['forecast'] * forecast_horizon
    })

    # Combine test + forecast
    results_df = pd.concat([test_results, forecast_df], ignore_index=True)
else:
    # No future forecasts - just test results
    results_df = test_results

print(f"\\nTotal results: {len(results_df)} rows ({len(test_results)} test + {forecast_horizon} forecasts)")
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

    # Use stage-specific max_tokens if available, otherwise use default
    stage4_config = SECONDARY_LLM_CONFIG.copy()
    stage4_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage4", SECONDARY_LLM_CONFIG["max_tokens"])

    llm = ChatOpenAI(**stage4_config)
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

CRITICAL: Check if this is a FORECASTING task with forecast_horizon > 0!
If so, you must generate BOTH test predictions AND future forecasts.

Steps:
1. Load execution context (plan, data info, selected method)
2. **CHECK forecast_horizon in the plan** - if > 0, this is forecasting!
3. **CRITICAL**: Get the selected method's EXACT implementation code from Stage 3.5B using get_selected_method_code
4. Load the prepared data and VERIFY column names match what the method expects
5. **PART A: TEST SET PREDICTIONS** (validation)
   - **COPY VERBATIM** the winning_method_code from get_selected_method_code
   - **DO NOT MODIFY** the code - use it exactly as provided!
   - Use the EXACT data_split_strategy from Stage 3.5B
   - Execute the method exactly as it was run in benchmarking
   - Calculate metrics (MAE, RMSE, MAPE, R²)
   - **VALIDATE**: Your MAE MUST be within ±5% of Stage 3.5B benchmark MAE!
   - If validation fails, you are NOT using the same code/strategy - debug it!
6. **PART B: FUTURE FORECASTS** (if forecast_horizon > 0)
   - Generate forecasts for the next N periods (N = forecast_horizon)
   - Use recursive/iterative approach
   - Create DataFrame with predicted values (actual = NaN for future)
   - Mark with 'prediction_type' = 'forecast'
7. Combine test predictions + future forecasts into single DataFrame
8. Save predictions parquet and execution result JSON using save_predictions tool
9. Verify outputs are correct

The results DataFrame must include:
- Date/time column (for time series plots)
- 'actual' column (values for test set, NaN for forecasts)
- 'predicted' column (model predictions)
- 'prediction_type' column ('test' or 'forecast')
- Original relevant columns for context

CRITICAL REQUIREMENTS:
- **MUST USE**: The EXACT winning_method_code from Stage 3.5B (DO NOT write your own version!)
- **MUST USE**: The EXACT data_split_strategy from Stage 3.5B
- **MUST VALIDATE**: Test MAE within ±5% of benchmark (e.g., if benchmark=0.246, yours should be ~0.23-0.26)
- If forecast_horizon > 0: Generate future forecasts (NOT just test set)!
- If you get an error, ANALYZE it and FIX it - don't just try again blindly
- If metrics don't match, you did something wrong - debug before proceeding!

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
