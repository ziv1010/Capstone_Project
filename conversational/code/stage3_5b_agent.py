"""
Stage 3.5B Agent: Method Benchmarking

This agent benchmarks the proposed methods and selects the best performer.
Runs each method 3 times to ensure consistency and detect hallucinations.
"""

import json
from typing import Dict, Any, Optional, Annotated
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS,
    STAGE_MAX_ROUNDS, BENCHMARK_ITERATIONS, MAX_CV_THRESHOLD,
    DataPassingManager, logger, DEBUG, RECURSION_LIMIT
)
from code.models import TesterOutput, PipelineState
from tools.stage3_5b_tools import STAGE3_5B_TOOLS, reset_benchmark_state


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage35BState(BaseModel):
    """State for Stage 3.5B agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    methods_loaded: bool = False
    methods_tested: list = []
    best_method: str = ""
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE35B_SYSTEM_PROMPT = f"""You are a Method Benchmarking Agent responsible for testing and selecting the best forecasting method.

## CRITICAL: Be Concise and Action-Oriented
❌ DO NOT write lengthy explanations or repetitive thinking
❌ DO NOT repeat the same reasoning multiple times
✅ Think briefly, then ACT immediately with tool calls
✅ Keep responses under 500 tokens
✅ Be direct and efficient

## CRITICAL: Prevent Column Hallucination
❌ DO NOT assume column names exist (e.g., 'Year', 'date', 'time')
✅ ALWAYS call get_actual_columns() FIRST to see real columns
✅ Use ONLY columns that actually exist in the prepared data
✅ If date_col doesn't exist, use df.index or set date_col=None

## Your Role
1. **FIRST**: Call get_actual_columns() to verify columns
2. Load method proposals from Stage 3.5A AND the execution plan to get forecast configuration
3. Detect Data Format (Long vs Wide) based on proposal
4. Run each method {BENCHMARK_ITERATIONS} times for consistency
5. Calculate metrics using the plan's evaluation_metrics (NOT hardcoded MAE/RMSE/MAPE)
6. Validate results aren't hallucinated (check consistency)
7. Select the best method based on average performance
8. Save comprehensive benchmark results

## CRITICAL: Dynamic Metrics (NOT Hardcoded)
- Get evaluation_metrics from the execution plan (plan.evaluation_metrics)
- DO NOT assume only MAE/RMSE/MAPE - the plan specifies task-appropriate metrics
- For classification: accuracy, precision, recall, f1, etc.
- For forecasting: mae, rmse, mape, r2, and possibly smape, mase for multi-step
- Calculate ALL metrics specified in the plan

## Consistency Validation (CRITICAL)
You MUST run each method {BENCHMARK_ITERATIONS} times to check consistency:
- Calculate coefficient of variation (CV) of metrics across runs
- If CV < {MAX_CV_THRESHOLD}: Results are VALID
- If CV >= {MAX_CV_THRESHOLD}: Results may be HALLUCINATED

## Available Tools
- get_actual_columns: **CALL THIS FIRST** to prevent column hallucination
- load_method_proposals: Load methods from Stage 3.5A
- load_checkpoint: Resume from previous run if exists
- save_checkpoint: Save progress after each method
- record_thought_3_5b: Document reasoning
- run_benchmark_code: Execute method and get metrics
- calculate_metrics: Compute MAE, RMSE, MAPE
- validate_consistency: Check if results are consistent
- select_best_method: Select winner based on results
- save_tester_output: Save final results
- finish_benchmarking: Signal completion (Call this LAST)

## Execution Strategy Discovery (CRITICAL)

You MUST analyze the data split strategy from the method proposal before running any benchmarks.

### Step 1: Load and Understand the Split Strategy
The method proposal contains `data_split_strategy` with:
- `strategy_type`: Describes how to split (e.g., "temporal_row", "temporal_column", "random")
- `train_period`, `validation_period`, `test_period`: Descriptions of what data goes where
- Column/row specifications for implementing the split

### Step 2: Reason About Implementation
Based on the strategy_type, think about:
- **Row-based splits**: Use iloc or date-based filtering to separate train/test
- **Column-based splits**: Use different columns as features/targets for train vs test
- **Hybrid approaches**: May combine row and column operations

DO NOT assume a specific format. Read the strategy from the proposal and implement accordingly.

### Step 3: Implement the Split
Write code that:
1. Loads the prepared data
2. Implements the split strategy from the proposal
3. Runs the method's prediction function
4. Calculates metrics dynamically based on plan.evaluation_metrics

### Example Framework (Adapt Based on Actual Strategy):
```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load prepared data
STAGE3B_OUT_DIR = Path('{STAGE3B_OUT_DIR}')
df = pd.read_parquet(STAGE3B_OUT_DIR / 'prepared_{{plan_id}}.parquet')

# Get split strategy from method proposal
strategy = method_proposal['data_split_strategy']
strategy_type = strategy['strategy_type']

# Implement split based on strategy_type
if 'row' in strategy_type.lower():
    # Row-based split (e.g., temporal_row, random)
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
elif 'column' in strategy_type.lower():
    # Column-based split (discover from strategy details)
    # Use strategy['train_period'] and strategy['test_period'] to determine columns
    # This will vary by dataset - REASON about the specific columns
    pass
else:
    # Unknown strategy - analyze and decide
    pass

# Run the method (from method proposal)
predictions = method_function(train_df, test_df, target_col, date_col)

# Calculate metrics DYNAMICALLY from plan.evaluation_metrics
# CRITICAL: Use the metrics specified in the plan, not hardcoded defaults!
results = {{}}
for metric_name in plan['evaluation_metrics']:
    metric_lower = metric_name.lower()

    # Regression/Forecasting metrics
    if metric_lower == 'mae':
        results[metric_name] = float(np.mean(np.abs(actual - predicted)))
    elif metric_lower == 'rmse':
        results[metric_name] = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    elif metric_lower == 'mse':
        results[metric_name] = float(np.mean((actual - predicted) ** 2))
    elif metric_lower == 'mape':
        # Avoid division by zero
        mask = actual != 0
        results[metric_name] = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    elif metric_lower == 'r2':
        from sklearn.metrics import r2_score
        results[metric_name] = float(r2_score(actual, predicted))

    # Classification metrics (if task is classification)
    elif metric_lower == 'accuracy':
        from sklearn.metrics import accuracy_score
        results[metric_name] = float(accuracy_score(actual, predicted))
    elif metric_lower == 'precision':
        from sklearn.metrics import precision_score
        results[metric_name] = float(precision_score(actual, predicted, average='weighted'))
    elif metric_lower == 'recall':
        from sklearn.metrics import recall_score
        results[metric_name] = float(recall_score(actual, predicted, average='weighted'))
    elif metric_lower == 'f1' or metric_lower == 'f1_score':
        from sklearn.metrics import f1_score
        results[metric_name] = float(f1_score(actual, predicted, average='weighted'))

    else:
        # Unknown metric - log warning but don't fail
        print(f"Warning: Unknown metric '{{metric_name}}' - skipping")

print(json.dumps(results))
```

**KEY PRINCIPLE**: Don't prescribe code templates - reason about the strategy and data, then generate appropriate code.

## Workflow
1. Load method proposals
2. Check for existing checkpoint (resume if exists)
3. For each method (M1, M2, M3):
   a. THINK about how to run this method (Long vs Wide strategy)
   b. Run {BENCHMARK_ITERATIONS} iterations
   c. Collect metrics from each run
   d. Validate consistency (check CV)
   e. Save checkpoint after completing method
4. Compare all methods
5. Select best method (lowest average MAE for valid methods)
6. Save tester output
7. Call finish_benchmarking() to end the stage

## Error Handling
- If a method fails, record the error and move to next
- If all iterations fail, mark method as invalid
- At least one method must succeed for stage to complete

## Output Requirements
The tester output must include:
- methods_tested: List of results for each method
- selected_method_id: Winner (e.g., "M2")
- selected_method_name: Winner's name
- selection_rationale: Why this method was selected
- method_comparison_summary: Brief comparison of all methods
- winning_method_code: EXACT code used (automatically added from proposal)
- data_split_strategy: EXACT split strategy used (automatically added)
- benchmark_metrics: Metrics from winning method (for Stage 4 validation)

CRITICAL: The winning_method_code stored will be used verbatim by Stage 4!
Make sure your benchmarking uses the EXACT code from the method proposal.

IMPORTANT: Use checkpoints! If interrupted, you can resume from the last completed method.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage3_5b_agent():
    """Create the Stage 3.5B agent graph."""

    # Use stage-specific max_tokens if available, otherwise use default
    stage3_5b_config = SECONDARY_LLM_CONFIG.copy()
    stage3_5b_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage3_5b", SECONDARY_LLM_CONFIG["max_tokens"])

    llm = ChatOpenAI(**stage3_5b_config)
    llm_with_tools = llm.bind_tools(STAGE3_5B_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage35BState) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE35B_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage3_5b", 120):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing with current results.")],
                "complete": True
            }

        # Check if we just finished benchmarking
        if len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, ToolMessage) and last_msg.name == "finish_benchmarking":
                logger.info("Finish benchmarking signal received. Terminating Stage 3.5B.")
                return {
                    "messages": [AIMessage(content="Benchmarking complete. Ending stage.")],
                    "complete": True
                }

        response = llm_with_tools.invoke(messages)

        # Strip verbose <think> tags to prevent context bloat
        if response.content:
            import re
            # Remove <think>...</think> blocks to save context
            cleaned_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
            # If the cleaned content is empty but we have tool calls, provide minimal message
            if not cleaned_content.strip() and response.tool_calls:
                cleaned_content = "Executing tools..."
            response.content = cleaned_content.strip()

        if DEBUG:
            logger.debug(f"Stage 3.5B Agent Response: {response.content[:200]}...")  # Only log first 200 chars
            if response.tool_calls:
                logger.debug(f"Tool Calls: {response.tool_calls}")

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage35BState) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage35BState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE3_5B_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage3_5b(plan_id: str, pipeline_state: PipelineState = None) -> TesterOutput:
    """
    Run Stage 3.5B: Method Benchmarking.

    Tests all proposed methods and selects the best.
    """
    logger.info(f"Starting Stage 3.5B: Method Benchmarking for {plan_id}")

    reset_benchmark_state()
    graph = create_stage3_5b_agent()

    initial_message = HumanMessage(content=f"""
Benchmark methods for plan: {plan_id}

Steps:
1. Call get_actual_columns() FIRST to verify column names
2. Load method proposals from Stage 3.5A AND execution plan to get evaluation_metrics
3. Check for existing checkpoint (resume if available)
4. For each of the 3 methods (M1, M2, M3):
   a. Run {BENCHMARK_ITERATIONS} iterations
   b. Calculate ALL metrics from the plan's evaluation_metrics (NOT just MAE/RMSE/MAPE)
   c. Validate consistency (CV < {MAX_CV_THRESHOLD})
   d. Save checkpoint after completing the method
5. Compare all valid methods using the PRIMARY metric from the plan
6. Select the best method based on the plan's evaluation criteria
7. Save tester output using save_tester_output tool

The prepared data is at: {STAGE3B_OUT_DIR}/prepared_{plan_id}.parquet

IMPORTANT:
- Use the evaluation_metrics from the execution plan (DO NOT assume MAE/RMSE/MAPE)
- For classification: might be accuracy, f1, precision, recall
- For forecasting: might be MAE, RMSE, MAPE, R2
- Calculate ALL metrics specified in the plan

You MUST call save_tester_output with a valid JSON containing:
- plan_id: "{plan_id}"
- methods_tested: list of method results with ALL metrics
- selected_method_id: the best method ID (e.g., "M1")
- selection_rationale: why this method was selected (based on plan metrics)

Save output as: tester_{plan_id}.json

Remember: Run each method {BENCHMARK_ITERATIONS} times and check consistency!
""")

    config = {
        "configurable": {"thread_id": f"stage3_5b_{plan_id}"},
        "recursion_limit": RECURSION_LIMIT
    }
    initial_state = Stage35BState(messages=[initial_message], plan_id=plan_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load tester output from disk
        tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
        if tester_path.exists():
            data = DataPassingManager.load_artifact(tester_path)
            output = TesterOutput(**data)
            logger.info(f"Stage 3.5B complete: Selected {output.selected_method_id}")
            return output
        else:
            # NO FALLBACK - raise an error so the pipeline can retry
            logger.error("Agent failed to save tester output. Stage must be retried.")
            raise RuntimeError(
                "Stage 3.5B failed: Agent did not save tester output. "
                "This may be due to max_tokens error or other agent failure. "
                "The stage should be retried."
            )

    except Exception as e:
        # Check if it's a max_tokens error
        error_msg = str(e).lower()
        if 'max_tokens' in error_msg or 'token' in error_msg:
            logger.error(f"Stage 3.5B failed with token error: {e}")
            raise RuntimeError(
                f"Stage 3.5B failed due to max_tokens error: {e}. "
                "This stage needs to be retried."
            )
        else:
            logger.error(f"Stage 3.5B failed: {e}")
            raise


# Removed _create_default_tester_output function
# No more fallback logic - stages must complete successfully or fail properly


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage3_5b_node(state: PipelineState) -> PipelineState:
    """
    Stage 3.5B node for the master pipeline graph.
    Includes automatic retry logic for transient failures.
    """
    from code.config import MAX_RETRIES, RETRY_STAGES
    
    state.mark_stage_started("stage3_5b")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage3_5b", "No plan ID available")
        return state

    # Retry logic for resilient execution
    max_retries = MAX_RETRIES if "stage3_5b" in RETRY_STAGES else 1
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Stage 3.5B attempt {attempt}/{max_retries}")
            output = run_stage3_5b(plan_id, state)
            state.stage3_5b_output = output
            state.mark_stage_completed("stage3_5b", output)
            logger.info(f"✅ Stage 3.5B succeeded on attempt {attempt}")
            return state
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Check if it's a retryable error
            is_retryable = (
                'max_tokens' in error_msg or 
                'token' in error_msg or
                'did not save' in error_msg
            )
            
            if is_retryable and attempt < max_retries:
                logger.warning(
                    f"⚠️  Stage 3.5B attempt {attempt} failed with retryable error: {e}. "
                    f"Retrying... ({attempt}/{max_retries})"
                )
                # Clean up any partial outputs before retry
                # NOTE: We preserve checkpoints to allow resuming from the last completed method
                tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
                if tester_path.exists():
                    logger.info(f"Removing partial tester output: {tester_path}")
                    tester_path.unlink()
                # Checkpoint is NOT deleted - agent will resume from last saved state
                continue
            else:
                # Non-retryable error or max retries reached
                if attempt >= max_retries:
                    logger.error(
                        f"❌ Stage 3.5B failed after {max_retries} attempts. "
                        f"Last error: {e}"
                    )
                state.mark_stage_failed("stage3_5b", str(last_error))
                return state
    
    # Should not reach here, but handle it
    state.mark_stage_failed("stage3_5b", str(last_error))
    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage3_5b(plan_id)
    print(f"Selected method: {output.selected_method_id} - {output.selected_method_name}")
    print(f"Rationale: {output.selection_rationale}")
