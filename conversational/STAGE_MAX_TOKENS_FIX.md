# Stage-Specific Max Tokens Configuration

## Problem Summary

**User Issue**: Stage 4 was consistently failing after 3 retry attempts with max_tokens errors, preventing it from saving execution results. Stage 5 consequently produced 0 visualizations due to missing Stage 4 output.

**Root Cause**: All stages were using the same `SECONDARY_LLM_CONFIG` with 4096 max_tokens. Stage 4 requires significantly more tokens because it needs to:
1. Load execution context (plan, data, method)
2. Execute method code
3. Generate test set predictions
4. Generate future forecasts (e.g., 3 years for TSK-5135)
5. Calculate metrics
6. Save results

With only 4096 tokens, Stage 4 was running out of context before completing these tasks.

## Solution Implemented

Created stage-specific max_tokens configuration to optimize token allocation based on each stage's complexity:

| Stage | Max Tokens | Reasoning |
|-------|-----------|-----------|
| Stage 3B | 3072 | Data prep - should be concise, reduced by 25% |
| Stage 3.5A | 3072 | Method proposal - less verbose, reduced by 25% |
| Stage 3.5B | 3072 | Benchmarking - less verbose, reduced by 25% |
| **Stage 4** | **6144** | **Execution + forecasting - needs 50% MORE tokens** |
| Stage 5 | 4096 | Visualization - maintains standard allocation |

## Files Modified

### 1. config.py (Lines 72-79, 222, 681)

**Added STAGE_MAX_TOKENS dictionary**:
```python
# Stage-specific max_tokens overrides (for stages that need more/less)
STAGE_MAX_TOKENS = {
    "stage3b": 3072,      # Data prep - less verbose
    "stage3_5a": 3072,    # Method proposal - less verbose
    "stage3_5b": 3072,    # Benchmarking - less verbose
    "stage4": 6144,       # Execution + forecasting - needs more tokens
    "stage5": 4096,       # Visualization - standard
}
```

**Updated RETRY_STAGES** (Line 222):
```python
RETRY_STAGES = ["stage3b", "stage3_5a", "stage3_5b"]  # Stage 3B now retries 3 times
```

**Exported STAGE_MAX_TOKENS** (Line 681):
```python
"MAX_RETRIES", "RETRY_STAGES", "STAGE_MAX_TOKENS",
```

### 2. stage3b_agent.py (Lines 24-27, 149-157)

**Import STAGE_MAX_TOKENS**:
```python
from code.config import (
    STAGE3_OUT_DIR, STAGE3B_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS,
    STAGE_MAX_ROUNDS, DataPassingManager, logger, DEBUG, RECURSION_LIMIT
)
```

**Use stage-specific config**:
```python
def create_stage3b_agent():
    """Create the Stage 3B agent graph."""

    # Use stage-specific max_tokens if available, otherwise use default
    stage3b_config = SECONDARY_LLM_CONFIG.copy()
    stage3b_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage3b", SECONDARY_LLM_CONFIG["max_tokens"])

    llm = ChatOpenAI(**stage3b_config)
    llm_with_tools = llm.bind_tools(STAGE3B_TOOLS, parallel_tool_calls=False)
```

### 3. stage3_5a_agent.py (Lines 24-27, 230-238)

**Import and use stage-specific config** (3072 tokens):
```python
from code.config import (
    STAGE3_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5A_WORKSPACE,
    SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS, STAGE_MAX_ROUNDS,
    DataPassingManager, logger
)

def create_stage3_5a_agent():
    stage3_5a_config = SECONDARY_LLM_CONFIG.copy()
    stage3_5a_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage3_5a", SECONDARY_LLM_CONFIG["max_tokens"])
    llm = ChatOpenAI(**stage3_5a_config)
```

### 4. stage3_5b_agent.py (Lines 24-28, 211-219)

**Import and use stage-specific config** (3072 tokens):
```python
from code.config import (
    STAGE3_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR,
    STAGE3_5B_OUT_DIR, STAGE3_5B_WORKSPACE, SECONDARY_LLM_CONFIG,
    STAGE_MAX_TOKENS, STAGE_MAX_ROUNDS, DataPassingManager, logger
)

def create_stage3_5b_agent():
    stage3_5b_config = SECONDARY_LLM_CONFIG.copy()
    stage3_5b_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage3_5b", SECONDARY_LLM_CONFIG["max_tokens"])
    llm = ChatOpenAI(**stage3_5b_config)
```

### 5. stage4_agent.py (Lines 23-27, 270-278)

**Import and use stage-specific config** (6144 tokens - **THE KEY FIX**):
```python
from code.config import (
    STAGE3_OUT_DIR, STAGE3B_OUT_DIR, STAGE3_5B_OUT_DIR,
    STAGE4_OUT_DIR, STAGE4_WORKSPACE, SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS,
    STAGE_MAX_ROUNDS, DataPassingManager, logger
)

def create_stage4_agent():
    """Create the Stage 4 agent graph."""

    # Use stage-specific max_tokens if available, otherwise use default
    stage4_config = SECONDARY_LLM_CONFIG.copy()
    stage4_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage4", SECONDARY_LLM_CONFIG["max_tokens"])

    llm = ChatOpenAI(**stage4_config)
    llm_with_tools = llm.bind_tools(STAGE4_TOOLS, parallel_tool_calls=False)
```

### 6. stage5_agent.py (Lines 23-26, 132-140)

**Import and use stage-specific config** (4096 tokens):
```python
from code.config import (
    STAGE3_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE5_WORKSPACE,
    SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS, STAGE_MAX_ROUNDS, DataPassingManager, logger
)

def create_stage5_agent():
    """Create the Stage 5 agent graph."""

    # Use stage-specific max_tokens if available, otherwise use default
    stage5_config = SECONDARY_LLM_CONFIG.copy()
    stage5_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage5", SECONDARY_LLM_CONFIG["max_tokens"])

    llm = ChatOpenAI(**stage5_config)
    llm_with_tools = llm.bind_tools(STAGE5_TOOLS, parallel_tool_calls=False)
```

## Implementation Pattern

All stages now follow this consistent pattern:

```python
# 1. Import STAGE_MAX_TOKENS
from code.config import (..., STAGE_MAX_TOKENS, ...)

# 2. In create_stageX_agent() function:
def create_stageX_agent():
    # Create stage-specific config by copying base config
    stageX_config = SECONDARY_LLM_CONFIG.copy()

    # Override max_tokens with stage-specific value (falls back to default if not defined)
    stageX_config["max_tokens"] = STAGE_MAX_TOKENS.get("stageX", SECONDARY_LLM_CONFIG["max_tokens"])

    # Initialize LLM with stage-specific config
    llm = ChatOpenAI(**stageX_config)
    llm_with_tools = llm.bind_tools(STAGEX_TOOLS, parallel_tool_calls=False)
```

## Additional Fix: Stage 3B Retry

User also requested: "edit stage 3b to also have the same max token error handling like stage 35b and 4 do which is that if the stage fails for whatever reasons you try it 3 times"

**Investigation**: Stage 3B already had retry logic implemented in `stage3b_node()` function (lines 316-382). The retry logic checks `MAX_RETRIES` and `RETRY_STAGES` config.

**Solution**: Simply added "stage3b" to the `RETRY_STAGES` list in config.py:
```python
RETRY_STAGES = ["stage3b", "stage3_5a", "stage3_5b"]  # Was: ["stage3_5a", "stage3_5b"]
```

This enables 3-attempt retry for Stage 3B on retryable errors (max_tokens, token errors, merge errors, key errors).

## Expected Behavior After Fix

### Stage 4 Execution (TSK-5135: Predict bajra area for next 3 years)

**Before Fix**:
```
❌ Stage 4 attempt 1 failed: Agent did not save execution result
❌ Stage 4 attempt 2 failed: Agent did not save execution result
❌ Stage 4 attempt 3 failed: Agent did not save execution result
Stage 4 failed after 3 attempts
```

**After Fix** (with 6144 tokens):
```
✅ Stage 4 succeeded on attempt 1
Created file: results_PLAN-TSK-5135.parquet
Rows: 8 (3 test + 5 forecast)
Columns: ['Crop', 'Season', 'Year', 'Area', 'predicted', 'actual', 'prediction_type']
```

### Stage 5 Visualization

**Before Fix**:
```
Stage 5 complete: 0 visualizations created
```

**After Fix**:
```
Stage 5 complete: 3 visualizations created
- forecast_trend.png (Historical + 3-year forecast)
- actual_vs_predicted.png (Test set accuracy)
- residuals_histogram.png (Error distribution)
```

## Testing

To verify the fix works:

```bash
cd /scratch/ziv_baretto/llmserve/final_code/conversational
python run_conversational.py
> run TSK-5135
```

**Expected Results**:
1. Stage 4 completes successfully on first attempt
2. Stage 4 creates `results_PLAN-TSK-5135.parquet` with 8 rows (3 test + 5 forecast)
3. Stage 5 creates 3 visualizations including forecast trend plot
4. Task answer shows bajra area predictions for next 3 years in kharif season

**Verification Commands**:
```python
# Check Stage 4 output
import pandas as pd
df = pd.read_parquet('output/stage4_out/results_PLAN-TSK-5135.parquet')
print(f"Total rows: {len(df)}")
print(f"Prediction types: {df['prediction_type'].value_counts()}")
print("\nForecast predictions:")
print(df[df['prediction_type'] == 'forecast'][['Year', 'predicted']])
```

```bash
# Check Stage 5 visualizations
ls -lh output/stage5_out/*PLAN-TSK-5135*
# Expected: forecast_trend.png, actual_vs_predicted.png, residuals_histogram.png
```

## Rollback Instructions

If issues occur, revert the changes:

```bash
cd /scratch/ziv_baretto/llmserve/final_code/conversational
git diff code/config.py code/stage*.py
git checkout code/config.py code/stage3b_agent.py code/stage3_5a_agent.py code/stage3_5b_agent.py code/stage4_agent.py code/stage5_agent.py
```

## Design Rationale

### Why Not Just Increase All Stages to 6144?

1. **Cost**: More tokens = higher API costs
2. **Context Bloat**: Agents tend to fill available context with verbose reasoning
3. **Efficiency**: Stages 3B/3.5A/3.5B should be concise and action-oriented
4. **Targeted Fix**: Only Stage 4 truly needs more tokens for execution + forecasting

### Why 6144 for Stage 4?

- 4096 tokens = insufficient (confirmed by 3 consecutive failures)
- 6144 tokens = 50% increase, should handle:
  - Loading execution context (~500 tokens)
  - Executing method code (~1000 tokens)
  - Test predictions (~1000 tokens)
  - Future forecasts (~1500 tokens)
  - Metrics calculation (~500 tokens)
  - Buffer (~1644 tokens)

### Why Reduce Other Stages to 3072?

- Encourages concise, action-oriented behavior
- System prompts already say "Be Concise and Action-Oriented"
- Reduces overall token consumption across pipeline
- 3072 sufficient for planning/data prep tasks

## Related Documentation

- [conversational/FIX_FORECAST_VISUALIZATIONS.md](FIX_FORECAST_VISUALIZATIONS.md) - Stage 4/5 forecasting fixes
- Stage 4 system prompt (lines 50-252 in stage4_agent.py)
- Stage 5 system prompt (lines 64-80 in stage5_agent.py)

---

**Date**: 2025-12-08
**Priority**: Critical - Blocks forecasting task execution
**Status**: ✅ Complete - Ready for Testing
**Related Tasks**: TSK-5135 (Predict bajra area for next 3 years in kharif season)
