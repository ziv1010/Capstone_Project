# Fix for Forecasting Visualization Issues

## Problem Summary

User Task: "Predict rice cultivation area for the **next 5 years**"
Current Output: Only showing test set actual vs predicted (3 rows), NO future forecasts

**Root Causes**:
1. **Stage 4** doesn't generate future forecasts (only test set predictions)
2. **Stage 5** has hardcoded visualizations (actual vs predicted scatter, residuals)
3. Neither stage checks `forecast_horizon` from the execution plan

## What's Been Fixed

### ✅ Stage 4 System Prompt Updates

**File**: `conversational/code/stage4_agent.py`

**Changes Made**:
1. Added explicit instructions to check `forecast_horizon` in execution plan
2. Split execution into two parts:
   - PART A: Test set predictions (for validation/metrics)
   - PART B: Future forecasts (if forecast_horizon > 0)
3. Updated execution template to show recursive forecasting logic
4. Added `prediction_type` column to distinguish 'test' vs 'forecast' rows
5. Updated initial message to emphasize forecast generation

**Lines Modified**: 50-252, 417-461

### ✅ Stage 5 System Prompt Updates

**File**: `conversational/code/stage5_agent.py`

**Changes Made**:
1. Added task-aware visualization instructions
2. Differentiated between forecasting tasks vs other tasks
3. Emphasized forecast trend plot as most important for forecasting

**Lines Modified**: 64-80

### ⚠️ Stage 5 Tools - NEEDS COMPLETION

**File**: `conversational/tools/stage5_tools.py`

**What Needs to Be Done**:

Add forecast trend plot generation to `create_standard_plots` function (starting at line 453):

```python
@tool
def create_standard_plots(plan_id: str) -> str:
    # ... existing setup code ...

    df = pd.read_parquet(predictions_path)

    # Check if this is a forecasting task
    has_prediction_type = 'prediction_type' in df.columns
    is_forecasting = has_prediction_type and 'forecast' in df['prediction_type'].unique()

    created_plots = []

    # === NEW: FORECAST TREND PLOT (for forecasting tasks) ===
    if is_forecasting and date_cols:
        date_col = date_cols[0]
        pred_col, actual_col = pred_cols[0], actual_cols[0]

        # Separate test and forecast data
        test_df = df[df['prediction_type'] == 'test'].copy()
        forecast_df = df[df['prediction_type'] == 'forecast'].copy()

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot historical actuals
        ax.plot(test_df[date_col], test_df[actual_col],
                'o-', label='Historical Actual', linewidth=2, markersize=8)

        # Plot test predictions
        ax.plot(test_df[date_col], test_df[pred_col],
                's--', label='Test Predictions', linewidth=2, markersize=6, alpha=0.7)

        # Plot future forecasts
        ax.plot(forecast_df[date_col], forecast_df[pred_col],
                'D:', label='Future Forecasts', linewidth=2, markersize=8, color='red')

        # Add vertical line separating history from forecast
        if len(test_df) > 0:
            last_historical_date = test_df[date_col].max()
            ax.axvline(x=last_historical_date, color='gray', linestyle='--',
                      alpha=0.5, label='Forecast Start')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(target_col if 'target_col' in locals() else 'Value', fontsize=12)
        ax.set_title('Forecast Trend: Historical Data + Future Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(STAGE5_OUT_DIR / f'{plan_id}_forecast_trend.png', dpi=150)
        plt.close()
        created_plots.append('forecast_trend.png')

    # === EXISTING: Test Set Accuracy (only test predictions) ===
    if pred_cols and actual_cols:
        test_only = df[df['prediction_type'] == 'test'] if 'prediction_type' in df.columns else df
        if len(test_only) > 0:
            pred_col, actual_col = pred_cols[0], actual_cols[0]

            # Filter out NaN actuals
            valid_test = test_only.dropna(subset=[actual_col, pred_col])

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(valid_test[actual_col], valid_test[pred_col], alpha=0.6, s=100)
            ax.plot([valid_test[actual_col].min(), valid_test[actual_col].max()],
                   [valid_test[actual_col].min(), valid_test[actual_col].max()],
                   'r--', label='Perfect Prediction', linewidth=2)
            ax.set_xlabel('Actual', fontsize=12)
            ax.set_ylabel('Predicted', fontsize=12)
            ax.set_title('Test Set Accuracy: Actual vs Predicted', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(STAGE5_OUT_DIR / f'{plan_id}_actual_vs_predicted.png', dpi=150)
            plt.close()
            created_plots.append('actual_vs_predicted.png')

    # === EXISTING: Residuals (test set only) ===
    # ... keep existing residuals code ...

    # === REMOVE or UPDATE: Old time series plot ===
    # The old time series plot (lines 515-535) should be removed or updated
    # because the new forecast trend plot is better
```

## Testing the Fix

### Expected Behavior After Full Implementation

1. **Stage 4 Output** (`results_PLAN-TSK-9391.parquet`):
   ```
   Columns: ['Crop', 'Season', 'Year', 'Area', 'predicted', 'actual', 'prediction_type']

   Rows:
   - 3 rows with prediction_type='test' (2024-25 data)
   - 5 rows with prediction_type='forecast' (2025-2030 future)
   Total: 8 rows
   ```

2. **Stage 5 Visualizations**:
   - `forecast_trend.png`: Shows historical + test + 5 future years
   - `actual_vs_predicted.png`: Test set accuracy only
   - `residuals_histogram.png`: Error distribution (test set)

### Manual Testing Steps

```bash
cd /scratch/ziv_baretto/llmserve/final_code/conversational
python run_conversational.py
> run TSK-9391
```

**Check Stage 4 Output**:
```python
import pandas as pd
df = pd.read_parquet('output/stage4_out/results_PLAN-TSK-9391.parquet')
print(df.columns)
print(df['prediction_type'].value_counts())
print(df[df['prediction_type'] == 'forecast'][['Year', 'predicted']])
```

**Check Stage 5 Visualizations**:
```bash
ls -lh output/stage5_out/*PLAN-TSK-9391*
# Should see: forecast_trend.png, actual_vs_predicted.png, residuals_histogram.png
```

## Implementation Status

### ✅ Completed
- Stage 4 system prompt updates
- Stage 4 execution workflow updates
- Stage 5 system prompt updates

### ⚠️ Partially Complete
- Stage 5 tool updates (code template provided above)

### 📝 Recommended Next Steps

1. **Complete Stage 5 Tool Updates**:
   - Apply the forecast trend plot code to `stage5_tools.py`
   - Test with a forecasting task

2. **Handle Edge Cases**:
   - What if date_col doesn't exist or isn't datetime?
   - What if forecast_horizon = 0?
   - What if the model can't generate recursive forecasts?

3. **Improve Stage 4 Forecasting Logic**:
   - Current template shows basic recursive approach
   - May need model-specific implementations (ARIMA, Prophet, etc.)
   - Consider adding confidence intervals for forecasts

4. **Validate Metrics**:
   - Test metrics should still match Stage 3.5B benchmarks
   - Forecast metrics can't be calculated (no actuals)
   - Document this clearly in outputs

## Key Principles Applied

1. **Task-Aware Execution**: Check `forecast_horizon` to determine behavior
2. **Separation of Concerns**: Test predictions ≠ Future forecasts
3. **Clear Labeling**: Use `prediction_type` column to distinguish rows
4. **Visualization Appropriateness**: Different plots for different task types

## Files Modified

1. `conversational/code/stage4_agent.py` - Forecast generation logic
2. `conversational/code/stage5_agent.py` - Task-aware visualization instructions
3. `conversational/tools/stage5_tools.py` - Needs forecast trend plot implementation

## Rollback Instructions

If issues occur:
```bash
cd /scratch/ziv_baretto/llmserve/final_code/conversational
git diff conversational/code/stage4_agent.py
git diff conversational/code/stage5_agent.py
git checkout conversational/code/stage4_agent.py  # revert if needed
```

---

**Date**: 2025-12-08
**Priority**: Critical - Blocks forecasting task visualization
**Status**: Partially Complete (needs Stage 5 tool implementation)
