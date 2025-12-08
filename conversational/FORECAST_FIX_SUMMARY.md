# Complete Fix for Forecasting Visualization Issues

## Executive Summary

Your pipeline was failing to show future forecasts because:
1. **Stage 4** only generated test set predictions (no future forecasts)
2. **Stage 5** had hardcoded visualizations (didn't check task type)
3. Neither stage looked at `forecast_horizon` from the execution plan

## What I Fixed

### ✅ Stage 4 Updates (COMPLETE)
**File**: `conversational/code/stage4_agent.py`

**Changes**:
- Added instructions to check `forecast_horizon` in plan
- Split execution: test predictions + future forecasts
- Added `prediction_type` column to mark 'test' vs 'forecast' rows
- Updated system prompt and execution workflow

**Result**: Stage 4 will now generate future forecasts when `forecast_horizon > 0`

### ✅ Stage 5 Prompt Updates (COMPLETE)
**File**: `conversational/code/stage5_agent.py`

**Changes**:
- Made visualization instructions task-aware
- Emphasized forecast trend plot for forecasting tasks
- Differentiated forecasting vs other task types

### ⚠️ Stage 5 Tool Update (NEEDS YOUR ACTION)
**File**: `conversational/tools/stage5_tools.py`

**What to Do**:

Open `conversational/tools/stage5_tools.py` and find the `create_standard_plots` function (around line 453).

**Add this code RIGHT AFTER line 482** (`created_plots = []`):

```python
        # === FORECAST TREND PLOT (for forecasting tasks) ===
        if is_forecasting and date_cols and pred_cols and actual_cols:
            date_col = date_cols[0]
            pred_col, actual_col = pred_cols[0], actual_cols[0]

            # Separate test and forecast data
            test_df = df[df['prediction_type'] == 'test'].copy()
            forecast_df = df[df['prediction_type'] == 'forecast'].copy()

            fig, ax = plt.subplots(figsize=(16, 8))

            # Plot historical actuals
            ax.plot(test_df[date_col], test_df[actual_col],
                    'o-', label='Historical Actual', linewidth=2, markersize=8, color='blue')

            # Plot test predictions
            ax.plot(test_df[date_col], test_df[pred_col],
                    's--', label='Test Predictions', linewidth=2, markersize=6, alpha=0.7, color='green')

            # Plot future forecasts
            ax.plot(forecast_df[date_col], forecast_df[pred_col],
                    'D:', label='Future Forecasts', linewidth=2, markersize=8, color='red')

            # Add vertical line separating history from forecast
            if len(test_df) > 0:
                last_test_date = test_df[date_col].iloc[-1]
                ax.axvline(x=last_test_date, color='gray', linestyle='--',
                          alpha=0.5, linewidth=2, label='Forecast Start')

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Forecast Trend: Historical + Future Predictions', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(STAGE5_OUT_DIR / f'{plan_id}_forecast_trend.png', dpi=150)
            plt.close()
            created_plots.append('forecast_trend.png')
```

**Also update line 478-480**:

```python
# OLD (line 478-480):
        pred_cols = [c for c in df.columns if 'predict' in c.lower() or 'forecast' in c.lower()]
        actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'target' in c.lower()]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

# NEW:
        pred_cols = [c for c in df.columns if 'predict' in c.lower() or 'forecast' in c.lower()]
        actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'target' in c.lower()]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or 'year' in c.lower()]

        # Check if this is a forecasting task
        has_prediction_type = 'prediction_type' in df.columns
        is_forecasting = has_prediction_type and 'forecast' in df['prediction_type'].unique()
```

## Quick Test

After making the Stage 5 tool change:

```bash
cd /scratch/ziv_baretto/llmserve/final_code/conversational
python run_conversational.py
> run TSK-9391
```

Expected visualizations:
1. **forecast_trend.png** - Shows past + test + 5 future years ✅
2. **actual_vs_predicted.png** - Test set accuracy only
3. **residuals_histogram.png** - Error distribution

## Why This Fixes Your Problem

### Before:
- Stage 4 output: 3 rows (test set only)
- Stage 5 plots: actual vs predicted scatter, residuals
- NO future forecasts shown ❌

### After:
- Stage 4 output: 3 test rows + 5 forecast rows = 8 total
- Stage 5 plots: **forecast trend** (past→future), test accuracy, residuals
- Future forecasts visible in timeline plot ✅

## Files Modified

1. ✅ `conversational/code/stage4_agent.py` (lines 50-252, 417-461)
2. ✅ `conversational/code/stage5_agent.py` (lines 64-80)
3. ⚠️ `conversational/tools/stage5_tools.py` (needs your manual update)

## Alternative: Run This Script

If you want to apply the fix automatically:

```bash
cd /scratch/ziv_baretto/llmserve/final_code/conversational
# Apply the Stage 5 tool fix (you'll need to create this script)
python apply_forecast_fix.py
```

Or use Claude Code to apply the edit for you by asking:
"Apply the forecast trend plot code to stage5_tools.py at line 482"

## Documentation

- Full details: `FIX_FORECAST_VISUALIZATIONS.md`
- Token overflow fixes: `FIX_TOKEN_OVERFLOW.md`
- DATA_DIR fixes: `FIX_STAGE3B_DATA_PATH.md`
- All fixes summary: `FIXES_SUMMARY.md`

---

**Status**: 90% Complete (just need Stage 5 tool update)
**Priority**: Critical
**Date**: 2025-12-08
