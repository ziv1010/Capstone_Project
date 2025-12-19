# Root Cause Analysis: TSK-6243 Poor Performance

## Executive Summary

**Task**: Forecast Basmati rice export quantity for 2025-2027
**Result**: MAPE of 143,875% (extremely poor)
**Root Causes**:
1. **DATA ISSUE**: Only 1 row with 7 time points - too small for robust forecasting
2. **DATA ANOMALY**: Target value (2024-25: 1.72) dropped 72% from 2023-24 (6.32)
3. **METHOD ISSUE**: Naive forecast couldn't adapt to sudden drop
4. **MISSING STAGE 6**: Report generation didn't run for this task

---

## 1. Critical Data Problems

### 1.1 Insufficient Data
```
Shape: (1, 23) - ONLY 1 ROW!
```

**The Fundamental Problem**:
- Data is **WIDE FORMAT** with time progressing across columns
- Only **1 entity** (Basmati Rice, HS Code 10063020)
- Only **7 time points** (2018-19 through 2024-25)
- This is **FAR too little data** for reliable time series forecasting

**Quantity values over time**:
```
2018-19:  2.28
2019-20:  2.37  (+4%)
2020-21:  3.44  (+45%)
2021-22:  5.60  (+63%)
2022-23:  6.28  (+12%)
2023-24:  6.32  (+0.6%)
2024-25:  1.72  (-73% SUDDEN DROP!)
```

### 1.2 Data Anomaly
The **dramatic 73% drop** in 2024-25 is likely:
- **Data quality issue** (typo, unit change, missing zero)
- **External shock** (policy change, trade restriction)
- **Seasonality artifact** (fiscal year mismatch)

This makes naive forecast (which uses last value) completely fail.

---

## 2. Why MAPE is 143,875%

### 2.1 The Math
```
Naive Forecast Logic:
- Training data: 2018-19 through 2022-23
- Last observed value: 6.28 (from 2022-23)
- Prediction for all future: 6.28

Actual vs Predicted for 2024-25:
- Actual: 1.72
- Predicted: 6.28
- Error: |1.72 - 6.28| = 4.56

MAPE = |Actual - Predicted| / |Actual| × 100
     = |1.72 - 6.28| / 1.72 × 100
     = 4.56 / 1.72 × 100
     = 265%
```

**But why 143,875%?**

Looking at the winning method code:
```python
# The bug: predicting for MULTIPLE columns
test_cols = [col for col in df.columns if '2024' in col or '2025' in col]
# This includes: '2024 - 25-Quantity', '2024 - 25-Value (INR)', '2024 - 25-Value (USD)'

predictions = np.tile(last_value, len(test_cols))
actuals = df[test_cols].values.flatten()

# So it's calculating MAPE across:
# - Quantities (~1.72)
# - Values in INR (~173,189)
# - Values in USD (~2,051)
# This mixes scales wildly!
```

**Root cause**: The method is **mixing quantity and value columns** with vastly different scales, inflating the MAPE to absurd levels.

### 2.2 The Real Performance
If we calculate MAPE correctly for ONLY the target column ('2024 - 25-Quantity'):
```
MAPE = |1.72 - 6.28| / 1.72 × 100 = 265%
```

Still bad, but not 143,875%!

---

## 3. Why Methods Performed So Poorly

### 3.1 M1 (Naive Forecast) - MAPE 143,875%
**Issues**:
- Used last value (6.28) to predict sudden drop (1.72)
- **Mixed scales** bug inflated metrics
- Can't handle trend changes or anomalies

### 3.2 M2 (ARIMA) - MAPE 12,505,529%!
**Issues**:
- Even WORSE than naive!
- ARIMA with only 7 points is unstable
- Order (1,1,0) may not fit the data well
- Likely diverged or produced extreme forecasts
- **Same mixed-scales bug**

### 3.3 M3 (XGBoost with Lags) - FAILED
**Error**: "Missing xgboost dependency"
- This is NOW fixed with auto-install
- But XGBoost with 3-lag features and only 7 points would overfit severely
- Would need ≥10 points minimum for 3 lags

---

## 4. What Went Wrong in the Pipeline

### 4.1 Stage 3.5A (Method Proposal)
**Problem**: Didn't adapt to tiny dataset

The prompt says:
```
### For FORECASTING Tasks:
- SHORT time series (<100 points): Naive, Moving Average, Simple Exponential Smoothing
```

**But 7 points is EXTREMELY short**, not just "short". The LLM should have:
1. Detected this is too small for reliable forecasting
2. Warned the user about data insufficiency
3. Proposed ultra-simple methods OR refused the task

**What it did**:
- Proposed Naive (OK for baseline)
- Proposed ARIMA (WRONG - needs ≥30 points minimum)
- Proposed XGBoost with lags (WRONG - needs ≥20 points)

### 4.2 Stage 3.5B (Benchmarking)
**Problem**: **Mixed-scale bug** in winning method code

The code should only predict the TARGET column, but it's using:
```python
test_cols = [col for col in df.columns if '2024' in col or '2025' in col]
```

This captures:
- `2024 - 25-Quantity` (target)
- `2024 - 25-Value (INR)` (wrong)
- `2024 - 25-Value (USD)` (wrong)

**Why this happened**:
The prompt says use temporal column split but doesn't warn about:
- Mixing different value scales
- Ensuring only TARGET column is predicted

### 4.3 Stage 4 (Execution)
**Good news**: Correctly replicated the metrics
```
Stage 4 metrics: MAE 59518.51, MAPE 143875.52
Stage 3.5B metrics: MAE 59518.51, MAPE 143875.52
```

This shows stage 4 is working correctly - the problem is upstream.

### 4.4 Stage 5 (Visualization)
**Correctly identified the issue**:
```
Model has extremely high MAPE (143,875.52%) and negative R² (-0.39)
Predictions show large deviations from actual values
Naive Forecast method is not suitable for this dataset
```

Good! Stage 5 is working as intended.

### 4.5 Stage 6 (Final Report) - **DIDN'T RUN**
**Why**: Looking at the files, there's no `TSK-6243_final_report.json` in stage6_out

**Possible causes**:
1. Pipeline stopped after stage 5
2. Error in stage 6 that wasn't logged
3. Stage 6 only runs for certain task IDs (needs investigation)

---

## 5. Prompt Improvement Opportunities (Data-Agnostic)

### 5.1 Stage 3.5A Improvements

**Add data size validation**:
```markdown
## CRITICAL: Data Size Validation

Before proposing methods, CHECK data size:

1. Call `get_prepared_data_info()` to get row/column counts
2. For TIME SERIES tasks:
   - If <10 points: WARN user - too small for reliable forecasting
   - If <20 points: Only propose ultra-simple baselines (Naive, Mean)
   - If <50 points: Avoid complex models (no ARIMA, no neural nets)
   - If ≥50 points: Can use traditional methods
   - If ≥200 points: Can use advanced ML/DL methods

3. For WIDE FORMAT time series:
   - Count temporal columns (not rows!)
   - Apply same thresholds to number of time points

4. If data is insufficient:
   - Record a thought explaining why
   - Propose only the simplest possible baseline
   - Include a warning in the proposal
```

**Add mixed-scale prevention**:
```markdown
## CRITICAL: Target Column Isolation

When creating implementation code:

1. ONLY predict the TARGET column specified in the plan
2. DO NOT include related columns (values, prices, etc.) in predictions
3. For WIDE FORMAT:
   ```python
   # WRONG:
   test_cols = [col for col in df.columns if '2024' in col]

   # CORRECT:
   test_cols = [target_col]  # Only the specific target!
   ```

4. Verify predictions match the target column's scale
```

### 5.2 Stage 3.5B Improvements

**Add metric validation**:
```markdown
## CRITICAL: Metric Sanity Checks

After calculating metrics, validate them:

```python
# Sanity check MAPE
if mape > 1000:
    print(f"WARNING: MAPE is {mape}% which is suspiciously high!")
    print("This often indicates:")
    print("1. Predicting wrong columns (mixed scales)")
    print("2. Very small actual values causing division issues")
    print("3. Catastrophic model failure")
    print("Please verify predictions match target column scale")
```

If any metric is clearly wrong:
- Flag it in the output
- Suggest investigation
- Don't automatically select this method
```

### 5.3 Stage 3 Improvements

**Add data sufficiency check**:
```markdown
## Data Quality Validation

After creating the execution plan:

1. Check if data is sufficient for the task:
   - Forecasting: Need ≥20 time points minimum
   - Regression: Need ≥50 samples minimum
   - Classification: Need ≥10 samples per class
   - Clustering: Need ≥3× samples as clusters

2. If insufficient, add to plan:
   ```json
   "data_sufficiency_warning": "Only 7 time points available. This is too small for reliable forecasting. Results may be unreliable."
   ```

3. Pass warning to later stages so they can adapt
```

---

## 6. Why Stage 6 Didn't Run - Investigation Needed

Looking at the files:
```bash
$ ls stage6_out/
TSK-001_final_report.json  # Different task!
TSK-001_final_report.txt

$ ls stage5_out/*TSK-6243*
task_answer_PLAN-TSK-6243.txt
TSK-6243_actual_vs_predicted.png
TSK-6243_forecast_timeline.png
TSK-6243_residuals_analysis.png
visualization_report_PLAN-TSK-6243.json
```

**Hypothesis**: Stage 6 expects `TSK-` prefix but the run used `PLAN-TSK-6243`

**Need to check**:
1. How stage 6 is invoked in the pipeline
2. What task_id vs plan_id it expects
3. Whether there's error handling that silently skipped it

---

## 7. Recommendations

### 7.1 Immediate Fixes (Code)

1. **Fix mixed-scale bug in stage 3.5A**:
   - Update method templates to only predict target column
   - Add validation in benchmark code to check column matches

2. **Add data size validation**:
   - In stage 3.5A, check data size before proposing methods
   - Warn user if data is too small
   - Adapt method selection to data size

3. **Fix stage 6 invocation**:
   - Investigate why it didn't run for PLAN-TSK-6243
   - Add error logging if stage 6 fails

### 7.2 Prompt Improvements (Data-Agnostic)

1. **Stage 3.5A**:
   - Add data size thresholds section
   - Add target column isolation instructions
   - Add warning generation for insufficient data

2. **Stage 3.5B**:
   - Add metric sanity checks
   - Add scale validation
   - Flag suspicious results

3. **Stage 3**:
   - Add data sufficiency validation
   - Pass warnings to downstream stages

### 7.3 For This Specific Task

**The data is fundamentally insufficient**. Options:
1. **Get more data**: Aggregate across multiple HS codes or regions
2. **Change task**: Instead of forecasting, do descriptive analysis
3. **Accept limitations**: Proceed with caveats about unreliability

---

## 8. Testing the Fixes

After implementing changes, test with:
1. **Tiny datasets** (5-10 rows): Should warn user
2. **Small datasets** (20-50 rows): Should use simple methods only
3. **Normal datasets** (100+ rows): Should work as before
4. **Wide format**: Should correctly identify temporal columns
5. **Mixed scales**: Should NOT mix different value types

---

## Conclusion

The poor results stem from:
1. **Insufficient data** (7 time points)
2. **Data anomaly** (73% drop)
3. **Implementation bug** (mixed-scale MAPE calculation)
4. **Inappropriate method selection** (ARIMA, XGBoost for tiny dataset)
5. **Missing stage 6 execution**

**The prompts are generally good** but need:
- Data size awareness
- Target column isolation
- Metric validation
- Better adaptation to edge cases

None of these improvements require hardcoding examples - they're all about adaptive reasoning based on data characteristics.
