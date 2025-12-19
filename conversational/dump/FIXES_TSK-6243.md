# Implementation Fixes for TSK-6243 Issues

## Summary of Issues

1. **Mixed-scale bug**: Method code mixes quantity and value columns causing inflated MAPE
2. **Insufficient data handling**: No validation for tiny datasets (7 time points)
3. **Inappropriate method selection**: ARIMA/XGBoost proposed for 7-point time series
4. **Stage 6 execution**: Requires manual invocation (design choice, not a bug)

---

## Fix 1: Mixed-Scale Bug in Stage 3.5A Prompt

### File: `conversational/code/stage3_5a_agent.py`

### Problem
The winning method code has this bug:
```python
# WRONG: Captures ALL columns with '2024' or '2025'
test_cols = [col for col in df.columns if '2024' in col or '2025' in col]
# This gets: '2024-25-Quantity', '2024-25-Value (INR)', '2024-25-Value (USD)'

actuals = df[test_cols].values.flatten()  # Mixes different scales!
```

### Solution
Add this section to the `STAGE35A_SYSTEM_PROMPT` after line 152:

```markdown
## CRITICAL: Target Column Isolation for WIDE FORMAT Data

When working with WIDE FORMAT time series (time progresses across columns):

**PROBLEM TO AVOID**:
```python
# ❌ WRONG: This mixes different value types!
test_cols = [col for col in df.columns if '2024' in col]
# Gets: '2024-Quantity', '2024-Value-INR', '2024-Value-USD' <- MIXED SCALES!

actuals = df[test_cols].values.flatten()
predictions = some_model.predict()
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100  # WRONG MAPE!
```

**CORRECT APPROACH**:
```python
# ✅ CORRECT: Only use the TARGET column specified in the plan!
target_col = '2024-25-Quantity'  # From execution plan

# For wide format, predict ONLY the target:
last_train_value = df[train_cols[-1]].values  # Last observed
prediction = last_train_value  # or model.predict()

# Get actual from target column only
actual = df[target_col].values

# Calculate metrics on matching scales
mape = np.mean(np.abs((actual - prediction) / actual)) * 100  # CORRECT!
```

**RULES**:
1. **ALWAYS** use `target_col` from the execution plan
2. **NEVER** substring match years/periods to get test columns (e.g., '2024' in col)
3. **VERIFY** predictions and actuals are the same shape and scale
4. For WIDE format:
   - Training columns: Earlier time periods
   - Target column: ONE specific future column (from plan)
   - Prediction: Must match target column's scale exactly

**VALIDATION**:
Before calculating metrics, add this check:
```python
# Validate scales match
if abs(np.mean(predictions) - np.mean(actuals)) / np.mean(actuals) > 100:
    print("WARNING: Predictions and actuals have vastly different scales!")
    print(f"Prediction mean: {np.mean(predictions)}")
    print(f"Actual mean: {np.mean(actuals)}")
```
```

**Insertion Point**: After line 152 ("**KEY PRINCIPLE**: Split strategy should follow...")

---

## Fix 2: Data Size Validation in Stage 3.5A

### File: `conversational/code/stage3_5a_agent.py`

### Problem
No validation for insufficient data. 7 time points is too small for ARIMA or XGBoost.

### Solution
Add this section to the `STAGE35A_SYSTEM_PROMPT` after line 98 (Selection Process):

```markdown
## CRITICAL: Data Size Validation and Method Selection

Before proposing methods, CHECK if there's enough data:

### Step 1: Assess Data Volume
Call `load_plan_and_data()` and check:
- For ROW-BASED time series: Number of rows
- For COLUMN-BASED time series: Number of temporal columns

### Step 2: Apply Data Size Thresholds

**For TIME SERIES / FORECASTING:**
- **< 10 time points**: CRITICAL - Too small for reliable forecasting
  - M1: Naive (last value) or Mean
  - M2: Simple Moving Average (if ≥5 points)
  - M3: Linear trend (if shows clear trend)
  - ❌ DO NOT use: ARIMA, XGBoost, Neural Networks

- **10-20 time points**: VERY LIMITED - Use simple methods
  - M1: Naive or Mean
  - M2: Simple Exponential Smoothing
  - M3: Linear Regression on time
  - ❌ DO NOT use: ARIMA (needs ≥30), XGBoost with lags (needs ≥20)

- **20-50 time points**: LIMITED - Traditional methods only
  - M1: Naive or Moving Average
  - M2: ARIMA (simple orders only: (1,0,0) or (1,1,0))
  - M3: Random Forest with basic features
  - ⚠️ AVOID: Complex models, many parameters

- **50-100 time points**: ADEQUATE - Most methods work
  - M1: Naive or Moving Average
  - M2: ARIMA, Exponential Smoothing
  - M3: XGBoost, Random Forest with lag features

- **100+ time points**: GOOD - All methods available
  - M1: Naive or seasonal naive
  - M2: SARIMA, Prophet, Holt-Winters
  - M3: LSTM, XGBoost, ensemble methods

**For REGRESSION / CLASSIFICATION:**
- **< 50 samples**: Use simple linear/logistic models only
- **< 100 samples**: Traditional ML (trees, SVM) with careful validation
- **100-1000 samples**: All traditional ML methods
- **1000+ samples**: All methods including deep learning

**For CLUSTERING:**
- **< 3× k samples**: Too few (where k = number of clusters)
- Minimum 100 samples recommended for k-means

### Step 3: Document Data Limitations
If data is insufficient, add this to your proposal:
```python
record_observation_3_5a("""
DATA SIZE WARNING:
- Detected only {N} time points
- This is below recommended minimum of {min_recommended}
- Proposed methods are simplified to account for limited data
- Results may have high variance/low reliability
- Recommendation: Collect more data if possible
""")
```

### Step 4: Adapt Method Complexity
Choose methods that match the data size. **DO NOT** propose methods that require more data than available.

**Example**: If only 7 time points:
```python
methods = [
    "M1: Naive Forecast (last value)",
    "M2: Mean Forecast (average of all values)",
    "M3: Linear Trend (fit line through points)"
]
# NOT: ARIMA, XGBoost, Prophet, LSTM
```
```

**Insertion Point**: After line 98 ("### Step 3: Consider the TASK requirements...")

---

## Fix 3: Metric Sanity Checks in Stage 3.5B

### File: `conversational/code/stage3_5b_agent.py`

### Problem
No validation when metrics are clearly wrong (MAPE > 100,000%).

### Solution
Add this to the `STAGE3_5B_SYSTEM_PROMPT` after line 125 (workflow section):

```markdown
## CRITICAL: Metric Sanity Checks

After calculating metrics for each iteration, validate them:

```python
def validate_metrics(metrics: dict, method_name: str):
    """Check if metrics are reasonable."""
    warnings = []

    # MAPE checks
    if 'mape' in metrics:
        mape = metrics['mape']
        if mape > 1000:
            warnings.append(f"MAPE is {mape:.0f}% - suspiciously high!")
            warnings.append("Possible causes:")
            warnings.append("  1. Predicting wrong columns (mixed scales)")
            warnings.append("  2. Very small actual values causing division issues")
            warnings.append("  3. Model completely failed")
            warnings.append("  4. Actual values near zero")

    # MAE vs actual value range check
    if 'mae' in metrics and 'actual_mean' in metrics:
        mae = metrics['mae']
        actual_mean = abs(metrics['actual_mean'])
        if actual_mean > 0 and mae / actual_mean > 10:
            warnings.append(f"MAE ({mae:.2f}) is {mae/actual_mean:.1f}x the mean actual value")
            warnings.append("Model predictions are very far from actual values")

    # R² checks
    if 'r2' in metrics:
        r2 = metrics['r2']
        if r2 < -1:
            warnings.append(f"R² is {r2:.2f} - worse than predicting mean!")

    if warnings:
        print(f"\n⚠️  METRIC VALIDATION WARNINGS for {method_name}:")
        for w in warnings:
            print(f"  {w}")
        print()

    return len(warnings) > 0

# Use it after calculating metrics:
has_issues = validate_metrics(
    {'mape': mape, 'mae': mae, 'r2': r2, 'actual_mean': np.mean(actual)},
    method_name
)
if has_issues:
    record_observation_3_5b(f"Validation warnings detected for {method_name}")
```

**When to flag a method as invalid**:
- MAPE > 10,000%: Almost certainly a bug
- R² < -10: Model is extremely bad
- MAE > 100× actual mean: Wrong scale or catastrophic failure

Document these issues in the method result and consider marking it as not valid.
```

**Insertion Point**: After line 125 in workflow section

---

## Fix 4: Improved Method Templates

### File: `conversational/tools/stage3_5a_tools.py`

### Location: `get_method_templates()` function

Add a template for very small datasets:

```python
templates['ultra_simple_forecasting'] = {
    "name": "Ultra-Simple Forecasting (for <10 time points)",
    "methods": {
        "naive": """
def predict_naive(train_df, test_df, target_col, date_col, **params):
    '''Naive forecast: use last observed value.'''
    import pandas as pd
    import numpy as np

    if len(train_df) == 0:
        return pd.DataFrame({'predicted': [0]*len(test_df)}).set_index(test_df.index)

    last_value = train_df[target_col].iloc[-1]
    predictions = [last_value] * len(test_df)

    return pd.DataFrame({'predicted': predictions}, index=test_df.index)
""",
        "mean": """
def predict_mean(train_df, test_df, target_col, date_col, **params):
    '''Mean forecast: use average of all training values.'''
    import pandas as pd
    import numpy as np

    if len(train_df) == 0:
        return pd.DataFrame({'predicted': [0]*len(test_df)}).set_index(test_df.index)

    mean_value = train_df[target_col].mean()
    predictions = [mean_value] * len(test_df)

    return pd.DataFrame({'predicted': predictions}, index=test_df.index)
""",
        "linear_trend": """
def predict_linear_trend(train_df, test_df, target_col, date_col, **params):
    '''Fit a simple linear trend and extrapolate.'''
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression

    if len(train_df) < 2:
        # Not enough points for trend
        last_value = train_df[target_col].iloc[-1] if len(train_df) > 0 else 0
        return pd.DataFrame({'predicted': [last_value]*len(test_df)}, index=test_df.index)

    # Create time index
    X_train = np.arange(len(train_df)).reshape(-1, 1)
    y_train = train_df[target_col].values

    # Fit trend
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    X_test = np.arange(len(train_df), len(train_df) + len(test_df)).reshape(-1, 1)
    predictions = model.predict(X_test)

    return pd.DataFrame({'predicted': predictions}, index=test_df.index)
"""
    }
}
```

---

## Fix 5: Documentation Updates

### Create: `conversational/docs/DATA_SIZE_GUIDELINES.md`

```markdown
# Data Size Guidelines for Method Selection

## Time Series / Forecasting

| Time Points | Recommended Methods | Avoid | Notes |
|-------------|-------------------|-------|-------|
| < 10 | Naive, Mean, Linear Trend | ARIMA, ML models | Too small for reliable forecasting |
| 10-20 | Naive, Simple Exp Smoothing, Linear Regression | ARIMA, XGBoost with lags | Use simplest methods |
| 20-50 | Naive, ARIMA (simple), Random Forest | Complex ARIMA, LSTM | Traditional methods only |
| 50-100 | All traditional methods | Deep learning | Good for most tasks |
| 100+ | All methods | None | Sufficient for advanced methods |

## Regression / Classification

| Samples | Recommended | Notes |
|---------|------------|-------|
| < 50 | Linear/Logistic Regression | Use regularization |
| 50-100 | Simple tree models, SVM | Use cross-validation |
| 100-1000 | Random Forest, XGBoost | Good for most ML |
| 1000+ | All methods including DL | Plenty of data |

## Clustering

| Samples | Recommended | Notes |
|---------|------------|-------|
| < 3k | Warning | k = number of clusters |
| < 100 | K-means (small k), hierarchical | Limited clustering |
| 100+ | All methods | Good for clustering |

## Why These Limits?

### ARIMA Needs ≥ 30 Points
- Parameter estimation requires sufficient data
- (p,d,q) orders consume degrees of freedom
- With 7 points and order (1,1,0), you have ~4 effective points

### XGBoost with Lags Needs ≥ 20 Points
- 3 lags = lose 3 points
- Need train/validation/test splits
- Tree depth needs variety in samples

### Deep Learning Needs ≥ 100 Points
- Many parameters to fit
- Needs separate validation set
- Risk of severe overfitting on small data

## What Happens with Too Little Data?

- **High Variance**: Results change dramatically with small data changes
- **Overfitting**: Models memorize rather than learn patterns
- **Unstable**: Metrics vary wildly between runs
- **Unreliable**: Cannot trust predictions

## Recommendations When Data is Limited

1. **Acknowledge limitations** in the output
2. **Use simplest possible methods**
3. **Report uncertainty** (confidence intervals if possible)
4. **Suggest data collection** as next step
5. **Consider alternative approaches** (aggregation, related data sources)
```

---

## Testing the Fixes

### Test Case 1: Tiny Dataset (7 points)
- Should propose only: Naive, Mean, Linear Trend
- Should warn about limited data
- Should NOT propose: ARIMA, XGBoost, Prophet

### Test Case 2: Mixed Scales
- Verify predictions only use target column
- Check MAPE is reasonable
- Validate prediction/actual scales match

### Test Case 3: Normal Dataset (100+ points)
- Should propose standard methods
- No data warnings
- Metrics should be reasonable

---

## Implementation Checklist

- [ ] Update `stage3_5a_agent.py` with target column isolation section
- [ ] Update `stage3_5a_agent.py` with data size validation section
- [ ] Update `stage3_5b_agent.py` with metric sanity checks
- [ ] Update `stage3_5a_tools.py` with ultra-simple templates
- [ ] Create `DATA_SIZE_GUIDELINES.md` documentation
- [ ] Test with TSK-6243 (should warn about 7 points)
- [ ] Test with normal dataset (should work as before)
- [ ] Document changes in CHANGELOG

---

## Expected Impact

**For TSK-6243 specifically**:
- Will propose Naive, Mean, Linear Trend (appropriate for 7 points)
- Will warn user about insufficient data
- MAPE will be ~265% (correct) not 143,875% (bug)
- Will recommend data collection

**For normal datasets**:
- No change in behavior
- Better error detection
- More appropriate method selection
- Better user communication

**Data-agnostic**:
- All fixes are based on data characteristics
- No hardcoded examples
- Adaptive to any dataset size
- Works across all task types

---

## Stage 6 Status

**Finding**: Stage 6 CAN run but requires explicit invocation.

**To run stage 6 for TSK-6243**:
```python
from conversational.code.stage6_agent import run_stage6
result = run_stage6('PLAN-TSK-6243')
```

**Recommendation**: This is likely by design. Stage 6 creates final reports and may be optional depending on use case. Check with pipeline designer if it should auto-run after stage 5.
