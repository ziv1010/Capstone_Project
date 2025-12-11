# Guardrail System - Complete Guide

## Overview

The guardrail system provides **multi-layered validation** for every stage of the AI pipeline. Think of it as a quality control inspector that checks the work at each step, detects issues (including LLM hallucinations), and provides feedback for automatic corrections.

### Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Stage 1   │ → │ Guardrail 1 │ → │   Stage 2   │ → ...
│ (Task Eval) │    │  Validator  │    │ (Data Prep) │
└─────────────┘    └─────────────┘    └─────────────┘
                          ↓
                   ┌─────────────┐
                   │  Issues?    │
                   └─────────────┘
                          ↓
                   YES → Retry with Feedback
                   NO  → Continue Pipeline
```

**Key Principles:**
- **Defense-in-Depth**: Three layers of validation (Input → Process → Output)
- **Automatic Retry**: Up to 2 retries with specific feedback when issues are detected
- **Continue on Failure**: Pipeline continues even after max retries (but issues are logged)
- **Moderate Strictness**: Only severe issues (>70% null values) trigger retries

---

## How Guardrails Work

### 1. Execution Flow

For each stage, the guardrail:

```python
1. ✅ Stage executes (e.g., Stage 4 trains model)
2. 🛡️  Guardrail validates the output
3. 📊 Checks data quality, business logic, accuracy
4. ⚠️  Detects critical issues (if any)
5. 💬 Generates actionable feedback
6. 🔄 Sends feedback to stage for retry (if needed)
7. ✅ Continue or retry based on results
```

### 2. Validation Severity Levels

| Severity | Threshold | Action | Example |
|----------|-----------|--------|---------|
| **Critical** | >70% null values, hallucinations, missing files | Triggers retry | Predictions are 25σ from actual values |
| **Warning** | 35-70% null values, minor issues | Log but continue | 50% of dataset has missing values |
| **Info** | <35% null values, passes checks | Continue | All checks passed |

### 3. Retry Logic

```python
Attempt 1: Stage executes
           ↓
        Guardrail validates
           ↓
    Critical issue detected?
           ↓
    YES → Generate specific feedback
           ↓
Attempt 2: Stage re-executes WITH feedback
           ↓
        Guardrail re-validates
           ↓
    Still failing?
           ↓
    YES → One more retry
           ↓
Attempt 3: Final attempt with feedback
           ↓
        Guardrail final validation
           ↓
    Pass or fail → Continue pipeline
    (All attempts saved for debugging)
```

---

## Stage-by-Stage Guardrails

### Stage 1 Guardrail: Task Evaluation Validator

**Purpose**: Ensure the initial task evaluation is thorough and well-reasoned

**What It Checks:**

#### Input Validation
- ✅ **Task Description Present**: Verifies task description exists and is not empty
  - *Critical if missing*
  - Suggestion: "Provide a task description in the input"

- ✅ **Task Feasibility Assessed**: Checks that feasibility field exists and is valid
  - *Critical if missing*
  - Suggestion: "Analyze task feasibility before proceeding"

#### Process Validation
- ✅ **Reasoning Quality**: Validates that reasoning is substantive (>50 chars)
  - *Warning if too brief*
  - Suggestion: "Provide more detailed reasoning (current: X chars, expected: >50)"

- ✅ **Null Value Detection**: Checks for excessive null/None values in critical fields
  - *Critical if >70% nulls*
  - *Warning if 35-70% nulls*
  - Suggestion: "Complete the required fields in task evaluation"

#### Output Validation
- ✅ **Output Structure**: Ensures all required fields are present
  - *Critical if missing required fields*
  - Suggestion: "Ensure output includes: task_description, feasibility, reasoning"

**Example Report:**
```json
{
  "stage_name": "stage1",
  "overall_status": "passed",
  "checks": [
    {
      "check_name": "task_description_present",
      "check_type": "data_quality",
      "passed": true,
      "severity": "critical",
      "message": "Task description is present and valid"
    }
  ],
  "requires_retry": false
}
```

---

### Stage 2 Guardrail: Data Preparation Validator

**Purpose**: Ensure data is properly loaded, cleaned, and ready for analysis

**What It Checks:**

#### Input Validation
- ✅ **Dataset File Exists**: Verifies the specified dataset file is present
  - *Critical if missing*
  - Suggestion: "Ensure dataset file exists at specified path"

#### Process Validation
- ✅ **Data Loading Success**: Checks if data was loaded successfully
  - *Critical if failed*
  - Suggestion: "Fix data loading errors and retry"

- ✅ **Data Quality**:
  - Row count validation (>0 rows expected)
  - Column count validation (>0 columns expected)
  - Null value percentage per column (<70% for critical)
  - *Critical if dataset is empty or mostly null*
  - Suggestion: "Dataset has X% null values in critical columns - review data quality"

#### Output Validation
- ✅ **Cleaned Data Available**: Ensures cleaned dataset exists
  - *Critical if missing*
  - Suggestion: "Save cleaned dataset before proceeding"

- ✅ **Summary Statistics**: Validates that data summary is generated
  - *Warning if missing*
  - Suggestion: "Generate descriptive statistics for the cleaned dataset"

**Example Checks:**
```python
# Check null percentage
null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100

if null_pct > 70:
    # CRITICAL: >70% nulls
    feedback = "Dataset has 85% null values - check data source and ETL pipeline"
elif null_pct > 35:
    # WARNING: 35-70% nulls
    feedback = "Dataset has 50% null values - consider imputation or feature selection"
```

---

### Stage 3 Guardrail: Execution Plan Validator

**Purpose**: Ensure the analysis/modeling plan is sound and executable

**What It Checks:**

#### Input Validation
- ✅ **Task Context Available**: Verifies task and data info are available
  - *Critical if missing*
  - Suggestion: "Ensure Stage 1 and 2 completed successfully"

#### Process Validation
- ✅ **Plan Completeness**: Checks if execution plan has all required components
  - Target variable defined
  - Features identified
  - Approach specified
  - *Critical if major components missing*
  - Suggestion: "Define target variable and feature set in execution plan"

- ✅ **Plan Feasibility**: Validates the plan is realistic
  - Checks if target variable exists in dataset
  - Checks if features are available
  - *Critical if plan is infeasible*
  - Suggestion: "Target variable 'X' not found in dataset - verify column names"

#### Output Validation
- ✅ **Executable Steps**: Ensures plan has concrete, actionable steps
  - *Warning if steps are vague*
  - Suggestion: "Provide more specific execution steps"

- ✅ **File References**: Validates that referenced files exist
  - *Critical if data files missing*
  - Suggestion: "Ensure cleaned dataset exists before execution"

**Example Validation:**
```python
# Check if target variable exists
if plan.target_variable not in dataset_columns:
    # CRITICAL
    feedback = f"Target variable '{plan.target_variable}' not in dataset. Available columns: {dataset_columns}"
    suggestion = "Update target variable to match an existing column"
```

---

### Stage 3B Guardrail: Data Preparation Validator

**Purpose**: Ensure data is properly prepared for modeling with ZERO null values

**What It Checks:**

#### Input Validation
- ✅ **Execution Plan Exists**: Verifies Stage 3 plan is available
  - *Critical if missing*
  - Suggestion: "Ensure Stage 3 planning completed successfully before data preparation"

- ✅ **Prepared Data Output Exists**: Checks if data preparation completed
  - *Critical if missing*
  - Suggestion: "Ensure Stage 3B agent completes data preparation and saves output"

#### Process Validation
- ✅ **Prepared File Exists**: Verifies the prepared data file was saved
  - *Critical if missing*
  - Suggestion: "Ensure data preparation saves the prepared dataset to the specified path"

- ✅ **Row Count Reasonable**: Checks for excessive data loss during preparation
  - *Warning if >90% data lost*
  - Suggestion: "Review data filters and join logic - significant data loss may indicate issues"

- ✅ **Data Not Empty**: Ensures dataset has rows after preparation
  - *Critical if 0 rows*
  - Suggestion: "Review data filtering logic - dataset should not be empty after preparation"

#### Output Validation (CRITICAL)
- ✅ **ZERO NULL VALUES REQUIREMENT**: Most critical check - prepared data MUST have zero nulls
  - *Critical if ANY nulls found*
  - Suggestion: "ALL null values must be handled via imputation, deletion, or feature engineering. Review missing value strategy and apply appropriate imputation."

- ✅ **Actual Null Verification**: Loads and verifies the saved file has no nulls
  - *Critical if nulls detected in file*
  - Suggestion: "Data file contains nulls - rerun data preparation with proper null handling"

- ✅ **Ready for Modeling Flag**: Validates data is marked as ready
  - *Critical if not ready*
  - Suggestion: "Review data preparation output and ensure all requirements are met before marking ready for modeling"

- ✅ **Row/Column Count Match**: Ensures metadata matches actual saved file
  - *Warning if mismatch*
  - Suggestion: "Ensure row count in metadata matches actual saved file"

**Example Check:**
```python
# CRITICAL: Zero null values requirement
if quality_report.columns_with_nulls:
    # CRITICAL FAILURE
    feedback = f"Prepared data contains null values in {len(columns_with_nulls)} columns"
    suggestion = "ALL null values must be handled. Apply imputation strategy."
```

---

### Stage 3.5A Guardrail: Method Proposal Validator

**Purpose**: Ensure proposed methods are appropriate and use valid column references (no hallucinations)

**What It Checks:**

#### Input Validation
- ✅ **Execution Plan Exists**: Verifies Stage 3 plan is available
  - *Critical if missing*
  - Suggestion: "Ensure Stage 3 planning completed successfully"

- ✅ **Prepared Data Exists**: Checks if Stage 3B completed
  - *Critical if missing*
  - Suggestion: "Ensure Stage 3B data preparation completed successfully"

#### Process Validation
- ✅ **Exactly 3 Methods Proposed**: Validates method count
  - *Critical if not exactly 3*
  - Suggestion: "Propose exactly 3 methods: M1 (baseline), M2 (traditional/statistical), M3 (advanced ML)"

- ✅ **Unique Method IDs**: Checks for duplicate IDs
  - *Critical if duplicates*
  - Suggestion: "Each method must have a unique ID (e.g., M1, M2, M3)"

- ✅ **Method Diversity**: Ensures baseline method included
  - *Warning if no baseline*
  - Suggestion: "Include a baseline method (e.g., naive forecast, mean prediction)"

- ✅ **Implementation Code Present**: Validates each method has code
  - *Critical if code missing or too short*
  - Suggestion: "Provide complete, executable implementation code for {method_name}"

#### Output Validation (CRITICAL - HALLUCINATION CHECKS)
- ✅ **Target Column Hallucination Check**: Verifies target column exists in data
  - *Critical if column doesn't exist*
  - Suggestion: "Use only columns that exist in the data. Call get_actual_columns() to see available columns and update target_column."
  - **Example**: `Target column 'Sales' does NOT exist. Available: ['revenue', 'profit', 'units']`

- ✅ **Date Column Hallucination Check**: Verifies date column exists (if specified)
  - *Critical if column doesn't exist*
  - Suggestion: "If no date column exists, set date_column=None and use df.index."

- ✅ **Feature Columns Hallucination Check**: Verifies all feature columns exist
  - *Critical if any features don't exist*
  - Suggestion: "Use only columns that exist in the data. Call get_actual_columns() to verify column names."
  - **Example**: `Feature columns do NOT exist: ['Year', 'Month']. Available: ['date', 'revenue']`

- ✅ **Column References in Code**: Scans method implementation for invalid column names
  - *Warning if potential hallucinated columns found*
  - Suggestion: "Review implementation code and ensure all column references are valid"

- ✅ **Task Type Appropriateness**: Ensures methods match task category
  - *Warning if task is FORECASTING but method isn't time series*
  - Suggestion: "For forecasting tasks, use time series methods like ARIMA, Prophet, LSTM, etc."

**Example Hallucination Detection:**
```python
# Load prepared data
df = pd.read_csv(prepared_file)
actual_columns = set(df.columns)

# Check target column
if proposal.target_column not in actual_columns:
    # CRITICAL HALLUCINATION DETECTED
    feedback = f"Target column '{proposal.target_column}' does NOT exist"
    suggestion = "Call get_actual_columns() and use only existing columns"
```

---

### Stage 3.5B Guardrail: Benchmarking Validator

**Purpose**: Ensure benchmarking results are consistent and not hallucinated

**What It Checks:**

#### Input Validation
- ✅ **Method Proposals Exist**: Verifies Stage 3.5A completed
  - *Critical if missing*
  - Suggestion: "Ensure Stage 3.5A method proposal completed successfully"

- ✅ **Prepared Data Exists**: Checks if Stage 3B completed
  - *Critical if missing*
  - Suggestion: "Ensure Stage 3B data preparation completed successfully"

#### Process Validation
- ✅ **All Methods Tested**: Ensures all proposed methods were benchmarked
  - *Critical if methods skipped*
  - Suggestion: "Benchmark ALL proposed methods: {method_ids}"

- ✅ **Selected Method Valid**: Verifies selected method was actually tested
  - *Critical if selected method not in tested methods*
  - Suggestion: "Select a method that was actually benchmarked"

- ✅ **Multiple Iterations**: Checks each method ran expected number of times
  - *Warning if fewer iterations*
  - Suggestion: "Run each method {expected} times for consistency validation"

#### Output Validation (CRITICAL - HALLUCINATION CHECKS)
- ✅ **Consistency Check (Coefficient of Variation)**: Detects hallucinated results via variability
  - *Critical if CV > 50%* - High variability suggests hallucination
  - *Warning if CV > 20%* - Moderate variability
  - Suggestion: "Results are inconsistent across runs. Verify method implementation is deterministic or uses proper random seeding. May indicate fabricated results."
  - **Example**: `Method M2 has high variability (CV=67%) - results may be hallucinated or unstable`

- ✅ **Best Method Has Valid Results**: Ensures selected method didn't fail
  - *Critical if best method invalid*
  - Suggestion: "Cannot select a method that failed benchmarking - choose a method with valid results"

- ✅ **Metrics Completeness**: Checks for NaN or missing metrics
  - *Warning if metrics incomplete*
  - Suggestion: "Ensure all evaluation metrics are calculated successfully"

- ✅ **Selection Rationale Present**: Validates rationale exists and is substantive
  - *Warning if missing or too brief*
  - Suggestion: "Provide clear rationale explaining why this method was selected (based on metrics)"

- ✅ **Rationale References Metrics**: Ensures rationale cites actual metric values
  - *Warning if no metrics mentioned*
  - Suggestion: "Rationale should cite actual metric values (e.g., 'Selected M2 because it achieved lowest MAE of 0.15')"

**Example Hallucination Detection:**
```python
# Check consistency across iterations
cv = method_result.coefficient_of_variation

if cv > 0.5:
    # CRITICAL: Results likely hallucinated
    feedback = f"Method has high variability (CV={cv:.2%}) - results may be hallucinated"
    suggestion = "Verify implementation is deterministic. May indicate fabricated results."
elif cv > 0.2:
    # WARNING: Moderate variability
    feedback = f"Method has moderate variability (CV={cv:.2%})"
    suggestion = "Consider increasing iterations or checking for randomness"
```

---

### Stage 4 Guardrail: Model Training Validator (WITH HALLUCINATION DETECTION)

**Purpose**: Ensure model is trained correctly and predictions are not hallucinated

**What It Checks:**

#### Input Validation
- ✅ **Training Data Available**: Verifies training data exists
  - *Critical if missing*
  - Suggestion: "Ensure Stage 2 data preparation completed successfully"

- ✅ **Execution Plan Present**: Checks if modeling plan is available
  - *Critical if missing*
  - Suggestion: "Ensure Stage 3 planning completed successfully"

#### Process Validation
- ✅ **Model Training Success**: Validates model was trained
  - Checks for error messages
  - Verifies model artifacts exist
  - *Critical if training failed*
  - Suggestion: "Fix model training errors: {error_details}"

- ✅ **Predictions File Exists**: Ensures predictions were generated
  - *Critical if missing*
  - Suggestion: "Model must generate predictions file"

#### Output Validation - Hallucination Detection

**🚨 CRITICAL CHECKS FOR HALLUCINATIONS:**

1. **Prediction vs Actual Comparison**
   ```python
   # Check if predictions are wildly different from actual values
   if abs(pred_mean - actual_mean) > 10 * actual_std:
       # CRITICAL: Predictions are hallucinated
       feedback = f"Predictions appear hallucinated - mean prediction ({pred_mean:.2f})
                    is {deviation:.1f} standard deviations from actual data"
       suggestion = "Verify model is using actual data, not generating random values.
                     Retrain model with correct data."
   ```

2. **Variance Ratio Analysis**
   ```python
   # Check if prediction variance is unrealistic
   variance_ratio = pred_std / actual_std
   if variance_ratio > 5 or variance_ratio < 0.1:
       # CRITICAL: Variance suggests hallucinated data
       feedback = f"Prediction variance ({pred_std:.2f}) is {variance_ratio:.2f}x
                    actual variance ({actual_std:.2f}) - suggests hallucinated data"
       suggestion = "Check if model is generating synthetic data instead of real predictions"
   ```

3. **Constant Predictions Detection**
   ```python
   # Check if predictions are suspiciously constant
   unique_ratio = num_unique_predictions / total_predictions
   if unique_ratio < 0.01 and len(predictions) > 100:
       # CRITICAL: Model predicting same value repeatedly
       feedback = f"Model is predicting constant values ({unique_ratio*100:.1f}% unique)
                    - likely not learning patterns"
       suggestion = "Check model training - may be predicting mean/mode constantly.
                     Review feature engineering and model complexity."
   ```

4. **Missing Results File**
   ```python
   # Check if results file exists
   if not results_file.exists():
       # CRITICAL: No output generated
       feedback = "Model execution results file not found"
       suggestion = "Ensure model completes execution and saves results to
                     {expected_path}"
   ```

**Example Hallucination Detection:**
```json
{
  "check_name": "hallucination_detection",
  "check_type": "accuracy",
  "passed": false,
  "severity": "critical",
  "message": "Predictions appear hallucinated - mean prediction (1500.23) is 25.3 standard deviations from actual data mean (45.67)",
  "suggestion": "Verify model is using actual data, not generating random values. Retrain model with correct data.",
  "details": {
    "pred_mean": 1500.23,
    "actual_mean": 45.67,
    "deviation_sigmas": 25.3
  }
}
```

---

### Stage 5 Guardrail: Insights & Visualization Validator

**Purpose**: Ensure insights are accurate, evidence-based, and visualizations are meaningful

**What It Checks:**

#### Input Validation
- ✅ **Model Results Available**: Verifies Stage 4 completed successfully
  - *Critical if missing*
  - Suggestion: "Ensure model training completed before generating insights"

#### Process Validation
- ✅ **Insights Generated**: Checks if insights were created
  - *Critical if missing*
  - Suggestion: "Generate insights based on model results"

- ✅ **Visualization Files Exist**: Validates that visualization files are present
  - *Warning if missing*
  - Suggestion: "Create visualizations to support insights"

#### Output Validation - Hallucination Detection

**🚨 CRITICAL CHECKS FOR HALLUCINATIONS:**

1. **Metric Reference Validation**
   ```python
   # Check if metrics are referenced correctly
   known_metrics = ["accuracy", "precision", "recall", "f1-score", "r2", "mse", "rmse"]

   if metric_mentioned not in known_metrics and metric_mentioned not in results:
       # CRITICAL: Made-up metric
       feedback = f"Insight references unknown metric '{metric_mentioned}' not found in results"
       suggestion = "Only reference metrics that were actually calculated.
                     Available metrics: {known_metrics}"
   ```

2. **Data Column Validation**
   ```python
   # Check if referenced columns exist
   if mentioned_column not in dataset.columns:
       # CRITICAL: Referenced non-existent column
       feedback = f"Insight references column '{mentioned_column}' that doesn't exist in dataset"
       suggestion = "Verify column names against actual dataset.
                     Available columns: {dataset.columns.tolist()}"
   ```

3. **Claims Without Evidence**
   ```python
   # Check for unsubstantiated claims
   claim_keywords = ["significant", "important", "correlation", "trend"]

   if any(keyword in insight.lower() for keyword in claim_keywords):
       if not has_supporting_data(insight):
           # CRITICAL: Unsupported claim
           feedback = "Insight makes claims without statistical evidence"
           suggestion = "Provide specific metrics, p-values, or correlation coefficients
                        to support claims"
   ```

4. **Excessive Precision Detection**
   ```python
   # Check for suspicious precision
   if "99.99%" in insight or "100%" in insight:
       # WARNING: Suspiciously perfect metrics
       feedback = "Insight claims suspiciously perfect metrics (99.99% or 100%)"
       suggestion = "Verify these metrics are accurate - perfect scores are rare in
                     real-world models"
   ```

**Example Checks:**
```json
{
  "check_name": "metric_reference_validation",
  "check_type": "accuracy",
  "passed": false,
  "severity": "critical",
  "message": "Insight references metric 'super_accuracy' not found in model results",
  "suggestion": "Only reference metrics that were actually calculated. Available metrics: ['accuracy', 'precision', 'recall', 'f1-score']",
  "details": {
    "invalid_metric": "super_accuracy",
    "available_metrics": ["accuracy", "precision", "recall", "f1-score"]
  }
}
```

---

## Understanding Guardrail Reports

### Report Structure

Each guardrail generates a JSON report with the following structure:

```json
{
  "stage_name": "stage4",
  "overall_status": "failed",  // "passed", "warning", or "failed"
  "checks": [
    {
      "check_name": "hallucination_detection",
      "check_type": "accuracy",
      "passed": false,
      "severity": "critical",
      "message": "Predictions appear hallucinated...",
      "details": {...},
      "suggestion": "Verify model is using actual data..."
    }
  ],
  "execution_time_ms": 123.45,
  "timestamp": "2024-01-15T10:30:00",
  "requires_retry": true,
  "feedback_for_agent": "GUARDRAIL FEEDBACK - Critical issues detected...",
  "failed_checks_summary": [
    "hallucination_detection: Predictions appear hallucinated..."
  ]
}
```

### Key Fields Explained

| Field | Description | Values |
|-------|-------------|--------|
| `overall_status` | Overall validation result | `passed`, `warning`, `failed` |
| `checks` | List of all validation checks performed | Array of check objects |
| `requires_retry` | Whether stage should retry | `true` if critical failures, `false` otherwise |
| `feedback_for_agent` | Actionable feedback for the agent | Specific instructions on what to fix |
| `failed_checks_summary` | Quick summary of failures | Array of failure messages |

### Reading Check Results

Each check has:
- **check_name**: What was checked (e.g., `hallucination_detection`)
- **check_type**: Category (`data_quality`, `safety`, `business_logic`, `accuracy`)
- **passed**: `true` or `false`
- **severity**: `critical`, `warning`, or `info`
- **message**: Human-readable explanation
- **suggestion**: What to do to fix it
- **details**: Additional technical data

---

## Consolidated Guardrail Report

At the end of the pipeline, a consolidated report is generated:

```json
{
  "plan_id": "PLAN-TSK-001",
  "timestamp": "2024-01-15T10:30:00",
  "overall_status": "warning",
  "total_critical_failures": 0,
  "total_warnings": 2,
  "total_passed": 45,
  "stage_reports": {
    "stage1": {...},
    "stage2": {...},
    "stage3": {...},
    "stage4": {...},
    "stage5": {...}
  },
  "summary": "Pipeline completed with 2 warnings. Review Stage 2 data quality..."
}
```

### Accessing Reports

**Via Conversational Interface:**
```
User: "Show guardrail report for TSK-001"
Agent: [Fetches and displays consolidated report]

User: "What did the Stage 4 guardrail find?"
Agent: [Shows Stage 4 specific checks and results]
```

**Via File System:**
```bash
# Individual stage reports
output/guardrails_out/guardrail_stage1_TSK-001.json
output/guardrails_out/guardrail_stage2_TSK-001.json
...

# Consolidated report
output/guardrails_out/guardrail_report_PLAN-TSK-001.json

# Retry attempt reports
output/guardrails_out/guardrail_stage4_TSK-001_attempt1.json
output/guardrails_out/guardrail_stage4_TSK-001_attempt2.json
```

---

## Feedback Loop in Action

### Example: Hallucination Detected and Fixed

**Attempt 1:**
```
[INFO] Running stage4
[INFO] 🛡️  Running guardrail for stage4 (attempt 1)
[ERROR] ❌ Guardrail stage4: failed

Critical Issues Detected:
  ✗ hallucination_detection: Predictions appear hallucinated - mean prediction
    (1500.23) is 25.3σ from actual data mean (45.67)

  ✗ variance_hallucination: Prediction variance (850.5) is 127.3x actual
    variance (6.68) - suggests hallucinated data

Feedback for Agent:
  - hallucination_detection: Verify model is using actual data, not generating
    random values. Retrain model with correct data.
  - variance_hallucination: Check if model is generating synthetic data instead
    of real predictions

[WARNING] 🔄 Retrying stage4 (attempt 2) with guardrail feedback
```

**Attempt 2:**
```
[INFO] Running stage4 (with feedback injected into state)
[INFO] 🛡️  Running guardrail for stage4 (attempt 2)
[INFO] ✅ Guardrail stage4: passed

All checks passed:
  ✓ hallucination_detection: Predictions are within 2σ of actual values
  ✓ variance_hallucination: Prediction variance is realistic (ratio: 1.15)
  ✓ constant_predictions: Sufficient prediction diversity (78.5% unique)
  ✓ results_file_exists: Model results saved successfully
```

### What Happens During Retry

1. **Feedback Injection**: Guardrail feedback is stored in `state.guardrail_reports[f"{stage}_feedback"]`
2. **Stage Re-execution**: The stage agent receives the feedback and adjusts its approach
3. **Re-validation**: Guardrail runs again on the new output
4. **Decision**: Continue if passed, retry again if still failing (up to max retries)

---

## Check Types Reference

### Data Quality Checks
- Null value detection
- Data type validation
- Required field presence
- File existence verification
- Dataset size validation

### Safety Checks
- Malicious code detection (future)
- Sensitive data exposure (future)
- Resource usage limits (future)

### Business Logic Checks
- Task feasibility
- Plan completeness
- Reference integrity
- Column existence
- Metric availability

### Accuracy Checks
- Hallucination detection
- Variance analysis
- Metric reference validation
- Claims evidence validation
- Prediction sanity checks

---

## Configuration

### Validation Strictness

Currently set to **Moderate**:
```python
# Null value thresholds
CRITICAL_NULL_THRESHOLD = 70  # % - triggers retry
WARNING_NULL_THRESHOLD = 35   # % - logs warning
INFO_NULL_THRESHOLD = 0       # % - informational

# Retry configuration
MAX_STAGE_RETRIES = 2  # 3 total attempts per stage
RETRY_ON_CRASH = True  # Retry once if guardrail crashes
CONTINUE_ON_FAILURE = True  # Pipeline continues after max retries
```

### Enabling/Disabling Guardrails

```python
# In master_orchestrator.py
state = run_pipeline_stages(
    state=state,
    stages=["stage1", "stage2", "stage3", "stage4", "stage5"],
    enable_guardrails=True  # Set to False to disable
)
```

---

## Best Practices

### For Users

1. **Always Check Guardrail Reports**: After task completion, review the consolidated report
   ```
   User: "Show guardrail report for TSK-001"
   ```

2. **Investigate Warnings**: Even if the pipeline completes, warnings indicate potential issues
   ```
   User: "What warnings did Stage 2 have?"
   ```

3. **Review Retry Attempts**: Check why retries occurred to understand data quality issues
   ```bash
   ls output/guardrails_out/*_attempt*.json
   ```

4. **Monitor Hallucination Flags**: Stage 4 and 5 hallucination checks are critical for accuracy
   ```
   User: "Did any stage detect hallucinations?"
   ```

### For Developers

1. **Use Moderate Strictness**: Too strict = false positives, too lenient = missed issues
2. **Implement Specific Suggestions**: Make feedback actionable, not generic
3. **Log All Attempts**: Save retry attempts for debugging and pattern analysis
4. **Test Guardrails Independently**: Run `test_guardrails.py` before deployment
5. **Monitor Retry Rates**: High retry rates indicate systemic issues

---

## Troubleshooting

### Issue: Guardrail Always Fails
**Symptoms**: Stage retries 3 times and still fails
**Causes**:
- Data quality is genuinely poor (>70% nulls)
- Threshold is too strict for your use case
- Bug in the validation logic

**Solutions**:
1. Check the actual data: `cat output/stage2_out/cleaned_data.csv`
2. Review failed checks: `cat output/guardrails_out/guardrail_stageX_*.json`
3. Adjust thresholds if needed (in `guardrails.py`)

### Issue: Hallucination False Positives
**Symptoms**: Guardrail flags hallucination but predictions look correct
**Causes**:
- Statistical thresholds too aggressive
- Unusual but legitimate data distribution
- Missing actual values for comparison

**Solutions**:
1. Review the specific check details in the report
2. Verify actual values are available: `df['actual'].describe()`
3. Adjust sigma thresholds if needed (currently 10σ)

### Issue: No Feedback Generated
**Symptoms**: `requires_retry` is `true` but `feedback_for_agent` is `null`
**Causes**:
- Check failed but no suggestion was provided
- Bug in feedback generation logic

**Solutions**:
1. Ensure all checks have `suggestion` field populated
2. Review `generate_report()` method in `guardrails.py`

---

## Summary

The guardrail system provides comprehensive, automated quality control:

✅ **8 Stage-Specific Guardrails** - Each stage (1, 2, 3, 3b, 3.5a, 3.5b, 4, 5) has targeted validation
✅ **12+ Hallucination Detection Checks** - Catches LLM-fabricated data across multiple stages
✅ **Automatic Retry with Feedback** - Up to 3 attempts with specific guidance
✅ **Moderate Strictness** - Balanced validation (>70% nulls = critical)
✅ **Conversational Access** - Query reports via natural language
✅ **Full Audit Trail** - All attempts and reports saved
✅ **Defense-in-Depth for Stage 3** - Comprehensive validation across planning, data prep, method proposal, and benchmarking

**The system ensures high-quality, accurate outputs while maintaining pipeline reliability and providing transparency into validation decisions.**

---

## Additional Resources

- [GUARDRAILS_IMPLEMENTATION.md](GUARDRAILS_IMPLEMENTATION.md) - Original implementation guide
- [GUARDRAIL_FEEDBACK_LOOP.md](GUARDRAIL_FEEDBACK_LOOP.md) - Technical feedback loop details
- [FEEDBACK_LOOP_SUMMARY.md](FEEDBACK_LOOP_SUMMARY.md) - Implementation summary with examples
- [code/guardrails.py](code/guardrails.py) - Source code with all check logic
- [test_guardrails.py](test_guardrails.py) - Test suite for validation

For questions or issues, review the logs in `logs/` directory or check individual guardrail reports in `output/guardrails_out/`.
