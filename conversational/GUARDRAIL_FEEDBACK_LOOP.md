# Guardrail Feedback Loop Enhancement

## Overview
Enhanced the guardrail system with an intelligent **feedback loop** that detects issues (especially hallucinations) and sends detailed feedback to stage agents for correction, allowing them to retry and fix problems automatically.

## Architecture

### Before (Simple Validation)
```
Stage → Guardrail → Log Issues → Continue (regardless of failures)
```

### After (Intelligent Feedback Loop)
```
Stage → Guardrail → Detect Issues → Send Feedback to Stage → Stage Retries with Feedback → Guardrail Validates Again → Pass or Fail After Max Retries
```

## Key Features

### 1. **Actionable Feedback Generation**
Guardrails now generate specific, actionable feedback for agents:
- **What went wrong**: Detailed description of failed checks
- **Why it's a problem**: Explanation of the issue severity
- **How to fix it**: Specific suggestions for correction

### 2. **Hallucination Detection**
Advanced checks to detect when LLMs are fabricating data:

**Stage 4 (Execution Results)**:
- ✓ Predictions vs actual data comparison (detects wildly different values)
- ✓ Variance analysis (detects unrealistic prediction spread)
- ✓ Constant prediction detection (detects when model outputs same value)
- ✓ Missing results detection (detects fabricated file paths)

**Stage 5 (Visualizations)**:
- ✓ Insight grounding check (ensures insights reference actual metrics)
- ✓ Non-existent metric detection (catches references to non-existent data)
- ✓ Excessive precision detection (flags too many specific numbers)
- ✓ Claims without evidence (detects unsupported predictions)

### 3. **Intelligent Retry Logic**
- **Max Retries**: 2 retries per stage (total 3 attempts)
- **Feedback Injection**: Failed attempt feedback stored in `state.guardrail_reports["{stage}_feedback"]`
- **Attempt Tracking**: Each retry saved with attempt number for debugging
- **Graceful Degradation**: After max retries, pipeline continues with logged failures

## How It Works

### Step-by-Step Flow

1. **Stage Execution**
   ```python
   state = STAGE_NODES[stage](state)
   ```

2. **Guardrail Validation**
   ```python
   guardrail = STAGE_GUARDRAILS[stage](stage)
   report = guardrail.validate(stage_output, state)
   ```

3. **Issue Detection**
   ```python
   if report.overall_status == "failed" and report.requires_retry:
       # Critical issues detected
   ```

4. **Feedback Generation**
   ```python
   feedback_for_agent = """
   GUARDRAIL FEEDBACK - Critical issues detected in stage4:
   - hallucination_detection: Verify model is using actual data, not generating random values
   - constant_predictions: Model appears to be outputting same value repeatedly

   Please address these issues and regenerate the output.
   """
   ```

5. **Stage Retry with Feedback**
   ```python
   state.guardrail_reports[f"{stage}_feedback"] = feedback_for_agent
   state = STAGE_NODES[stage](state)  # Retry with feedback
   ```

6. **Revalidation**
   ```python
   # Guardrail runs again on new output
   report = guardrail.validate(new_output, state)
   ```

## Enhanced Guardrail Reports

### New Fields in StageGuardrailReport

```python
class StageGuardrailReport(BaseModel):
    # Existing fields
    stage_name: str
    overall_status: str  # "passed", "warning", "failed"
    checks: List[GuardrailCheckResult]
    execution_time_ms: float
    timestamp: datetime

    # NEW: Feedback loop fields
    requires_retry: bool                     # Whether stage should retry
    feedback_for_agent: Optional[str]        # Detailed feedback for agent
    failed_checks_summary: List[str]         # Summary of all failures
```

### Example Feedback Message

```
GUARDRAIL FEEDBACK - Critical issues detected in stage4:

- hallucination_detection: Predictions appear hallucinated - mean prediction (1234.56) is 15.3 standard deviations from actual data mean (89.23). Verify model is using actual data, not generating random values. Retrain model with correct data.

- variance_hallucination: Prediction variance (500.00) is 0.05x actual variance (10000.00) - suggests hallucinated data. Check if model is generating synthetic data instead of real predictions.

- constant_predictions: Predictions are nearly constant (std=0.0001) - likely hallucinated. Model appears to be outputting same value repeatedly. Retrain with diverse data.

Please address these issues and regenerate the output.
```

## Hallucination Detection Checks

### Stage 4: Execution Results

#### 1. **Prediction vs Actual Comparison**
```python
if abs(mean_prediction - mean_actual) > 10 * std_actual:
    # Hallucination detected: predictions wildly different from actuals
    suggestion = "Verify model is using actual data, not generating random values"
```

#### 2. **Variance Ratio Analysis**
```python
variance_ratio = std_prediction / std_actual
if variance_ratio > 5 or variance_ratio < 0.1:
    # Hallucination detected: unrealistic variance
    suggestion = "Check if model is generating synthetic data"
```

#### 3. **Constant Predictions**
```python
if std_prediction < 0.01 * abs(mean_prediction):
    # Hallucination detected: nearly constant predictions
    suggestion = "Model outputting same value repeatedly. Retrain with diverse data."
```

#### 4. **Missing Results File**
```python
if not results_file.exists():
    # Possible hallucination: fabricated file path
    suggestion = "Ensure Stage 4 actually saves results. Check if file path is hallucinated."
```

### Stage 5: Visualization Insights

#### 1. **Metric Reference Validation**
```python
if "MAE" in insights_text and not execution_result.metrics:
    # Hallucination detected: referencing non-existent metrics
    suggestion = "Ensure insights are grounded in actual execution results, not fabricated"
```

#### 2. **Data Column Validation**
```python
if "accuracy" in insights_text and "actual" not in results_df.columns:
    # Hallucination detected: referencing non-existent data
    suggestion = "Insights reference data that doesn't exist"
```

#### 3. **Claims Without Evidence**
```python
if "predict" in answer_text and not execution_result.metrics:
    # Hallucination detected: predictions claims without supporting data
    suggestion = "Ensure task answer is based on actual execution results"
```

#### 4. **Excessive Precision**
```python
numbers = extract_numbers(answer_text)
if len(numbers) > 10:
    # Possible hallucination: too many specific numbers
    suggestion = "Cross-check all cited numbers against execution results"
```

## Usage Examples

### Scenario 1: Hallucinated Predictions Detected

```
[Stage 4 executes]
🛡️  Running guardrail for stage4 (stage attempt 1)

❌ Guardrail FAILED with 2 critical issues:
  - hallucination_detection: Predictions appear hallucinated - mean prediction (5000.00) is 25.0 standard deviations from actual data mean (200.00)
  - variance_hallucination: Prediction variance (10.00) is 0.001x actual variance (10000.00) - suggests hallucinated data

⚠️  Guardrail requests retry for stage4
ℹ️  Feedback for agent:
    GUARDRAIL FEEDBACK - Critical issues detected in stage4:
    - hallucination_detection: Verify model is using actual data, not generating random values. Retrain model with correct data.
    - variance_hallucination: Check if model is generating synthetic data instead of real predictions

    Please address these issues and regenerate the output.

🔄 Retrying stage4 (attempt 2) with guardrail feedback

[Stage 4 re-executes with feedback]
🛡️  Running guardrail for stage4 (stage attempt 2)
✅ Guardrail stage4: passed

Pipeline continuing...
```

### Scenario 2: Max Retries Reached

```
[Stage 5 executes]
🛡️  Running guardrail for stage5 (stage attempt 1)

❌ Guardrail FAILED with 1 critical issue:
  - insight_hallucination: Insights reference metrics that don't exist in execution results

🔄 Retrying stage5 (attempt 2) with guardrail feedback
🛡️  Running guardrail for stage5 (stage attempt 2)

❌ Guardrail FAILED again

🔄 Retrying stage5 (attempt 3) with guardrail feedback
🛡️  Running guardrail for stage5 (stage attempt 3)

❌ Guardrail FAILED after 3 attempts
⚠️  Max retries reached for stage5, continuing despite failures

Pipeline continuing with warnings...
```

## Configuration

### Retry Limits
```python
# In master_orchestrator.py
max_stage_retries = 2           # Stage can retry twice (3 total attempts)
max_guardrail_attempts = 2       # Guardrail execution can retry once (2 attempts)
```

### Strictness Thresholds
```python
# Hallucination detection thresholds
PREDICTION_DEVIATION_THRESHOLD = 10  # sigma from actual mean
VARIANCE_RATIO_MIN = 0.1             # Min acceptable variance ratio
VARIANCE_RATIO_MAX = 5.0             # Max acceptable variance ratio
CONSTANT_PREDICTION_THRESHOLD = 0.01 # Relative std for constant detection
```

## Benefits

1. **Automatic Error Correction**: Agents fix issues without manual intervention
2. **Hallucination Prevention**: Detects and prevents fabricated data from propagating
3. **Quality Assurance**: Multiple validation passes ensure output quality
4. **Debugging Aid**: Attempt-based reports help trace issue evolution
5. **Transparency**: Users see exactly what was wrong and how it was fixed

## Files Modified

### 1. `code/guardrails.py`
**Changes:**
- Added `requires_retry`, `feedback_for_agent`, `failed_checks_summary` to `StageGuardrailReport`
- Enhanced `generate_report()` to create actionable feedback
- Added hallucination detection to `Stage4Guardrail`:
  - Prediction vs actual comparison
  - Variance ratio analysis
  - Constant predictions detection
  - Missing file detection
- Added hallucination detection to `Stage5Guardrail`:
  - Metric reference validation
  - Data column validation
  - Claims without evidence
  - Excessive precision detection

### 2. `code/master_orchestrator.py`
**Changes:**
- Implemented retry loop for stages with guardrail feedback
- Added `max_stage_retries = 2` configuration
- Store feedback in `state.guardrail_reports[f"{stage}_feedback"]`
- Save attempt-specific reports: `guardrail_{stage}_{plan_id}_attempt{N}.json`
- Enhanced logging with emoji indicators (🛡️ 🔄 ✅ ❌)
- Break retry loop on max attempts or guardrail pass

## Accessing Feedback Through Conversation

Users can query feedback through the conversation interface:

```
User: "Why did the pipeline retry stage 4?"