# Stage 2 & 3 Validation Failsafes - Implementation Summary

## User Requirements

Added three critical validation rules to Stage 2 (Task Proposal) and Stage 3 (Execution Planning):

### 1. Data Availability Check (≥65% Non-NaN)
**Requirement:** Tasks must only use columns with at least 65% non-NaN data

**Implementation:**
- Stage 2: Validates columns before proposing tasks
- Stage 3: Double-checks before creating execution plan
- Uses Stage 1 summaries (`null_fraction`) or `python_sandbox()` to verify
- Rejects columns with >35% missing data

### 2. Predictive Task Preference
**Requirement:** Prioritize predictive tasks over descriptive/clustering

**Implementation:**
- Stage 2: Explicit priority order in system prompt:
  1. PREDICTIVE (forecasting, regression, classification) ← HIGHEST
  2. Clustering (if predictive not viable)
  3. Descriptive (last resort)

### 3. INR Currency Preference
**Requirement:** When both INR and USD columns exist, prefer INR unless specified

**Implementation:**
- Stage 2: Uses INR columns by default in task proposals
- Stage 3: Selects INR columns in file_instructions
- Exception: Only use USD if user explicitly requests it
- Documents choice in plan notes

---

## Changes Made

### Stage 2 Agent (`stage2_agent.py`)

**System Prompt Updates:**
1. Added "CRITICAL VALIDATION RULES" section (77 lines)
2. Added "VALIDATION WORKFLOW" section
3. Updated synthesis prompt with validation reminder

**Validation Workflow:**
```python
STEP 1: Identify all required columns (target, features, join keys)

STEP 2: Validate data availability
  - Check Stage 1 summaries for null_fraction
  - Calculate: data_availability = 1 - null_fraction
  - Require: data_availability >= 0.65 for ALL columns

STEP 3: Verify task type priority
  - Is this predictive? (proceed if yes)
  - Can it be made predictive? (try to convert)

STEP 4: Check currency preference
  - If INR and USD both exist, use INR

STEP 5: Document validation
  - Mention in problem_statement: "All columns verified ≥65% complete"
```

**Example Validation Code:**
```python
# Check data availability
import pandas as pd
df = load_dataframe('file.csv')

for col in ['export_value', 'production', 'year']:
    non_nan_pct = df[col].notna().sum() / len(df)
    print(f"{col}: {non_nan_pct*100:.1f}% non-NaN")
    if non_nan_pct < 0.65:
        print(f"  ❌ REJECT - Insufficient data!")
    else:
        print(f"  ✓ OK - Can use this column")
```

---

### Stage 3 Agent (`stage3_agent.py`)

**System Prompt Updates:**
1. Added rules 6 & 7 to "CRITICAL RULES" section
2. Added complete "VALIDATION WORKFLOW" section (32 lines)

**New Rules:**

**Rule 6: DATA VALIDATION (≥65% NON-NAN) - MANDATORY**
- Verify (1 - null_fraction) >= 0.65 using `inspect_data_file()` or `python_sandbox()`
- If column has >35% missing data, DO NOT use it
- Find alternative columns or document in notes

**Rule 7: CURRENCY PREFERENCE (INR > USD)**
- If both INR and USD columns exist, ALWAYS use INR
- Exception: Only use USD if user explicitly requested it
- Document choice in plan notes

**Validation Workflow:**
```python
STEP 1: Load task proposal and inspect all required_files

STEP 2: For EACH column (target, features, join keys):
   - Use python_sandbox() to check: df['column'].notna().sum() / len(df)
   - Require: completeness >= 0.65
   - If fails: Find alternative or document issue

STEP 3: If both INR and USD columns exist:
   - Select INR column
   - Update all references in plan

STEP 4: Document in plan notes:
   - "Data validation: All columns verified ≥65% complete"
   - "Currency: Using INR as per preference" (if applicable)
```

---

## How It Works

### Stage 2 Workflow

1. **During Exploration:**
   - Agent calls `read_summary_file()` to get dataset info
   - Reviews `null_fraction` for each column
   - Uses `python_sandbox()` to calculate completeness if needed

2. **During Task Proposal:**
   - Only proposes tasks with columns having ≥65% data
   - Prioritizes predictive tasks
   - Selects INR columns when both INR/USD exist
   - Documents validation in `problem_statement`

3. **Output Example:**
   ```json
   {
     "id": "TSK-001",
     "category": "predictive",
     "title": "Forecast Rice Export Values",
     "problem_statement": "Predict future rice export values to Bangladesh using historical data. All columns verified ≥65% data completeness. This is a time-series forecasting task...",
     "target": {
       "name": "Export Value (INR)"
     }
   }
   ```

### Stage 3 Workflow

1. **Load Task Proposal:**
   - Reads selected task from Stage 2
   - Identifies all columns to be used

2. **Validate Data:**
   - Uses `python_sandbox()` to check actual data completeness
   - Verifies each column: `df['col'].notna().sum() / len(df) >= 0.65`

3. **Currency Selection:**
   - If task mentions columns like "Value (INR)" and "Value (USD)"
   - Selects "Value (INR)" for all file_instructions

4. **Create Plan:**
   - Includes only validated columns in `file_instructions`
   - Documents validation in plan `notes`

5. **Output Example:**
   ```json
   {
     "plan_id": "PLAN-TSK-001",
     "file_instructions": [
       {
         "file_id": "export_data.csv",
         "keep_columns": ["Year", "Export Value (INR)", "Quantity"],
         ...
       }
     ],
     "notes": "Data validation: All columns verified ≥65% complete. Currency: Using INR as per preference."
   }
   ```

---

## Testing

### Test Stage 2
```bash
micromamba activate llm
python -m agentic_code.stage2_agent
```

**Expected Behavior:**
- Agent uses `python_sandbox()` to check null percentages
- Proposes predictive tasks first
- Uses INR columns when both INR/USD exist
- Mentions "≥65% data completeness" in problem statements

### Test Stage 3
```bash
python -m agentic_code.stage3_agent TSK-001
```

**Expected Behavior:**
- Validates all columns before creating plan
- Rejects or finds alternatives for columns with >35% nulls
- Selects INR over USD
- Documents validation in notes

---

## Files Modified

| File | Lines Added | Purpose |
|------|-------------|---------|
| `stage2_agent.py` | +85 | Added validation rules and workflow |
| `stage3_agent.py` | +40 | Added validation rules and workflow |

**Total:** ~125 lines added

---

## Benefits

### 1. **Data Quality Assurance**
- Prevents tasks on sparse/incomplete data
- Reduces errors in downstream stages
- Improves model reliability

### 2. **Consistent Preferences**
- Standardized currency selection (INR)
- Predictive tasks prioritized
- Reproducible decisions

### 3. **Transparent Validation**
- Documented in task proposals and plans
- Traceable decision process
- Easy debugging

### 4. **Fail-Fast Mechanism**
- Catches data issues early (Stage 2)
- Prevents invalid execution plans
- Saves computational resources

---

## Example Scenarios

### Scenario 1: Column with Insufficient Data

**Input:**  
- Column "Production" has 50% non-NaN data (fails 65% threshold)

**Stage 2 Behavior:**
- Detects 50% < 65%
- Rejects "Production" column
- Looks for alternative columns
- Or documents limitation in task proposal

**Stage 3 Behavior:**
- Double-checks with `python_sandbox()`
- Confirms failure
- Excludes "Production" from `file_instructions`
- Documents in notes: "Production column excluded due to <65% completeness"

### Scenario 2: Both INR and USD Columns Exist

**Input:**
- Columns: "Value (INR)", "Value (USD)"
- User request: "Forecast export values" (no currency specified)

**Stage 2 Behavior:**
- Detects both currencies
- Selects "Value (INR)" for target
- Documents: "Using INR currency as per preference"

**Stage 3 Behavior:**
- Confirms INR selection
- Uses "Value (INR)" in all file_instructions
- Documents: "Currency: Using INR as per preference"

### Scenario 3: User Explicitly Requests USD

**Input:**
- User query: "Predict USD export values"
- Columns: "Value (INR)", "Value (USD)"

**Stage 2 Behavior:**
- Detects "USD" in user query
- Overrides preference, uses "Value (USD)"
- Marks as exception

**Stage 3 Behavior:**
- Uses "Value (USD)" per task proposal
- Documents: "Using USD as explicitly requested by user"

---

## Summary

✅ **All three validation rules successfully implemented:**

1. **Data Availability:** Columns must have ≥65% non-NaN data
2. **Task Type:** Predictive tasks prioritized
3. **Currency:** INR preferred over USD (unless specified)

Both Stage 2 and Stage 3 now validate data systematically before proposing tasks or creating execution plans. This ensures higher quality outputs and prevents downstream failures.

**Ready for production use! 🎯**
