# Excluded Columns Tracking - Implementation Summary

## Overview

Added systematic tracking of excluded columns across Stage 2, Stage 3, and downstream stages so agents understand WHY certain columns were dropped and can make informed decisions.

---

## Changes Made

### 1. Data Models (`models.py`)

**TaskProposal Model:**
```python
excluded_columns: List[Dict[str, str]] = Field(
    default_factory=list,
    description=(
        "Columns that were considered but excluded due to data quality issues. "
        "Each entry has 'column_name', 'file', and 'reason'"
    )
)
```

**Stage3Plan Model:**
```python
excluded_columns: List[Dict[str, str]] = Field(
    default_factory=list,
    description=(
        "Columns excluded during planning due to data quality. "
        "Each entry has 'column_name', 'file', and 'reason'"
    )
)
```

---

### 2. Stage 2 (`stage2_agent.py`)

**Updated Synthesis Prompt:**
- Added `excluded_columns` to required JSON structure
- Agents must document ANY column considered but rejected
- Example format provided in prompt

**Example Output:**
```json
{
  "id": "TSK-001",
  "excluded_columns": [
    {
      "column_name": "Price_USD",
      "file": "export_data.csv",
      "reason": "Only 45% non-NaN data, below 65% threshold. Using Price_INR instead."
    },
    {
      "column_name": "Legacy_ID",
      "file": "production.csv",
      "reason": "90% missing data, unusable for analysis"
    }
  ]
}
```

---

### 3. Stage 3 (`stage3_agent.py`)

**Updated System Prompt:**
- Added `excluded_columns` to Stage3Plan schema documentation
- Required agents to document columns rejected during validation
- Specified exact format: `{"column_name": "...", "file": "...", "reason": "..."}`

**Example Documentation:**
```json
{
  "plan_id": "PLAN-TSK-001",
  "excluded_columns": [
    {
      "column_name": "Production_Tonnes",
      "file": "production.csv",
      "reason": "Only 52% non-NaN data (below 65% threshold). Cannot use reliably."
    },
    {
      "column_name": "Export_USD",
      "file": "export.csv",
      "reason": "Using Export_INR instead per currency preference (both had >65% data)"
    }
  ]
}
```

---

### 4. Stage 3B (`stage3b_agent.py`)

**Added Exclusion Context:**
- Loads Stage 3 plan at startup
- Extracts `excluded_columns` information
- Informs agent via human message about excluded columns

**Agent Receives:**
```
**EXCLUDED COLUMNS FROM STAGE 3:**
The following columns were considered but excluded during planning:
- Production_Tonnes from production.csv: Only 52% non-NaN data (below 65% threshold)
- Export_USD from export.csv: Using Export_INR instead per currency preference

Do NOT attempt to use these columns. They were excluded for data quality reasons.
```

---

### 5. Stage 3.5 (`stage3_5_agent.py`)

**Added Comprehensive Exclusion Context:**
- Loads excluded columns from both Stage 2 task proposal AND Stage 3 plan
- Merges both lists
- Informs agent about all excluded columns

**Agent Receives:**
```
**COLUMNS EXCLUDED DUE TO DATA QUALITY:**
The following columns were rejected in earlier stages:
- Price_USD from export_data.csv: Only 45% non-NaN data, below 65% threshold
- Production_Tonnes from production.csv: Only 52% non-NaN data (below 65% threshold)
- Legacy_ID from production.csv: 90% missing data, unusable for analysis

Be aware these columns are unavailable. Use alternatives if needed.
```

---

## How It Works

### Stage 2 Workflow

1. **During Validation:**
   ```python
   # Agent checks data quality
   completeness = df['Production_Tonnes'].notna().sum() / len(df)
   # completeness = 0.52 (52%)
   
   if completeness < 0.65:
       # Reject column, add to excluded_columns
       excluded_columns.append({
           "column_name": "Production_Tonnes",
           "file": "production.csv",
           "reason": "Only 52% non-NaN data (below 65% threshold)"
       })
   ```

2. **In Task Proposal JSON:**
   - Excluded columns documented in `excluded_columns` field
   - Saved to `task_proposals.json`

### Stage 3 Workflow

1. **Loads Task Proposal:**
   - Sees which columns were already excluded in Stage 2
   - Can reference this information

2. **During Own Validation:**
   - May find additional columns to exclude
   - Documents them in Stage 3 plan's `excluded_columns`

3. **In Execution Plan JSON:**
   - All excluded columns (Stage 2 + Stage 3) documented
   - Saved to `PLAN-TSK-001.json`

### Downstream Stages (3B, 3.5, 4)

1. **On Initialization:**
   - Load relevant JSON files (task proposal, plan)
   - Extract all `excluded_columns` entries
   - Build context message

2. **Inform Agent:**
   - Agent receives list of excluded columns with reasons
   - Understands why columns are missing
   - Can make informed decisions about alternatives

---

## Example End-to-End Flow

### Stage 2: Task Proposal
```
Agent validates columns:
- Export_Value_INR: 95% complete ✓
- Export_Value_USD: 48% complete ✗
- Year: 100% complete ✓
- Production_Tonnes: 52% complete ✗

Generates task proposal with:
excluded_columns: [
  {"column_name": "Export_Value_USD", "file": "export.csv", "reason": "Only 48% complete"},
  {"column_name": "Production_Tonnes", "file": "production.csv", "reason": "Only 52% complete"}
]
```

### Stage 3: Execution Planning
```
Agent loads task proposal, sees excluded columns.
Validates remaining columns, finds:
- Legacy_Code: 15% complete ✗ (not in task proposal)

Creates plan with:
excluded_columns: [
  {"column_name": "Legacy_Code", "file": "metadata.csv", "reason": "Only 15% complete"}
]
```

### Stage 3B: Data Preparation
```
Receives message:
"**EXCLUDED COLUMNS FROM STAGE 3:**
- Legacy_Code from metadata.csv: Only 15% complete

Do NOT attempt to use these columns."

Agent knows not to look for Legacy_Code when preparing data.
```

### Stage 3.5: Method Testing
```
Receives message:
"**COLUMNS EXCLUDED DUE TO DATA QUALITY:**
- Export_Value_USD from export.csv: Only 48% complete  
- Production_Tonnes from production.csv: Only 52% complete
- Legacy_Code from metadata.csv: Only 15% complete"

Agent understands why these columns are missing and uses alternatives.
```

---

## Benefits

### 1. **Transparency**
- Clear documentation of why columns were excluded
- Traceable decision-making process

### 2. **Context Preservation**
- Downstream stages understand constraints
- No confusion about missing columns

### 3. **Better Decision Making**
- Agents can choose appropriate alternatives
- Understand trade-offs made in earlier stages

### 4. **Debugging**
- Easy to track which columns were rejected and why
- Can verify validation rules are working correctly

### 5. **No Hardcoding**
- Agents dynamically discover and communicate exclusions
- Schema-agnostic approach

---

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `models.py` | +12 lines | Added excluded_columns to TaskProposal and Stage3Plan |
| `stage2_agent.py` | +13 lines | Updated prompt to require excluded_columns documentation |
| `stage3_agent.py` | +28 lines | Updated prompt to require excluded_columns documentation |
| `stage3b_agent.py` | +17 lines | Load and communicate excluded columns to agent |
| `stage3_5_agent.py` | +28 lines | Load exclusions from both Stage 2 and 3, inform agent |

**Total:** ~98 lines added

---

## Testing

```bash
# Test imports
micromamba activate llm
python -c "from agentic_code.models import TaskProposal, Stage3Plan; print('✅ Models updated')"

# Run pipeline and check excluded_columns in output JSONs
python -m agentic_code.master_agent TSK-001

# Check task proposal
cat output/stage2_out/task_proposals.json | jq '.proposals[0].excluded_columns'

# Check execution plan  
cat output/stage3_out/PLAN-TSK-001.json | jq '.excluded_columns'
```

---

## Summary

✅ **Complete column exclusion tracking implemented**

- Stage 2 documents excluded columns in task proposals
- Stage 3 documents additional excluded columns in plans
- Stage 3B receives Stage 3 exclusion context
- Stage 3.5 receives combined Stage 2 + Stage 3 context
- All downstream stages understand why columns are missing
- No hardcoded values - agents dynamically discover and communicate

**The pipeline now has full transparency about data quality decisions! 🎯**
