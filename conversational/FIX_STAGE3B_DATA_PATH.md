# Fix for Stage 3B Data Path Issues

## Problem Summary

Stage 3B (Data Preparation) was failing with `FileNotFoundError` when trying to load CSV files:

```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data/All-India-Estimates-of-Area,-Production-&-Yield-of-Food-Grains.csv'
```

### Root Cause

The system prompt in Stage 3B showed an **incorrect example** that misled the agent to hardcode the path:

```python
# BAD EXAMPLE (from old prompt):
DATA_DIR = Path('/path/to/data')  # Agent copies this literally!
df = pd.read_csv(DATA_DIR / 'file.csv')
```

The agent was literally copying this placeholder path instead of using the pre-defined `DATA_DIR` variable available in the sandbox.

## Solution Implemented

### 1. Fixed System Prompt Example

**File**: `conversational/code/stage3b_agent.py` (lines 116-132)

**Before**:
```python
# Load data
DATA_DIR = Path('/path/to/data')
df = pd.read_csv(DATA_DIR / 'file.csv')
```

**After**:
```python
# Load data - DATA_DIR is already provided in the sandbox
df = pd.read_csv(DATA_DIR / 'file.csv')
```

Added warning:
```
CRITICAL: DATA_DIR is automatically available in the sandbox - DO NOT redefine it!
```

### 2. Enhanced Tool Documentation

**File**: `conversational/tools/stage3b_tools.py` (lines 154-176)

Enhanced the `run_data_prep_code` tool docstring to explicitly state:

```python
"""
Available in sandbox (pre-defined):
- pd, np, json (imports)
- DATA_DIR (Path object pointing to data directory)
- STAGE3_OUT_DIR, STAGE3B_OUT_DIR (Path objects)
- load_dataframe() function

IMPORTANT: DATA_DIR is already defined - use it directly!
Example: df = pd.read_csv(DATA_DIR / 'yourfile.csv')
DO NOT write: DATA_DIR = Path('/path/to/data')
"""
```

## How the Sandbox Works

The `run_data_prep_code` tool injects these variables into the Python execution environment:

```python
additional = {
    'DATA_DIR': DATA_DIR,          # → /scratch/.../conversational/data
    'STAGE3_OUT_DIR': STAGE3_OUT_DIR,
    'STAGE3B_OUT_DIR': STAGE3B_OUT_DIR,
}
```

The agent's code runs with these pre-defined, so it should **never** redefine them.

## Correct Usage Pattern

Agent should generate code like this:

```python
import pandas as pd
import numpy as np

# Load data - DATA_DIR is pre-defined
df = pd.read_csv(DATA_DIR / 'All-India-Estimates-of-Area,-Production-&-Yield-of-Food-Grains.csv')

# Apply filters
df_filtered = df[(df['Crop'] == 'Rice') & (df['Season'] != 'Total')]

# Check nulls
print(f"Shape: {df_filtered.shape}")
print(f"Nulls: {df_filtered.isnull().sum().sum()}")
```

## Testing the Fix

Run Stage 3B again:

```bash
python run_conversational.py
> run TSK-9391
```

The agent should now:
1. ✅ Use DATA_DIR directly (not redefine it)
2. ✅ Successfully load CSV files from the data directory
3. ✅ Apply filters and create features
4. ✅ Save prepared data to parquet

## Additional Recommendations

### 1. Consider Reducing max_tokens (Still at 8192)

The `config.py` still has `max_tokens: 8192` which can cause overflow with large contexts. If token errors persist:

```python
# In config.py line 69:
"max_tokens": 4096,  # Reduced from 8192 to prevent context overflow
```

### 2. Monitor Agent Behavior

Watch for these patterns in logs:
- ✅ Good: `df = pd.read_csv(DATA_DIR / 'file.csv')`
- ❌ Bad: `DATA_DIR = Path('/path/to/data')`
- ❌ Bad: `df = pd.read_csv('file.csv')` (relative path)

### 3. Add Validation

Consider adding a check in `run_data_prep_code` that warns if code tries to redefine DATA_DIR:

```python
if 'DATA_DIR = ' in code or 'DATA_DIR=' in code:
    logger.warning("⚠️  Code attempts to redefine DATA_DIR - this will be overridden!")
```

## Files Modified

1. **conversational/code/stage3b_agent.py** (lines 116-132)
   - Fixed Python code example in system prompt
   - Added critical warning about DATA_DIR

2. **conversational/tools/stage3b_tools.py** (lines 154-169)
   - Enhanced tool docstring with clear instructions
   - Added explicit example of correct usage

## Rollback Instructions

If this causes issues, revert to:

```python
# In stage3b_agent.py:
DATA_DIR = Path('/path/to/data')  # (original example)
```

And update the docstring in stage3b_tools.py to match.

---

**Date**: 2025-12-08
**Fixed By**: Claude Code Assistant
**Priority**: Critical - Blocks all custom task execution
**Status**: Ready for Testing
