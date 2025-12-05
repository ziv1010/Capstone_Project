# Force-Save Mechanism for Stage 3.5a

## Problem
The Stage 3.5a agent sometimes enters a loop where it:
1. **Claims** it has completed and saved the method proposal
2. **Never actually calls** the `save_method_proposal_output()` tool
3. **Loops indefinitely** saying "successfully completed" without actually saving

## Solution: Two-Level Force-Save System

### 1. Automatic Detection (Built into agent code)
**Location:** `agentic_code/stage3_5a_agent.py` (lines 622-703)

**How it works:**
- After agent execution, checks if proposal file exists
- If missing, scans last 10 messages for "completion claims"
- If 3+ claims detected, triggers force-save extraction
- Attempts to extract proposal from tool call arguments
- Validates and saves the proposal

**Trigger conditions:**
```python
# Detects phrases like:
- "successfully completed"
- "successfully finalized"
- "ready for implementation"
- "proposal completed"
- "methods are ready"
```

### 2. Manual Force-Save Script
**Location:** `extract_and_save_tsk002.py`

**Usage:**
```bash
micromamba run -n llm python extract_and_save_tsk002.py
```

**What it does:**
- Reconstructs the complete proposal from agent's Round 7 reasoning
- Creates valid MethodProposalOutput with all 3 methods:
  1. Seasonal Moving Average Baseline (MAE)
  2. SARIMA Seasonal Model (RMSE)
  3. Random Forest with Seasonal Features (R²)
- Includes complete implementation code for each method
- Validates structure against Pydantic model
- Saves to proper output directory

### 3. Generic Force-Save Tool
**Location:** `force_save_proposal.py`

**Usage:**
```bash
# From terminal/logs
python force_save_proposal.py PLAN-TSK-002 [optional_log_file.txt]

# Creates minimal template if extraction fails
python force_save_proposal.py PLAN-TSK-XXX
```

**Features:**
- Attempts to parse proposal JSON from agent logs
- Falls back to creating minimal template for manual completion
- Can extract from various formats (JSON blocks, reasoning text, tool calls)

## Success Metrics

### PLAN-TSK-002 Status: ✅ SAVED
```
File: output/stage3_5a_method_proposal/method_proposal_PLAN-TSK-002_20251201_222111.json

Methods:
  1. Seasonal Moving Average Baseline (MAE)
  2. SARIMA Seasonal Model (RMSE)
  3. Random Forest with Seasonal Features (R²)

Data split: Train on Kharif/Rabi/Summer, validate on Total season
Target: Production-2024-25
Date column: Season

✅ Ready for Stage 3.5b
```

## How to Use for Future Tasks

### If agent loops without saving:

**Option 1: Let automatic detection handle it**
- The agent code now detects loops automatically
- Will attempt extraction after execution completes

**Option 2: Kill agent and use manual script**
```bash
# Ctrl+C to stop agent
micromamba run -n llm python extract_and_save_tsk002.py
# Or create task-specific extraction script
```

**Option 3: Use generic force-save**
```bash
# If you saved terminal output to a log file
python force_save_proposal.py PLAN-TSK-XXX agent_output.log

# Or let it create a template
python force_save_proposal.py PLAN-TSK-XXX
```

## Creating Task-Specific Extraction Scripts

Copy `extract_and_save_tsk002.py` as a template:

```python
def create_tskXXX_proposal():
    """Reconstruct from agent's reasoning."""
    proposal = {
        "plan_id": "PLAN-TSK-XXX",
        "task_category": "predictive",
        "methods_proposed": [
            # Extract from agent's Round 7 or wherever it proposed methods
            {...},
            {...},
            {...}
        ],
        "data_split_strategy": "...",
        "target_column": "...",
        # ... fill from agent logs
    }
    return proposal
```

## Prevention Tips

To reduce looping in future:
1. **Clearer system prompt:** Already updated with explicit save requirements
2. **Earlier tool checking:** Agent now validates earlier if save was called
3. **Timeout/round limits:** Already set to 26 rounds max
4. **Better tool validation:** Check if tool actually executed vs just claimed

## Files Modified

### Enhanced:
- `agentic_code/stage3_5a_agent.py` - Added loop detection and force-save extraction

### Created:
- `extract_and_save_tsk002.py` - Task-specific extraction for TSK-002
- `force_save_proposal.py` - Generic force-save utility
- `FORCE_SAVE_README.md` - This documentation

## Next Steps

1. ✅ PLAN-TSK-002 proposal saved successfully
2. → Run Stage 3.5b to test the 3 proposed methods
3. → Monitor for similar looping in other stages
4. → Consider applying similar patterns to other agents if needed

---

**Created:** 2025-12-01
**Author:** Claude Code (force-save implementation)
**Status:** Production-ready
