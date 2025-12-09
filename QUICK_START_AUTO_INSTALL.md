# Quick Start: Auto-Install Package Support

## TL;DR - What Changed?

**Before**: Methods using xgboost/lightgbm/catboost failed with "Missing dependency" error

**After**: Packages are automatically installed when needed ✓

## For Stage 3.5A (Method Proposal)

Just list the packages you need:

```json
{
  "methods_proposed": [
    {
      "method_id": "M3",
      "name": "XGBoost Regressor",
      "required_libraries": ["xgboost", "sklearn"],
      "implementation_code": "..."
    }
  ]
}
```

That's it! The system handles the rest.

## For Stage 3.5B (Benchmarking)

### If you're the agent:
Use the tools as normal - they now auto-install:

```python
# This now works even if xgboost isn't installed
run_benchmark_code(
    code=method_code,
    method_name="XGBoost",
    required_libraries="xgboost"
)
```

### If you're writing custom code:
```python
from conversational.tools.stage3_5b_tools import setup_ml_namespace

# Get namespace with auto-installed packages
namespace = setup_ml_namespace(required_libraries=['xgboost', 'lightgbm'])

# Use it
exec(your_code, namespace)
```

## Supported Packages (Auto-Install)

✓ xgboost
✓ lightgbm
✓ catboost
✓ prophet
✓ tensorflow
✓ keras
✓ torch/pytorch
✓ And more... (see [AUTO_INSTALL_GUIDE.md](AUTO_INSTALL_GUIDE.md))

## Test It

```bash
python3 test_auto_install_simple.py
```

Should show:
```
✓ XGBoost successfully loaded!
```

## What If It Fails?

1. Check internet connection
2. Check pip permissions
3. Check the execution output for error details
4. See [AUTO_INSTALL_GUIDE.md](AUTO_INSTALL_GUIDE.md) for troubleshooting

## Files Changed

- [conversational/tools/stage3_5b_tools.py](conversational/tools/stage3_5b_tools.py) - Added auto-install support
- `run_benchmark_code()` - Now accepts `required_libraries` parameter
- `test_single_method()` - Automatically reads `required_libraries` from proposals

## Example Fix

Your error:
```json
{
  "method_id": "M3",
  "error": "Missing xgboost dependency",
  "valid": false
}
```

Now becomes:
```json
{
  "method_id": "M3",
  "avg_mae": 45123.45,
  "valid": true
}
```

**That's it! No more missing dependency errors.**
