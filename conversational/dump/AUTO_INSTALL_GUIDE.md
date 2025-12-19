# Auto-Install Package Support for Stage 3.5B Benchmarking

## Overview

The stage3_5b benchmarking tools now support **automatic package installation**. When a method requires a package that isn't installed (like xgboost, lightgbm, catboost, etc.), the system will automatically install it and make it available for benchmarking.

## What Was Fixed

### Problem
Previously, when methods proposed in stage3_5a included packages like `xgboost`, the benchmarking stage would fail with:
- `NameError: name 'xgboost' is not defined`
- `ImportError: No module named 'xgboost'`
- Error: "Missing xgboost dependency"

The `required_libraries` field in method proposals was informational only and not used for actual dependency resolution.

### Solution
Added three new capabilities to `conversational/tools/stage3_5b_tools.py`:

1. **`install_package(package_name, timeout=120)`** - Installs packages via pip
2. **`import_with_auto_install(module_name, package_name, timeout=120)`** - Imports with auto-install fallback
3. **`setup_ml_namespace(required_libraries=None)`** - Creates execution namespace with auto-installed dependencies

## Supported Packages

### Core Packages (Always Available)
- **sklearn** → scikit-learn
- **statsmodels**
- pandas, numpy, json (pre-installed)

### Extended Packages (Auto-Install on Demand)
- **xgboost** - Gradient boosting framework
- **lightgbm** - Light Gradient Boosting Machine
- **catboost** - Categorical Boosting
- **prophet** - Time series forecasting
- **tensorflow** - Deep learning framework
- **keras** - High-level neural networks API
- **torch/pytorch** - PyTorch deep learning
- **cv2** → opencv-python - Computer vision
- **PIL** → pillow - Image processing
- **skimage** → scikit-image - Image processing

### Package Name Mapping
The system automatically maps import names to pip package names:
```python
PACKAGE_MAPPING = {
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    # ... and more
}
```

## How to Use

### In Method Proposals (Stage 3.5A)

When proposing methods, simply include the required libraries:

```json
{
  "method_id": "M3",
  "name": "XGBoost Regressor",
  "category": "gradient_boosting",
  "required_libraries": ["xgboost", "sklearn"],
  "implementation_code": "..."
}
```

### In Benchmarking Code (Stage 3.5B)

#### Option 1: Using `run_benchmark_code` tool
```python
# The agent will call:
run_benchmark_code(
    code="<your benchmark code>",
    method_name="XGBoost Regressor",
    required_libraries="xgboost,sklearn"  # comma-separated
)
```

#### Option 2: Using `test_single_method` tool
```python
# The required_libraries are automatically read from the method proposal
test_single_method(
    plan_id="PLAN-TSK-6243",
    method_id="M3"
)
```

### In Custom Python Code

If you need to use the auto-install functionality directly:

```python
from conversational.tools.stage3_5b_tools import (
    import_with_auto_install,
    setup_ml_namespace
)

# Import a single package
xgboost = import_with_auto_install('xgboost')
if xgboost:
    from xgboost import XGBRegressor
    model = XGBRegressor()

# Or setup full namespace
namespace = setup_ml_namespace(
    required_libraries=['xgboost', 'lightgbm', 'catboost']
)
# namespace now contains XGBRegressor, LGBMRegressor, CatBoostRegressor, etc.
```

## Example: Fixing the XGBoost Error

### Before (Failed)
```json
{
  "method_id": "M3",
  "error": "Missing xgboost dependency",
  "valid": false
}
```

### After (Success)
The system will:
1. Detect that `xgboost` is in `required_libraries`
2. Try to import `xgboost`
3. If ImportError, automatically run: `pip install xgboost`
4. Retry import
5. Add `XGBRegressor` to the execution namespace
6. Execute the method code successfully

```json
{
  "method_id": "M3",
  "avg_mae": 45123.45,
  "avg_rmse": 78234.12,
  "avg_mape": 95432.21,
  "valid": true
}
```

## Configuration

### Installation Timeout
Default: 120 seconds per package

To change:
```python
import_with_auto_install('large_package', timeout=300)  # 5 minutes
```

### Adding New Package Mappings

Edit `PACKAGE_MAPPING` in [stage3_5b_tools.py](conversational/tools/stage3_5b_tools.py:31-46):

```python
PACKAGE_MAPPING = {
    # ... existing mappings
    'your_import_name': 'pip-package-name',
}
```

## Testing

Run the test script to verify functionality:

```bash
python3 test_auto_install_simple.py
```

Expected output:
```
Test 4: Import xgboost (the problematic one)
  ℹ xgboost not found, attempting auto-install...
  Installing package: xgboost...
  ✓ Successfully installed xgboost
  ✓ Successfully imported xgboost after installation
  ✓ XGBoost successfully loaded!
  Version: 3.1.2
  ✓ XGBRegressor available: <class 'xgboost.sklearn.XGBRegressor'>
```

## Error Handling

### If Installation Fails
The system will:
1. Log a warning
2. Continue with other packages
3. Return `None` for failed imports
4. The method may still execute if the package isn't critical

### If Import Still Fails After Installation
- Check pip installation permissions
- Check internet connectivity
- Check package name spelling
- Review logs in the execution output

### Logs
Installation attempts are logged:
- `logger.info()` for successful installs
- `logger.warning()` for failed installs
- `logger.error()` for exceptions

## Changes Made to Codebase

### Modified Files
1. **[conversational/tools/stage3_5b_tools.py](conversational/tools/stage3_5b_tools.py)**
   - Added: `PACKAGE_MAPPING` dictionary (lines 31-46)
   - Added: `install_package()` function (lines 49-80)
   - Added: `import_with_auto_install()` function (lines 83-118)
   - Added: `setup_ml_namespace()` function (lines 121-206)
   - Updated: `run_benchmark_code()` - added `required_libraries` parameter (line 396)
   - Updated: `test_single_method()` - uses auto-install namespace (lines 982-994)

### New Files
1. **test_auto_install_simple.py** - Simple test script
2. **AUTO_INSTALL_GUIDE.md** - This documentation

## Benefits

1. **No More Missing Dependency Errors** - Packages are installed automatically
2. **Expanded Method Support** - Can now test gradient boosting, deep learning, and more
3. **Better Stage 3.5A → 3.5B Integration** - `required_libraries` now actively used
4. **Smoother Workflow** - No manual intervention needed for package installation
5. **Extensible** - Easy to add support for new packages

## Limitations

1. **Installation Time** - First use of a package adds installation time (~10-60 seconds)
2. **Internet Required** - Must have internet access to download packages
3. **Permissions** - Requires pip installation permissions
4. **Storage** - Each package requires disk space

## Future Enhancements

Possible improvements:
1. Pre-install common packages in a separate setup step
2. Cache installed packages across runs
3. Support for conda packages in addition to pip
4. Parallel package installation
5. Custom package repositories

## Support

For issues or questions:
1. Check logs in the benchmark execution output
2. Review the test script output
3. Check the method proposal's `required_libraries` field
4. Verify internet connectivity and pip permissions

---

**Last Updated**: 2025-12-10
**Version**: 1.0
**Author**: Auto-install enhancement for stage3_5b
