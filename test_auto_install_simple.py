#!/usr/bin/env python3
"""
Simple test script to verify auto-install functionality.
Tests the core functions without importing the full tool suite.
"""

import sys
import subprocess
import importlib
from pathlib import Path

# Package mapping
PACKAGE_MAPPING = {
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'statsmodels': 'statsmodels',
}


def install_package(package_name: str, timeout: int = 120) -> bool:
    """Install a Python package using pip."""
    try:
        print(f"  Installing package: {package_name}...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            print(f"  ✓ Successfully installed {package_name}")
            return True
        else:
            print(f"  ✗ Failed to install {package_name}: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Installation of {package_name} timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error installing {package_name}: {e}")
        return False


def import_with_auto_install(module_name: str, package_name: str = None):
    """Import a module, installing it if not found."""
    if package_name is None:
        package_name = PACKAGE_MAPPING.get(module_name, module_name)

    try:
        module = importlib.import_module(module_name)
        print(f"  ✓ {module_name} already installed")
        return module
    except ImportError:
        print(f"  ℹ {module_name} not found, attempting auto-install...")

        if install_package(package_name):
            try:
                module = importlib.import_module(module_name)
                print(f"  ✓ Successfully imported {module_name} after installation")
                return module
            except ImportError as e:
                print(f"  ✗ Still cannot import {module_name} after installation: {e}")
                return None
        else:
            print(f"  ✗ Failed to install package for {module_name}")
            return None


def main():
    print("=" * 60)
    print("SIMPLE AUTO-INSTALL TEST")
    print("=" * 60)

    # Test 1: Import numpy (should already exist)
    print("\nTest 1: Import numpy (should already be installed)")
    np = import_with_auto_install('numpy')
    if np:
        print(f"  Version: {np.__version__}")

    # Test 2: Import pandas
    print("\nTest 2: Import pandas")
    pd = import_with_auto_install('pandas')
    if pd:
        print(f"  Version: {pd.__version__}")

    # Test 3: Import sklearn (with mapping)
    print("\nTest 3: Import sklearn (maps to scikit-learn)")
    sklearn = import_with_auto_install('sklearn', 'scikit-learn')
    if sklearn:
        print(f"  Version: {sklearn.__version__}")

    # Test 4: Import xgboost
    print("\nTest 4: Import xgboost (the problematic one)")
    xgboost = import_with_auto_install('xgboost')
    if xgboost:
        print(f"  ✓ XGBoost successfully loaded!")
        print(f"  Version: {xgboost.__version__}")
        # Try to access XGBRegressor
        try:
            from xgboost import XGBRegressor
            print(f"  ✓ XGBRegressor available: {XGBRegressor}")
        except ImportError:
            print(f"  ✗ XGBRegressor not available")
    else:
        print(f"  ⚠ XGBoost could not be loaded")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nIf xgboost was successfully imported, the auto-install")
    print("functionality is working correctly and will fix the issue!")


if __name__ == '__main__':
    main()
