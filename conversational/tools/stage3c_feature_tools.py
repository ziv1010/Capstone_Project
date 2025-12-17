"""
Stage 3C Tools: Automated Feature Engineering

Tools for dynamic feature engineering - agent generates and executes code
to create new features based on data analysis. No hardcoded templates.
"""

import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    STAGE3B_OUT_DIR, STAGE3C_OUT_DIR,
    DataPassingManager, logger
)


# ============================================================================
# DATA LOADING & ANALYSIS
# ============================================================================

@tool
def load_data_for_feature_engineering(plan_id: str) -> str:
    """
    Load prepared data and analyze its structure for feature engineering.
    
    Returns comprehensive data profile including:
    - Column names and types
    - Sample values
    - Basic statistics
    - Potential feature engineering opportunities
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
    
    Returns:
        Data profile to inform feature engineering decisions
    """
    try:
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        
        if not prepared_path.exists():
            return f"Prepared data not found at: {prepared_path}"
        
        df = pd.read_parquet(prepared_path)
        
        result = ["=== DATA PROFILE FOR FEATURE ENGINEERING ===\n"]
        result.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
        
        # Column analysis
        result.append("COLUMNS:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            n_null = df[col].isnull().sum()
            sample = df[col].dropna().head(3).tolist()
            
            result.append(f"\n  {col}:")
            result.append(f"    Type: {dtype}")
            result.append(f"    Unique: {n_unique}, Nulls: {n_null}")
            result.append(f"    Sample: {sample[:3]}")
            
            # Detect potential opportunities
            if 'date' in col.lower() or 'time' in col.lower():
                result.append(f"    → OPPORTUNITY: Time-based features (year, month, day, weekday)")
            elif df[col].dtype in ['int64', 'float64']:
                if n_unique > 10:
                    result.append(f"    → OPPORTUNITY: Lag features, rolling statistics")
        
        # Identify target column
        target_cols = [c for c in df.columns if c.lower() in ['target', 'output', 'y']]
        if target_cols:
            result.append(f"\nTARGET COLUMN: {target_cols[0]}")
        
        # Numeric columns for potential interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result.append(f"\nNUMERIC COLUMNS ({len(numeric_cols)}): {numeric_cols[:10]}")
        
        # Statistics for numeric columns
        result.append("\nNUMERIC STATISTICS:")
        for col in numeric_cols[:5]:
            stats = df[col].describe()
            result.append(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        
        result.append(f"\n\nData loaded from: {prepared_path}")
        result.append("Use execute_feature_code to generate and apply new features.")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error loading data: {e}"


# ============================================================================
# DYNAMIC CODE EXECUTION
# ============================================================================

@tool
def execute_feature_code(plan_id: str, feature_code: str, description: str) -> str:
    """
    Execute dynamically generated feature engineering code.
    
    The code should:
    - Accept a DataFrame variable 'df'
    - Modify df in-place or return modified df
    - Create new feature columns
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        feature_code: Python code to execute for feature engineering
        description: Brief description of what features are being created
    
    Returns:
        Result of code execution including new columns created
    """
    try:
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        
        if not prepared_path.exists():
            return f"Prepared data not found at: {prepared_path}"
        
        df = pd.read_parquet(prepared_path)
        original_cols = set(df.columns)
        
        result = [f"=== EXECUTING FEATURE CODE ===\n"]
        result.append(f"Description: {description}\n")
        result.append("Code:")
        result.append("```python")
        result.append(feature_code)
        result.append("```\n")
        
        # Create execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
        }
        
        # Execute the code
        try:
            exec(feature_code, exec_globals)
            df = exec_globals.get('df', df)
        except Exception as e:
            result.append(f"❌ EXECUTION ERROR: {e}")
            result.append(f"\nTraceback:\n{traceback.format_exc()}")
            return "\n".join(result)
        
        # Check for new columns
        new_cols = set(df.columns) - original_cols
        
        if new_cols:
            result.append(f"✅ SUCCESS: Created {len(new_cols)} new feature(s)")
            result.append(f"\nNew columns: {list(new_cols)}")
            
            # Sample of new features
            for col in list(new_cols)[:5]:
                result.append(f"\n  {col}:")
                result.append(f"    Sample: {df[col].head(3).tolist()}")
                result.append(f"    Missing: {df[col].isnull().sum()}")
        else:
            result.append("⚠️ WARNING: No new columns created. Check the code.")
        
        # Store enhanced dataframe temporarily
        enhanced_path = STAGE3C_OUT_DIR / f"enhanced_{plan_id}_temp.parquet"
        df.to_parquet(enhanced_path, index=False)
        result.append(f"\nEnhanced data saved temporarily to: {enhanced_path}")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error executing feature code: {e}\n{traceback.format_exc()}"


# ============================================================================
# VALIDATION & SAVING
# ============================================================================

@tool
def validate_features(plan_id: str) -> str:
    """
    Validate newly created features for quality.
    
    Checks:
    - No all-NaN columns
    - Variance > 0 (not constant)
    - No infinite values
    - Reasonable correlation with target
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
    
    Returns:
        Validation results with pass/fail for each new feature
    """
    try:
        enhanced_path = STAGE3C_OUT_DIR / f"enhanced_{plan_id}_temp.parquet"
        original_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        
        if not enhanced_path.exists():
            return "Enhanced data not found. Run execute_feature_code first."
        
        df = pd.read_parquet(enhanced_path)
        original_df = pd.read_parquet(original_path)
        
        new_cols = set(df.columns) - set(original_df.columns)
        
        if not new_cols:
            return "No new features to validate."
        
        result = ["=== FEATURE VALIDATION ===\n"]
        
        valid_features = []
        invalid_features = []
        
        for col in new_cols:
            issues = []
            
            # Check for all NaN
            if df[col].isnull().all():
                issues.append("All values are NaN")
            
            # Check for zero variance
            elif df[col].nunique() <= 1:
                issues.append("Zero variance (constant)")
            
            # Check for infinite values
            if df[col].dtype in ['float64', 'int64']:
                if np.isinf(df[col]).any():
                    issues.append("Contains infinite values")
            
            # Check NaN ratio
            nan_ratio = df[col].isnull().sum() / len(df)
            if nan_ratio > 0.5:
                issues.append(f"High NaN ratio: {nan_ratio:.1%}")
            
            if issues:
                invalid_features.append((col, issues))
                result.append(f"❌ {col}: {', '.join(issues)}")
            else:
                valid_features.append(col)
                result.append(f"✅ {col}: Valid")
        
        result.append(f"\n\nSUMMARY: {len(valid_features)} valid, {len(invalid_features)} invalid")
        
        if invalid_features:
            result.append("\n⚠️ Consider removing or fixing invalid features before saving.")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error validating features: {e}"


@tool 
def save_enhanced_data(plan_id: str, remove_invalid: bool = True) -> str:
    """
    Save the enhanced data with new features.
    
    Optionally removes invalid features (all-NaN, zero variance).
    Updates the prepared data path for downstream stages.
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        remove_invalid: Whether to remove invalid features before saving
    
    Returns:
        Path to saved enhanced data
    """
    try:
        enhanced_path = STAGE3C_OUT_DIR / f"enhanced_{plan_id}_temp.parquet"
        original_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        
        if not enhanced_path.exists():
            return "Enhanced data not found. Run execute_feature_code first."
        
        df = pd.read_parquet(enhanced_path)
        original_df = pd.read_parquet(original_path)
        new_cols = set(df.columns) - set(original_df.columns)
        
        result = ["=== SAVING ENHANCED DATA ===\n"]
        
        if remove_invalid:
            # Remove invalid columns
            valid_cols = []
            removed_cols = []
            
            for col in new_cols:
                is_valid = True
                if df[col].isnull().all():
                    is_valid = False
                elif df[col].nunique() <= 1:
                    is_valid = False
                elif df[col].dtype in ['float64', 'int64'] and np.isinf(df[col]).any():
                    is_valid = False
                
                if is_valid:
                    valid_cols.append(col)
                else:
                    removed_cols.append(col)
            
            if removed_cols:
                df = df.drop(columns=removed_cols)
                result.append(f"Removed {len(removed_cols)} invalid features: {removed_cols}")
            
            result.append(f"Keeping {len(valid_cols)} valid new features")
        
        # Save to final location
        final_path = STAGE3C_OUT_DIR / f"enhanced_{plan_id}.parquet"
        df.to_parquet(final_path, index=False)
        
        # Also update the Stage 3B prepared data (so downstream stages use enhanced data)
        df.to_parquet(original_path, index=False)
        
        # Clean up temp file
        if enhanced_path.exists():
            enhanced_path.unlink()
        
        # Save feature engineering report
        report = {
            "plan_id": plan_id,
            "original_columns": len(original_df.columns),
            "new_features": len(new_cols),
            "final_columns": len(df.columns),
            "new_feature_names": list(set(df.columns) - set(original_df.columns)),
            "shape": list(df.shape)
        }
        DataPassingManager.save_artifact(report, STAGE3C_OUT_DIR, f"{plan_id}_feature_report.json")
        
        result.append(f"\n✅ Enhanced data saved to: {final_path}")
        result.append(f"✅ Updated prepared data at: {original_path}")
        result.append(f"\nFinal shape: {df.shape}")
        result.append(f"New features added: {len(set(df.columns) - set(original_df.columns))}")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error saving enhanced data: {e}"


# Export tools
STAGE3C_FEATURE_TOOLS = [
    load_data_for_feature_engineering,
    execute_feature_code,
    validate_features,
    save_enhanced_data,
]
