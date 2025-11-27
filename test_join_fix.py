#!/usr/bin/env python3
"""
Test if the join validation fix works
"""
import sys
from pathlib import Path
import json
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.models import Stage3Plan, JoinStep, FileInstruction, ArtifactSpec
from agentic_code.tools import save_stage3_plan
from agentic_code.config import DATA_DIR

print("=" * 80)
print("Testing Join Validation Fix")
print("=" * 80)

# Mock dataframes
df_production = pd.DataFrame({'Season': ['Kharif', 'Rabi'], 'Production': [100, 200]})
df_exports = pd.DataFrame({'HS_Code': ['Kharif', 'Rabi'], 'Value': [10, 20]})

# Mock file loading
def mock_load_dataframe(filename, **kwargs):
    if "Production" in filename:
        return df_production
    elif "Export" in filename:
        return df_exports
    raise ValueError(f"Unknown file: {filename}")

# Construct a plan that uses left_on and right_on
plan = {
    "plan_id": "PLAN-TSK-TEST",
    "selected_task_id": "TSK-TEST",
    "goal": "Test Join",
    "task_category": "predictive",
    "artifacts": {
        "intermediate_table": "test_data.parquet",
        "intermediate_format": "parquet",
        "expected_columns": ["Season", "HS_Code"],
    },
    "file_instructions": [
        {
            "file_id": "FILE-001",
            "original_name": "All-India-Estimates-of-Area,-Production-&-Yield-of-Food-Grains.csv",
            "alias": "production"
        },
        {
            "file_id": "FILE-002",
            "original_name": "Export-of-Rice-Varieties-to-Bangladesh,-2018-19-to-2024-25.csv",
            "alias": "exports"
        }
    ],
    "join_steps": [
        {
            "step": 1,
            "description": "Load production data",
            "left_table": "production",
            "join_type": "base",
            "join_keys": []
        },
        {
            "step": 2,
            "description": "Join exports data",
            "left_table": "production",
            "right_table": "exports",
            "join_type": "inner",
            "left_on": ["Season"],
            "right_on": ["HS_Code"]
        }
    ],
    "expected_model_types": ["Time Series"],
    "evaluation_metrics": ["MAE"]
}

# Patch load_dataframe and _list_data_files
with patch('agentic_code.tools.load_dataframe', side_effect=mock_load_dataframe):
    with patch('agentic_code.tools._list_data_files', return_value=[
        "All-India-Estimates-of-Area,-Production-&-Yield-of-Food-Grains.csv",
        "Export-of-Rice-Varieties-to-Bangladesh,-2018-19-to-2024-25.csv"
    ]):
        # Call the tool
        try:
            # We use invoke to call the tool properly
            result = save_stage3_plan.invoke(json.dumps(plan))
            print(f"\nTool Output:\n{result}")
            
            if "Plan saved successfully" in result:
                print("\n✅ TEST PASSED: Plan saved with left_on/right_on")
            else:
                print("\n⚠️  TEST FAILED: Unexpected output")
                
        except Exception as e:
            print(f"\n⚠️  TEST FAILED: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
