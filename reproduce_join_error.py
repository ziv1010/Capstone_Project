#!/usr/bin/env python3
"""
Reproduce the Stage 3 validation error due to join keys mismatch
"""
import sys
from pathlib import Path
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.models import Stage3Plan, JoinStep, FileInstruction, ArtifactSpec
from agentic_code.tools import save_stage3_plan
from agentic_code.config import DATA_DIR

print("=" * 80)
print("Reproducing Join Validation Error")
print("=" * 80)

# Construct a plan that tries to join two tables with different keys
# This mimics what the agent likely tried to do for TSK-001
plan = {
    "plan_id": "PLAN-TSK-001",
    "selected_task_id": "TSK-001",
    "goal": "Forecasting Rice Exports",
    "task_category": "predictive",
    "artifacts": {
        "intermediate_table": "TSK-001_data.parquet",
        "intermediate_format": "parquet",
        "expected_columns": ["Season", "HS_Code"],
    },
    "file_instructions": [
        {
            "original_name": "All-India-Estimates-of-Area,-Production-&-Yield-of-Food-Grains.csv",
            "alias": "production"
        },
        {
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
            # This will fail because 'Season' is in production but not exports
            # And 'HS Code' is in exports but not production
            "join_keys": ["Season"] 
        }
    ],
    "expected_model_types": ["Time Series"],
    "evaluation_metrics": ["MAE"]
}

print(f"\nAttempting to save plan with join_keys=['Season']...")
try:
    save_stage3_plan(json.dumps(plan))
    print("✅ Plan saved successfully (Unexpected!)")
except ValueError as e:
    print(f"\n✅ Caught expected error:\n{e}")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
