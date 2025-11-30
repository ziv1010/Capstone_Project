#!/usr/bin/env python3
"""Manual completion script for Stage 3.5b when the agent gets stuck."""

import json
from pathlib import Path

# Load the checkpoint with benchmark results
checkpoint_path = Path("output/stage3_5b_benchmarking/checkpoint_PLAN-TSK-002.json")
checkpoint = json.loads(checkpoint_path.read_text())

# Load the method proposals to get task_category
proposals_path = Path("output/stage3_5a_method_proposal/method_proposal_PLAN-TSK-002_20251201_035547.json")
proposals = json.loads(proposals_path.read_text())

# Select the best method (METHOD-3 has lowest MAE and RMSE)
best_result = min(checkpoint["completed_results"], key=lambda x: x["metrics"]["MAE"])
best_method_id = best_result["method_id"]
best_method = next(m for m in checkpoint["methods_to_test"] if m["method_id"] == best_method_id)

# Prepare the TesterOutput structure
output_data = {
    "plan_id": checkpoint["plan_id"],
    "task_category": proposals["task_category"],  # "predictive"

    # Methods proposed and benchmark results
    "methods_proposed": checkpoint["methods_to_test"],
    "benchmark_results": checkpoint["completed_results"],

    # Selection
    "selected_method_id": best_method_id,
    "selected_method": best_method,
    "selection_rationale": (
        f"{best_method['name']} was selected as the best performing method with "
        f"MAE={best_result['metrics']['MAE']:.2f} and RMSE={best_result['metrics']['RMSE']:.2f}. "
        f"This significantly outperforms the other methods: "
        f"Moving Average (MAE=57.70, RMSE=141.82) and ARIMA (MAE=1102.29, RMSE=1102.60). "
        f"The Random Forest model effectively captures non-linear patterns in the historical "
        f"production data, making it the most accurate choice for this forecasting task."
    ),

    # Data split strategy
    "data_split_strategy": checkpoint["data_split_strategy"]
}

# Save the tester output
print("=" * 80)
print("Completing Stage 3.5b manually...")
print("=" * 80)
print(f"\nSelected method: {best_method['name']} (ID: {best_method_id})")
print(f"Performance: MAE={best_result['metrics']['MAE']:.2f}, RMSE={best_result['metrics']['RMSE']:.2f}")
print("\nSaving tester output...")

# Validate the data structure
from agentic_code.models import TesterOutput
try:
    tester_output = TesterOutput.model_validate(output_data)
    print("✅ Data structure validated successfully")
except Exception as e:
    print(f"❌ Validation error: {e}")
    raise

# Save to file
from datetime import datetime
output_dir = Path("output/stage3_5b_benchmarking")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"tester_{checkpoint['plan_id']}_{timestamp}.json"
output_path.write_text(json.dumps(output_data, indent=2))

print(f"\n✅ Tester output saved to: {output_path}")
print("=" * 80)
