#!/usr/bin/env python3
"""Force-save tester output from checkpoint when agent gets stuck in loop."""

import json
from pathlib import Path
from datetime import datetime

# Paths
OUTPUT_DIR = Path(__file__).parent / "output"
STAGE3_5A_OUT_DIR = OUTPUT_DIR / "stage3_5a_method_proposal"
STAGE3_5B_OUT_DIR = OUTPUT_DIR / "stage3_5b_benchmarking"  # For both checkpoints and tester outputs (consolidated)

def force_save_tester(plan_id: str):
    """Force-save tester output from checkpoint."""
    
    # Load checkpoint
    checkpoint_path = STAGE3_5B_OUT_DIR / f"checkpoint_{plan_id}.json"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint_data = json.loads(checkpoint_path.read_text())
    print(f"✅ Loaded checkpoint for {plan_id}")
    print(f"   Methods completed: {checkpoint_data.get('methods_completed', [])}")
    
    # Load method proposals
    proposal_files = sorted(STAGE3_5A_OUT_DIR.glob(f"method_proposal_{plan_id}*.json"))
    if not proposal_files:
        print(f"❌ No method proposals found for {plan_id}")
        return
    
    proposal_data = json.loads(proposal_files[-1].read_text())
    methods_proposed = proposal_data.get("methods_proposed", [])
    print(f"✅ Loaded {len(methods_proposed)} method proposals")
    
    # Select best method from completed_results
    completed_results = checkpoint_data.get("completed_results", [])
    if not completed_results:
        print("❌ No completed results in checkpoint")
        return
    
    # Find best by lowest MAE
    best_result = min(completed_results, key=lambda r: r.get("metrics", {}).get("MAE", float('inf')))
    selected_method_id = best_result.get("method_id")
    
    print(f"\n🎯 Best method: {selected_method_id} with MAE {best_result.get('metrics', {}).get('MAE')}")
    
    # Find the method object
    selected_method = next((m for m in methods_proposed if m.get("method_id") == selected_method_id), None)
    if not selected_method:
        selected_method = methods_proposed[0] if methods_proposed else {}
        selected_method_id = selected_method.get("method_id", "METHOD-1")
    
    # Build tester output
    tester_output = {
        "plan_id": plan_id,
        "task_category": "predictive",
        "methods_proposed": methods_proposed,
        "benchmark_results": checkpoint_data.get("benchmark_results", completed_results),
        "selected_method_id": selected_method_id,
        "selected_method": selected_method,
        "selection_rationale": f"FORCE-SAVED manually. {selected_method.get('name', 'Method')} selected with lowest MAE: {best_result.get('metrics', {}).get('MAE', 'N/A')}",
        "data_split_strategy": checkpoint_data.get("data_split_strategy", ""),
        "detailed_procedure": "See selected method implementation code",
        "data_preprocessing_steps": checkpoint_data.get("data_preprocessing_steps", []) or [],
        "method_comparison_summary": f"Tested {len(methods_proposed)} methods. Best: {selected_method.get('name', 'N/A')}"
    }
    
    # Save to stage3_5b_benchmarking (consolidated directory with checkpoints)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STAGE3_5B_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}_{timestamp}.json"
    output_path.write_text(json.dumps(tester_output, indent=2))
    
    print(f"\n✅ FORCE-SAVED: {output_path.name}")
    print(f"   Selected: {selected_method.get('name', 'N/A')} ({selected_method_id})")
    print(f"   Methods tested: {len(methods_proposed)}")
    print(f"   Benchmark results: {len(tester_output['benchmark_results'])}")
    print(f"\n📍 File location: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python force_save_tester.py <plan_id>")
        print("Example: python force_save_tester.py PLAN-TSK-003")
        sys.exit(1)
    
    plan_id = sys.argv[1]
    force_save_tester(plan_id)
