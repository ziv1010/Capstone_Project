
import sys
import os
from pathlib import Path

# Add the current directory to sys.path to ensure we can import modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from agentic_code.stage3_agent import run_stage3
from agentic_code.config import STAGE3_OUT_DIR

def test_stage3(task_id="TSK-001"):
    print(f"Testing Stage 3 with task_id: {task_id}")
    
    # Run the stage
    try:
        final_state = run_stage3(task_id, debug=True)
        print("\nStage 3 execution completed.")
    except Exception as e:
        print(f"\nERROR during Stage 3 execution: {e}")
        return

    # Check if the plan was saved
    plan_file = STAGE3_OUT_DIR / f"PLAN-{task_id}.json"
    if plan_file.exists():
        print(f"\nSUCCESS: Plan file created at {plan_file}")
        print("Content preview:")
        print(plan_file.read_text()[:500] + "...")
    else:
        print(f"\nFAILURE: Plan file NOT found at {plan_file}")

if __name__ == "__main__":
    task_id = sys.argv[1] if len(sys.argv) > 1 else "TSK-001"
    test_stage3(task_id)
