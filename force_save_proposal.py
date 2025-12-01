"""
Force-save mechanism for Stage 3.5a when agent claims completion but doesn't actually save.

This script monitors agent output and triggers save when completion is detected.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def extract_proposal_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract proposal JSON from agent reasoning text.

    Looks for JSON structures in the text and validates them.
    """
    # Pattern 1: Look for explicit JSON blocks in code fences
    json_block_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if 'plan_id' in data and 'methods_proposed' in data:
                return data
        except:
            continue

    # Pattern 2: Look for raw JSON objects (starting with { and containing plan_id)
    json_pattern = r'\{[^{}]*"plan_id"[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            # Try to find the complete JSON by balancing braces
            brace_count = 0
            start_idx = text.index(match)
            for i, char in enumerate(text[start_idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[start_idx:start_idx + i + 1]
                        data = json.loads(json_str)
                        if 'plan_id' in data and 'methods_proposed' in data:
                            return data
                        break
        except:
            continue

    # Pattern 3: Look for structured data in reasoning blocks
    # Extract key-value pairs that match the MethodProposalOutput structure
    if 'plan_id' in text and ('methods_proposed' in text or 'Method 1:' in text or 'Seasonal Moving Average' in text):
        print("⚠️  Found proposal elements in text, but couldn't extract complete JSON")
        print("   Attempting to reconstruct from text...")

        # Try to extract plan_id
        plan_id_match = re.search(r'"?plan_id"?\s*[:=]\s*"?([A-Z]+-[A-Z]+-\d+)"?', text)
        if plan_id_match:
            plan_id = plan_id_match.group(1)
            print(f"   Found plan_id: {plan_id}")

            # Return partial data for manual completion
            return {
                "_partial": True,
                "plan_id": plan_id,
                "note": "Extracted from agent reasoning - needs manual completion"
            }

    return None


def create_minimal_proposal(plan_id: str) -> Dict[str, Any]:
    """Create a minimal valid proposal structure for manual completion."""
    return {
        "plan_id": plan_id,
        "task_category": "predictive",
        "methods_proposed": [
            {
                "method_id": "METHOD-1",
                "name": "Seasonal Moving Average Baseline",
                "description": "3-season moving average baseline method",
                "implementation_code": "# Placeholder - extract from agent logs",
                "libraries_required": ["pandas", "numpy"],
                "metric": "MAE"
            },
            {
                "method_id": "METHOD-2",
                "name": "SARIMA Seasonal Model",
                "description": "Seasonal ARIMA model with seasonal period detection",
                "implementation_code": "# Placeholder - extract from agent logs",
                "libraries_required": ["pandas", "statsmodels"],
                "metric": "RMSE"
            },
            {
                "method_id": "METHOD-3",
                "name": "Random Forest with Seasonal Features",
                "description": "Random Forest using season encoding and engineered features",
                "implementation_code": "# Placeholder - extract from agent logs",
                "libraries_required": ["pandas", "scikit-learn"],
                "metric": "R2"
            }
        ],
        "data_split_strategy": "80/20 split by season order (train: Kharif-Summer, validation: Total)",
        "date_column": "Season",
        "target_column": "Production-2024-25",
        "train_period": "Kharif to Summer seasons",
        "validation_period": "Total season",
        "test_period": None,
        "data_preprocessing_steps": [
            "Load prepared_PLAN-TSK-002.parquet",
            "Identify target column as latest production value",
            "Convert 'Season' to ordered numerical values",
            "Sort data by season order",
            "Split 80/20 for train/validation",
            "For tree model: One-hot encode season and include all area/yield features"
        ]
    }


def force_save_from_logs(plan_id: str, log_file: Optional[Path] = None) -> bool:
    """
    Parse agent logs and force-save the proposal.

    Args:
        plan_id: The plan ID (e.g., PLAN-TSK-002)
        log_file: Optional path to log file. If None, uses default output location.

    Returns:
        True if save succeeded, False otherwise
    """
    from agentic_code.config import STAGE3_5A_OUT_DIR

    # Check if already saved
    existing_files = sorted(STAGE3_5A_OUT_DIR.glob(f"method_proposal_{plan_id}*.json"))
    if existing_files:
        print(f"✅ Proposal already exists: {existing_files[-1].name}")
        return True

    # Try to extract from recent agent output
    print(f"🔍 Searching for proposal data for {plan_id}...")

    # If log file provided, read it
    if log_file and log_file.exists():
        log_text = log_file.read_text()
        proposal_data = extract_proposal_from_text(log_text)
    else:
        # Try to reconstruct from known structure
        print("⚠️  No log file provided, creating minimal template")
        proposal_data = create_minimal_proposal(plan_id)
        proposal_data["_note"] = "Auto-generated template - verify implementation codes"

    if not proposal_data:
        print("❌ Could not extract proposal from logs")
        print("   Creating minimal template for manual completion...")
        proposal_data = create_minimal_proposal(plan_id)
        proposal_data["_note"] = "Template created - MANUAL COMPLETION REQUIRED"

    # Add metadata
    proposal_data["created_at"] = datetime.now().isoformat()
    proposal_data["created_by"] = "force_save_mechanism"

    # Save it
    try:
        STAGE3_5A_OUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}_{timestamp}.json"
        output_path.write_text(json.dumps(proposal_data, indent=2))

        print(f"\n✅ FORCE-SAVED: {output_path.name}")
        print(f"   Plan ID: {proposal_data.get('plan_id')}")
        print(f"   Methods: {len(proposal_data.get('methods_proposed', []))}")
        print(f"   Data split: {proposal_data.get('data_split_strategy', 'N/A')[:80]}")

        if proposal_data.get("_note"):
            print(f"\n⚠️  NOTE: {proposal_data['_note']}")
        if proposal_data.get("_partial"):
            print(f"⚠️  This is a PARTIAL proposal - manual completion needed")

        return True

    except Exception as e:
        print(f"❌ Force-save failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python force_save_proposal.py <plan_id> [log_file]")
        print("Example: python force_save_proposal.py PLAN-TSK-002")
        print("Example: python force_save_proposal.py PLAN-TSK-002 agent_output.log")
        sys.exit(1)

    plan_id = sys.argv[1].strip()
    log_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    success = force_save_from_logs(plan_id, log_file)
    sys.exit(0 if success else 1)
