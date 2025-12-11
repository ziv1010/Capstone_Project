"""
Test script for guardrail system.

Tests that guardrails execute correctly and generate reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from code.master_orchestrator import run_pipeline_stages
from code.config import GUARDRAILS_OUT_DIR, logger
from code.guardrails import GuardrailReport
from code.config import DataPassingManager


def test_stage1_guardrails():
    """Test Stage 1 with guardrails enabled."""
    print("\n" + "="*60)
    print("Testing Stage 1 with Guardrails")
    print("="*60 + "\n")

    try:
        # Run Stage 1 with guardrails
        logger.info("Running Stage 1 with guardrails enabled...")
        state = run_pipeline_stages(
            stages=["stage1"],
            enable_guardrails=True
        )

        # Check if guardrail report was generated
        if hasattr(state, 'guardrail_reports') and 'stage1' in state.guardrail_reports:
            report = state.guardrail_reports['stage1']
            print(f"\n✅ Guardrail Report Generated:")
            print(f"   Stage: {report.stage_name}")
            print(f"   Status: {report.overall_status}")
            print(f"   Checks: {len(report.checks)}")
            print(f"   Execution Time: {report.execution_time_ms:.2f}ms")

            # Show check results
            print(f"\n   Check Results:")
            for check in report.checks:
                icon = "✅" if check.passed else ("❌" if check.severity == "critical" else "⚠️")
                print(f"     {icon} {check.check_name}: {check.message}")

            # Check if file was saved
            guardrail_files = list(GUARDRAILS_OUT_DIR.glob("guardrail_stage1_*.json"))
            if guardrail_files:
                print(f"\n✅ Guardrail report saved to disk: {guardrail_files[0].name}")
            else:
                print(f"\n⚠️  Warning: Guardrail report not found on disk")

            return True
        else:
            print("\n❌ Error: No guardrail report generated for Stage 1")
            return False

    except Exception as e:
        print(f"\n❌ Error testing Stage 1 guardrails: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_guardrail_report_structure():
    """Test that guardrail reports have correct structure."""
    print("\n" + "="*60)
    print("Testing Guardrail Report Structure")
    print("="*60 + "\n")

    guardrail_files = list(GUARDRAILS_OUT_DIR.glob("guardrail_*.json"))

    if not guardrail_files:
        print("⚠️  No guardrail reports found. Run test_stage1_guardrails() first.")
        return False

    try:
        # Load most recent report
        report_path = max(guardrail_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading report: {report_path.name}")

        data = DataPassingManager.load_artifact(report_path)

        # Check structure
        required_fields = ['stage_name', 'overall_status', 'checks', 'execution_time_ms', 'timestamp']
        missing = [f for f in required_fields if f not in data]

        if missing:
            print(f"❌ Missing fields: {missing}")
            return False

        print(f"✅ Report structure valid")
        print(f"   Fields: {list(data.keys())}")

        # Check metadata
        if '_meta' in data:
            print(f"✅ Metadata present: {data['_meta'].keys()}")

        return True

    except Exception as e:
        print(f"❌ Error loading report: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all guardrail tests."""
    print("\n" + "="*70)
    print("GUARDRAIL SYSTEM TEST SUITE")
    print("="*70)

    results = {}

    # Test 1: Stage 1 guardrails
    results['stage1_guardrails'] = test_stage1_guardrails()

    # Test 2: Report structure
    results['report_structure'] = test_guardrail_report_structure()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {'PASSED' if passed else 'FAILED'}")

    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Review errors above")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
