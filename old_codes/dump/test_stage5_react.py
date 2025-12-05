#!/usr/bin/env python3
"""
Test script for the new ReAct-based Stage 5 visualization agent.

This script demonstrates how the agent now:
1. Analyzes columns to understand given vs predicted data
2. Plans visualizations with explicit reasoning
3. Creates plots with detailed explanations
4. Saves comprehensive reports
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.stage5_agent import run_stage5

def main():
    """Run Stage 5 with ReAct framework on existing results."""

    # Check if plan_id is provided
    if len(sys.argv) < 2:
        print("Usage: python test_stage5_react.py <plan_id>")
        print("\nExample: python test_stage5_react.py PLAN-TSK-002")
        print("\nThis will run the enhanced Stage 5 visualization agent with:")
        print("  - Column analysis to identify given vs predicted data")
        print("  - Explicit planning for each visualization")
        print("  - Detailed explanations for each plot")
        print("  - Clear distinction between input and output data")
        sys.exit(1)

    plan_id = sys.argv[1]

    print("=" * 80)
    print("TESTING ENHANCED STAGE 5 WITH REACT FRAMEWORK")
    print("=" * 80)
    print(f"\nPlan ID: {plan_id}")
    print("\nThe agent will:")
    print("  1. Analyze the data columns (given vs predicted)")
    print("  2. Plan each visualization with reasoning")
    print("  3. Create plots with detailed explanations")
    print("  4. Save a comprehensive report")
    print("\n" + "=" * 80 + "\n")

    # Run Stage 5
    result = run_stage5(plan_id, max_rounds=30, debug=True)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nCheck the output directory for:")
    print("  - PNG plot files")
    print("  - plot_*_explanation.txt files")
    print("  - visualization_report_*.json")
    print("\nLocation: /scratch/ziv_baretto/llmserve/final_code/output/stage5_out/")

if __name__ == "__main__":
    main()
