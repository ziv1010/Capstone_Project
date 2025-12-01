#!/usr/bin/env python
"""
Quick test script for the enhanced Stage 1 agent with LangGraph.

Tests the analyze -> describe workflow on a small CSV file.
"""

import sys
from pathlib import Path

# Add agentic_code to path
sys.path.insert(0, str(Path(__file__).parent / "agentic_code"))

from agentic_code.stage1_agent import summarize_profile, build_stage1_graph
from agentic_code.utils import profile_csv
from agentic_code.config import DATA_DIR

def test_stage1_graph():
    """Test the LangGraph workflow for Stage 1."""
    print("=" * 80)
    print("Testing Enhanced Stage 1 with LangGraph")
    print("=" * 80)

    # Find a CSV file to test
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("Error: No CSV files found in data directory")
        return

    # Test with the first CSV file
    test_file = csv_files[0]
    print(f"\nTesting with file: {test_file.name}")
    print("-" * 80)

    # Step 1: Profile the CSV
    print("\n1. Profiling CSV file...")
    profile = profile_csv(test_file, sample_rows=100)
    print(f"   ✓ Profiled {profile['n_rows_sampled']} rows, {len(profile['columns'])} columns")

    # Step 2: Run the enhanced summarization with LangGraph
    print("\n2. Running LangGraph workflow (analyze -> describe)...")
    print("   → Node 1: Analyzing dataset context...")
    summary = summarize_profile(profile)
    print("   → Node 2: Generating descriptions with analysis...")
    print(f"   ✓ Generated summary for: {summary.dataset_name}")

    # Step 3: Display sample results
    print("\n3. Sample column descriptions:")
    print("-" * 80)
    for i, col in enumerate(summary.columns[:3]):  # Show first 3 columns
        print(f"\nColumn {i+1}: {col.name}")
        print(f"  Type: {col.logical_type}")
        print(f"  Description: {col.description}")
        if col.is_potential_key:
            print(f"  ✓ Potential key column")

    if len(summary.columns) > 3:
        print(f"\n... and {len(summary.columns) - 3} more columns")

    # Step 4: Show metadata
    print("\n4. Dataset metadata:")
    print("-" * 80)
    print(f"  Candidate primary keys: {summary.candidate_primary_keys}")
    if summary.notes:
        print(f"  Notes: {summary.notes}")

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    test_stage1_graph()
