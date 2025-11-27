#!/usr/bin/env python3
"""
Unit test for hallucination fix using mocked state
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.tools import trigger_pipeline_stages
from agentic_code.models import TaskProposal, TaskCategory

# Mock TaskProposal objects
mock_proposals = [
    TaskProposal(
        id="TSK-MOCK-1",
        category="predictive",
        title="Mock Prediction Task",
        problem_statement="Mock problem",
        required_columns=[],
        target_variable="target",
        feasibility_score=0.9,
        assumptions=[],
        risk_assessment=[]
    ),
    TaskProposal(
        id="TSK-MOCK-2",
        category="descriptive",
        title="Mock Descriptive Task",
        problem_statement="Mock problem",
        required_columns=[],
        target_variable="target",
        feasibility_score=0.8,
        assumptions=[],
        risk_assessment=[]
    )
]

# Mock state
mock_state = {
    "task_proposals": mock_proposals,
    "dataset_summaries": [],
    "stage3_plan": None,
    "execution_result": None,
    "visualization_report": None
}

print("=" * 80)
print("Testing trigger_pipeline_stages Output Format")
print("=" * 80)

# Patch run_partial_pipeline in master_agent
with patch('agentic_code.master_agent.run_partial_pipeline', return_value=mock_state):
    # Call the tool
    output = trigger_pipeline_stages.invoke({
        "start_stage": 2,
        "end_stage": 2,
        "user_query": "test query"
    })
    
    print(f"\nTool Output:\n{output}")
    
    # Verify output contains titles
    if "TSK-MOCK-1" in output and "Mock Prediction Task" in output:
        print("\n✅ TEST PASSED: Tool output contains proposal details")
    else:
        print("\n⚠️  TEST FAILED: Tool output missing proposal details")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
