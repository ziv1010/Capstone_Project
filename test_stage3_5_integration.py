"""
Test script to validate Stage 3.5 Tester agent integration.

This script checks:
1. Stage 3.5 agent can be imported
2. All required tools are available
3. Models are properly defined
4. Integration with master pipeline is correct
"""

import sys
sys.path.insert(0, '/scratch/ziv_baretto/llmserve/final_code')

from agentic_code.stage3_5_agent import (
    stage3_5_node,
    stage3_5_app,
    STAGE3_5_SYSTEM_PROMPT,
)
from agentic_code.tools import STAGE3_5_TOOLS
from agentic_code.models import TesterOutput, ForecastingMethod, BenchmarkResult
from agentic_code.master_agent import build_master_graph, PipelineState


def test_imports():
    """Test that all imports work correctly."""
    print("✓ All imports successful")
    return True


def test_tools_available():
    """Test that all required tools are available."""
    tool_names = [tool.name for tool in STAGE3_5_TOOLS]
    required_tools = [
        'record_thought',
        'record_observation', 
        'load_stage3_plan_for_tester',
        'search',
        'list_data_files',
        'inspect_data_file',
        'python_sandbox_stage3_5',
        'run_benchmark_code',
        'save_tester_output',
    ]
    
    for req_tool in required_tools:
        if req_tool not in tool_names:
            print(f"✗ Missing required tool: {req_tool}")
            return False
    
    print(f"✓ All {len(required_tools)} required tools available")
    return True


def test_models():
    """Test that models can be instantiated."""
    try:
        # Test ForecastingMethod
        method = ForecastingMethod(
            method_id="TEST-1",
            name="Test Method",
            description="A test method",
            implementation_code="print('test')",
            libraries_required=["pandas"]
        )
        
        # Test BenchmarkResult
        result = BenchmarkResult(
            method_id="TEST-1",
            method_name="Test Method",
            metrics={"MAE": 100.0, "RMSE": 150.0},
            train_period="2020-2023",
            validation_period="2024",
            execution_time_seconds=1.5,
            status="success"
        )
        
        # Test TesterOutput
        tester_output = TesterOutput(
            plan_id="PLAN-TSK-001",
            task_category="predictive",
            methods_proposed=[method],
            benchmark_results=[result],
            selected_method_id="TEST-1",
            selected_method=method,
            selection_rationale="Best performing method",
            data_split_strategy="2020-2023 train, 2024 validation"
        )
        
        print("✓ All models can be instantiated correctly")
        return True
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False


def test_master_integration():
    """Test that Stage 3.5 is integrated into master pipeline."""
    try:
        master_app = build_master_graph()
        
        # Check that stage3_5 node is in the graph
        graph_nodes = list(master_app.get_graph().nodes.keys())
        if "stage3_5" not in graph_nodes:
            print("✗ stage3_5 node not found in master graph")
            return False
        
        print("✓ Stage 3.5 integrated into master pipeline")
        print(f"  Pipeline nodes: {graph_nodes}")
        return True
    except Exception as e:
        print(f"✗ Master integration check failed: {e}")
        return False


def test_system_prompt():
    """Test that system prompt contains key instructions."""
    required_keywords = [
        "ReAct",
        "3 iterations",
        "record_thought",
        "record_observation",
        "benchmark",
        "consistency",
        "dataset-agnostic",
    ]
    
    missing = []
    for keyword in required_keywords:
        if keyword not in STAGE3_5_SYSTEM_PROMPT:
            missing.append(keyword)
    
    if missing:
        print(f"✗ System prompt missing keywords: {missing}")
        return False
    
    print("✓ System prompt contains all required instructions")
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Stage 3.5 Tester Agent Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Tools Available", test_tools_available),
        ("Model Instantiation", test_models),
        ("Master Integration", test_master_integration),
        ("System Prompt", test_system_prompt),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 All validation tests passed!")
        print("\nStage 3.5 Tester Agent is ready to use.")
        print("\nKey Features:")
        print("  • ReAct framework with thought/observation recording")
        print("  • 3 methods × 3 iterations benchmarking")
        print("  • Hallucination detection via consistency checks")
        print("  • Dataset-agnostic design")
        print("  • Integrated into master pipeline")
        return 0
    else:
        print("\n⚠️  Some validation tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
