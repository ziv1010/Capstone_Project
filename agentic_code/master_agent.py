"""
Master Agent: Unified Agentic AI Pipeline

Orchestrates all 5 stages using a single LangGraph workflow:
- Stage 1: Dataset Summarization
- Stage 2: Task Proposal Generation
- Stage 3: Execution Planning
- Stage 4: Execution
- Stage 5: Visualization

Each stage is a state node that processes data and updates the shared pipeline state.
"""

from __future__ import annotations

from typing import TypedDict, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

from .models import (
    DatasetSummary, TaskProposal, Stage3Plan,
    ExecutionResult, VisualizationReport
)
from .stage1_agent import stage1_node
from .stage2_agent import stage2_node
from .stage3_agent import stage3_node
from .stage4_agent import stage4_node
from .stage5_agent import stage5_node


# ===========================
# Unified Pipeline State
# ===========================

class PipelineState(TypedDict):
    """Unified state for the entire agentic pipeline."""
    # Current progress
    current_stage: int  # Current stage (1-5)
    completed_stages: List[int]  # List of completed stages
    
    # Stage outputs
    dataset_summaries: List[DatasetSummary]  # From Stage 1
    task_proposals: List[TaskProposal]  # From Stage 2
    selected_task_id: Optional[str]  # Which task to execute
    stage3_plan: Optional[Stage3Plan]  # From Stage 3
    execution_result: Optional[ExecutionResult]  # From Stage 4
    visualization_report: Optional[VisualizationReport]  # From Stage 5
    
    # Tracking
    errors: List[str]  # Track any errors
    started_at: str  # When the pipeline started


# ===========================
# Build Master Graph
# ===========================

def build_master_graph():
    """Build the master pipeline graph with all 5 stages.
    
    Returns:
        Compiled LangGraph application
    """
    builder = StateGraph(PipelineState)

    # Add all stage nodes
    builder.add_node("stage1", stage1_node)
    builder.add_node("stage2", stage2_node)  
    builder.add_node("stage3", stage3_node)
    builder.add_node("stage4", stage4_node)
    builder.add_node("stage5", stage5_node)

    # Set up linear progression through stages
    builder.set_entry_point("stage1")
    builder.add_edge("stage1", "stage2")
    builder.add_edge("stage2", "stage3")  
    builder.add_edge("stage3", "stage4")
    builder.add_edge("stage4", "stage5")
    builder.add_edge("stage5", END)

    # Compile
    master_app = builder.compile()
    
    return master_app


# ===========================
# Pipeline Runner
# ===========================

def run_full_pipeline(selected_task_id: Optional[str] = None) -> PipelineState:
    """Run the complete 5-stage pipeline.
    
    Args:
        selected_task_id: Task ID to execute (e.g., 'TSK-001').
                         If None, will use the first task from Stage 2.
        
    Returns:
        Final pipeline state with all results
    """
    print("\n" + "=" * 80)
    print("🚀 UNIFIED AGENTIC AI PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Initialize state
    initial_state: PipelineState = {
        "current_stage": 1,
        "completed_stages": [],
        "dataset_summaries": [],
        "task_proposals": [],
        "selected_task_id": selected_task_id,
        "stage3_plan": None,
        "execution_result": None,
        "visualization_report": None,
        "errors": [],
        "started_at": datetime.now().isoformat(),
    }
    
    # Build and run the graph
    master_app = build_master_graph()
    final_state = master_app.invoke(initial_state)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Completed stages: {final_state['completed_stages']}")
    print(f"Dataset summaries: {len(final_state.get('dataset_summaries', []))}")
    print(f"Task proposals: {len(final_state.get('task_proposals', []))}")
    if final_state.get('selected_task_id'):
        print(f"Executed task: {final_state['selected_task_id']}")
    if final_state.get('execution_result'):
        print(f"Execution status: {final_state['execution_result'].status}")
    if final_state.get('visualization_report'):
        print(f"Visualizations: {len(final_state['visualization_report'].visualizations)}")
    if final_state.get('errors'):
        print(f"\n⚠️  Errors encountered:")
        for error in final_state['errors']:
            print(f"  - {error}")
    print("=" * 80)
    
    return final_state


def run_partial_pipeline(
    start_stage: int = 1,
    end_stage: int = 5,
    selected_task_id: Optional[str] = None
) -> PipelineState:
    """Run a subset of the pipeline stages.
    
    Args:
        start_stage: Stage to start from (1-5)
        end_stage: Stage to end at (1-5)
        selected_task_id: Task ID to execute (required for stages 3+)
        
    Returns:
        Final pipeline state
        
    Raises:
        ValueError: If stage range is invalid
    """
    if not (1 <= start_stage <= end_stage <= 5):
        raise ValueError(f"Invalid stage range: {start_stage}-{end_stage}")
    
    print(f"\n🎯 Running pipeline stages {start_stage}-{end_stage}")
    
    # For now, we need to run sequentially from stage 1
    # (could be enhanced to support starting mid-pipeline by loading previous results)
    if start_stage > 1:
        print("⚠️  Note: Currently must start from Stage 1. Running full pipeline up to stage_stage.")
        return run_up_to_stage(end_stage, selected_task_id)
    
    return run_up_to_stage(end_stage, selected_task_id)


def run_up_to_stage(end_stage: int, selected_task_id: Optional[str] = None) -> PipelineState:
    """Run pipeline up to specified stage.
    
    Args:
        end_stage: Stage to end at (1-5)
        selected_task_id: Task ID to execute (required for stages 3+)
        
    Returns:
        Pipeline state after reaching end_stage
    """
    # Build a custom graph with only the stages we need
    builder = StateGraph(PipelineState)
    
    # Add nodes up to end_stage
    if end_stage >= 1:
        builder.add_node("stage1", stage1_node)
        builder.set_entry_point("stage1")
    
    if end_stage >= 2:
        builder.add_node("stage2", stage2_node)
        builder.add_edge("stage1", "stage2")
    
    if end_stage >= 3:
        builder.add_node("stage3", stage3_node)
        builder.add_edge("stage2", "stage3")
    
    if end_stage >= 4:
        builder.add_node("stage4", stage4_node)
        builder.add_edge("stage3", "stage4")
    
    if end_stage >= 5:
        builder.add_node("stage5", stage5_node)
        builder.add_edge("stage4", "stage5")
    
    # Connect last stage to END
    last_stage = f"stage{end_stage}"
    builder.add_edge(last_stage, END)
    
    partial_app = builder.compile()
    
    # Initialize and run
    initial_state: PipelineState = {
        "current_stage": 1,
        "completed_stages": [],
        "dataset_summaries": [],
        "task_proposals": [],
        "selected_task_id": selected_task_id,
        "stage3_plan": None,
        "execution_result": None,
        "visualization_report": None,
        "errors": [],
        "started_at": datetime.now().isoformat(),
    }
    
    return partial_app.invoke(initial_state)


# ===========================
# Export Master App
# ===========================

# Pre-built master application
master_app = build_master_graph()


if __name__ == "__main__":
    # Run the full pipeline
    import sys
    
    # Check if task ID provided
    task_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    if task_id:
        print(f"Running pipeline with task ID: {task_id}")
    else:
        print("Running pipeline (will auto-select first task from Stage 2)")
    
    run_full_pipeline(selected_task_id=task_id)
