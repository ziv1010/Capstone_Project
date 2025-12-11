"""
Master Orchestrator: Pipeline Coordination

This module coordinates the execution of pipeline stages,
manages state transitions, and handles the conversational interface.
"""

import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR, STAGE3B_OUT_DIR,
    STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR,
    OUTPUT_ROOT, StageTransition, DataPassingManager, logger
)
from code.guardrails import (
    Stage1Guardrail, Stage2Guardrail, Stage3Guardrail,
    Stage3bGuardrail, Stage3_5aGuardrail, Stage3_5bGuardrail,
    Stage4Guardrail, Stage5Guardrail, GuardrailReport
)
from code.models import PipelineState, StageStatus

# Import stage node functions
from code.stage1_agent import stage1_node, run_stage1_quick
from code.stage2_agent import stage2_node, run_stage2, run_stage2_for_query
from code.stage3_agent import stage3_node, run_stage3
from code.stage3b_agent import stage3b_node, run_stage3b
from code.stage3_5a_agent import stage3_5a_node, run_stage3_5a
from code.stage3_5b_agent import stage3_5b_node, run_stage3_5b
from code.stage4_agent import stage4_node, run_stage4
from code.stage5_agent import stage5_node, run_stage5
from code.conversation_agent import ConversationHandler, get_quick_summary


# ============================================================================
# STAGE ORDERING
# ============================================================================

STAGE_ORDER = [
    "stage1",
    "stage2",
    "stage3",
    "stage3b",
    "stage3_5a",
    "stage3_5b",
    "stage4",
    "stage5"
]

STAGE_NODES = {
    "stage1": stage1_node,
    "stage2": stage2_node,
    "stage3": stage3_node,
    "stage3b": stage3b_node,
    "stage3_5a": stage3_5a_node,
    "stage3_5b": stage3_5b_node,
    "stage4": stage4_node,
    "stage5": stage5_node,
}

# Map stage names to guardrail classes (all stages)
STAGE_GUARDRAILS = {
    "stage1": Stage1Guardrail,
    "stage2": Stage2Guardrail,
    "stage3": Stage3Guardrail,
    "stage3b": Stage3bGuardrail,
    "stage3_5a": Stage3_5aGuardrail,
    "stage3_5b": Stage3_5bGuardrail,
    "stage4": Stage4Guardrail,
    "stage5": Stage5Guardrail,
}


# ============================================================================
# PIPELINE BUILDER
# ============================================================================

def build_pipeline_graph(
    start_stage: str = "stage1",
    end_stage: str = "stage5"
) -> StateGraph:
    """
    Build a pipeline graph from start_stage to end_stage.

    Args:
        start_stage: First stage to execute
        end_stage: Last stage to execute

    Returns:
        Compiled StateGraph
    """
    # Get stage indices
    start_idx = STAGE_ORDER.index(start_stage)
    end_idx = STAGE_ORDER.index(end_stage)

    if start_idx > end_idx:
        raise ValueError(f"start_stage ({start_stage}) must come before end_stage ({end_stage})")

    stages_to_run = STAGE_ORDER[start_idx:end_idx + 1]

    # Build graph
    builder = StateGraph(PipelineState)

    # Add nodes
    for stage in stages_to_run:
        builder.add_node(stage, STAGE_NODES[stage])

    # Set entry point
    builder.set_entry_point(stages_to_run[0])

    # Add edges between stages
    for i in range(len(stages_to_run) - 1):
        builder.add_edge(stages_to_run[i], stages_to_run[i + 1])

    # Add final edge to END
    builder.add_edge(stages_to_run[-1], END)

    return builder.compile(checkpointer=MemorySaver())


def build_forecasting_pipeline() -> StateGraph:
    """
    Build the standard forecasting pipeline: 3 → 3B → 3.5A → 3.5B → 4 → 5
    """
    return build_pipeline_graph("stage3", "stage5")


# ============================================================================
# STATE LOADING
# ============================================================================

def load_cached_state(task_id: str = None) -> Tuple[PipelineState, str]:
    """
    Load cached state from disk and determine where to start.

    Returns:
        Tuple of (PipelineState, start_stage)
    """
    state = PipelineState(
        session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        selected_task_id=task_id
    )

    start_stage = "stage1"

    # Check Stage 1
    summaries = list(SUMMARIES_DIR.glob("*.summary.json"))
    if summaries:
        state.mark_stage_completed("stage1", {"count": len(summaries)})
        start_stage = "stage2"

    # Check Stage 2
    proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
    if proposals_path.exists():
        state.mark_stage_completed("stage2", {"path": str(proposals_path)})
        start_stage = "stage3"

    if task_id:
        plan_id = f"PLAN-{task_id}"

        # Check Stage 3
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        if plan_path.exists():
            state.mark_stage_completed("stage3", {"path": str(plan_path)})
            start_stage = "stage3b"

        # Check Stage 3B
        prepared_path = STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet"
        if prepared_path.exists():
            state.mark_stage_completed("stage3b", {"path": str(prepared_path)})
            start_stage = "stage3_5a"

        # Check Stage 3.5A
        method_path = STAGE3_5A_OUT_DIR / f"method_proposal_{plan_id}.json"
        if method_path.exists():
            state.mark_stage_completed("stage3_5a", {"path": str(method_path)})
            start_stage = "stage3_5b"

        # Check Stage 3.5B
        tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
        if tester_path.exists():
            state.mark_stage_completed("stage3_5b", {"path": str(tester_path)})
            start_stage = "stage4"

        # Check Stage 4
        result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
        if result_path.exists():
            state.mark_stage_completed("stage4", {"path": str(result_path)})
            start_stage = "stage5"

        # Check Stage 5
        viz_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json"
        if viz_path.exists():
            state.mark_stage_completed("stage5", {"path": str(viz_path)})
            start_stage = "complete"

    return state, start_stage


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_full_pipeline(task_id: str = None) -> PipelineState:
    """
    Run the full pipeline from Stage 1 to Stage 5.
    """
    logger.info("Running full pipeline")

    # Check cached state
    state, start_stage = load_cached_state(task_id)

    if start_stage == "complete":
        logger.info("Pipeline already complete for this task")
        return state

    # If no task_id and we need to run stage 3+, we need task proposals first
    if not task_id and start_stage in ["stage3", "stage3b", "stage3_5a", "stage3_5b", "stage4", "stage5"]:
        logger.error("Task ID required for stages 3+")
        raise ValueError("Task ID required to run stages 3-5")

    state.selected_task_id = task_id

    # Build and run graph
    graph = build_pipeline_graph(start_stage, "stage5")
    config = {"configurable": {"thread_id": f"pipeline_{task_id or 'full'}"}}

    final_state = graph.invoke(state, config)
    return final_state


def run_pipeline_stages(
    stages: List[str],
    task_id: str = None,
    user_query: str = None,
    resume_from_checkpoint: bool = True,
    enable_guardrails: bool = True
) -> PipelineState:
    """
    Run specific pipeline stages with checkpoint support.

    Args:
        stages: List of stage names to run
        task_id: Task ID (required for stages 3+)
        user_query: Optional user query for context
        resume_from_checkpoint: If True, skip stages that already have outputs

    Returns:
        Final pipeline state
    """
    logger.info(f"Running stages: {stages}")

    # Load cached state to check for existing outputs
    if resume_from_checkpoint and task_id:
        state, next_stage = load_cached_state(task_id)
        state.user_query = user_query

        # Filter out stages that are already complete
        if next_stage != "stage1":
            stages_to_skip = []
            for stage in stages:
                stage_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1
                next_idx = STAGE_ORDER.index(next_stage) if next_stage in STAGE_ORDER else len(STAGE_ORDER)
                if stage_idx < next_idx:
                    stages_to_skip.append(stage)
                    logger.info(f"Skipping {stage} - already has output (resuming from {next_stage})")

            stages = [s for s in stages if s not in stages_to_skip]

            if not stages:
                logger.info("All requested stages already complete!")
                return state
    else:
        # Create fresh state
        state = PipelineState(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            selected_task_id=task_id,
            user_query=user_query
        )

    # Run remaining stages sequentially
    for stage in stages:
        if stage not in STAGE_NODES:
            logger.warning(f"Unknown stage: {stage}")
            continue

        logger.info(f"Running {stage}")

        # Log state if in debug mode
        from code.config import DEBUG
        if DEBUG:
            logger.debug(f"State before {stage}: {state.model_dump_json(indent=2, exclude={'session_id'})}")

        # === NEW: Retry loop with guardrail feedback ===
        max_stage_retries = 2  # Stage can retry twice based on guardrail feedback
        stage_retry_count = 0
        guardrail_feedback = None
        report = None  # Initialize report variable

        while stage_retry_count <= max_stage_retries:
            # Execute stage (with feedback if retry)
            if stage_retry_count > 0 and guardrail_feedback:
                logger.warning(f"🔄 Retrying {stage} (attempt {stage_retry_count + 1}) with guardrail feedback")
                # Store feedback in state for stage to access
                state.guardrail_reports[f"{stage}_feedback"] = guardrail_feedback

            state = STAGE_NODES[stage](state)

            # Check for stage execution failure
            if stage in state.stages and state.stages[stage].status == StageStatus.FAILED:
                logger.error(f"Stage {stage} failed: {state.stages[stage].errors}")
                break  # Exit retry loop if stage itself failed

            # === Run guardrail after stage execution ===
            if not enable_guardrails or stage not in STAGE_GUARDRAILS:
                # No guardrail for this stage, proceed
                break  # Exit retry loop

            # Guardrail enabled for this stage
            logger.info(f"🛡️  Running guardrail for {stage} (stage attempt {stage_retry_count + 1})")

            # RETRY LOGIC for guardrail execution: Try once, retry on crash
            max_guardrail_attempts = 2
            guardrail_executed = False

            for attempt in range(max_guardrail_attempts):
                try:
                    guardrail = STAGE_GUARDRAILS[stage](stage)

                    # Get stage output
                    stage_output = getattr(state, f"{stage}_output", None)

                    # Run validation
                    start_time = time.time()
                    report = guardrail.validate(stage_output, state)
                    execution_time = (time.time() - start_time) * 1000
                    report.execution_time_ms = execution_time

                    # Store guardrail report in state
                    state.guardrail_reports[stage] = report

                    # Log results
                    logger.info(f"Guardrail {stage}: {report.overall_status}")

                    if report.overall_status == "failed":
                        critical_failures = [c for c in report.checks if not c.passed and c.severity == "critical"]
                        logger.error(f"Guardrail FAILED with {len(critical_failures)} critical issues:")
                        for check in critical_failures:
                            logger.error(f"  - {check.check_name}: {check.message}")

                        # Check if retry is needed
                        if report.requires_retry and stage_retry_count < max_stage_retries:
                            logger.warning(f"Guardrail requests retry for {stage}")
                            if report.feedback_for_agent:
                                logger.info(f"Feedback for agent:\n{report.feedback_for_agent}")
                                guardrail_feedback = report.feedback_for_agent
                                stage_retry_count += 1

                                # Save failed attempt report
                                guardrail_dir = OUTPUT_ROOT / "guardrails_out"
                                guardrail_dir.mkdir(parents=True, exist_ok=True)
                                report_filename = f"guardrail_{stage}_{task_id or 'session'}_attempt{stage_retry_count}.json"
                                DataPassingManager.save_artifact(
                                    data=report.model_dump(),
                                    output_dir=guardrail_dir,
                                    filename=report_filename,
                                    metadata={"stage": stage, "plan_id": task_id, "attempt": stage_retry_count}
                                )

                                guardrail_executed = True
                                break  # Break guardrail retry loop to retry stage
                            else:
                                logger.warning("Guardrail failed but no feedback provided, continuing pipeline")
                        else:
                            if stage_retry_count >= max_stage_retries:
                                logger.error(f"Max retries reached for {stage}, continuing despite failures")
                            logger.warning(f"Pipeline continuing despite guardrail failures")

                    elif report.overall_status == "warning":
                        warnings = [c for c in report.checks if not c.passed and c.severity == "warning"]
                        logger.warning(f"Guardrail warnings ({len(warnings)}):")
                        for check in warnings:
                            logger.warning(f"  - {check.check_name}: {check.message}")

                    # Save guardrail report to disk
                    guardrail_dir = OUTPUT_ROOT / "guardrails_out"
                    guardrail_dir.mkdir(parents=True, exist_ok=True)

                    report_filename = f"guardrail_{stage}_{task_id or 'session'}.json"
                    DataPassingManager.save_artifact(
                        data=report.model_dump(),
                        output_dir=guardrail_dir,
                        filename=report_filename,
                        metadata={"stage": stage, "plan_id": task_id}
                    )

                    guardrail_executed = True
                    break  # Success - break guardrail retry loop

                except Exception as e:
                    if attempt < max_guardrail_attempts - 1:
                        logger.warning(f"Guardrail execution failed for {stage} (attempt {attempt + 1}), retrying: {e}")
                        continue
                    else:
                        logger.error(f"Guardrail execution failed for {stage} after {max_guardrail_attempts} attempts: {e}")
                        logger.warning(f"Pipeline continuing without guardrail validation for {stage}")
                        guardrail_executed = True  # Mark as done to avoid infinite loop
                        break

            # If guardrail passed or we're done retrying, exit stage retry loop
            if guardrail_executed and (not report.requires_retry or stage_retry_count >= max_stage_retries):
                break  # Exit stage retry loop

    # === NEW: Generate consolidated guardrail report ===
    if enable_guardrails and state.guardrail_reports:
        consolidated_report = _generate_consolidated_guardrail_report(state, task_id)

        # Save consolidated report
        guardrail_dir = OUTPUT_ROOT / "guardrails_out"
        DataPassingManager.save_artifact(
            data=consolidated_report.model_dump(),
            output_dir=guardrail_dir,
            filename=f"guardrail_report_PLAN-{task_id}.json" if task_id else "guardrail_report_session.json",
            metadata={"type": "consolidated_report", "plan_id": task_id}
        )

        logger.info(f"Consolidated guardrail report: {consolidated_report.overall_status}")

    return state


def _generate_consolidated_guardrail_report(state: PipelineState, task_id: str) -> GuardrailReport:
    """Generate consolidated report from all stage guardrails"""
    stage_reports = state.guardrail_reports

    total_critical = sum(
        len([c for c in r.checks if not c.passed and c.severity == "critical"])
        for r in stage_reports.values()
    )
    total_warnings = sum(
        len([c for c in r.checks if not c.passed and c.severity == "warning"])
        for r in stage_reports.values()
    )

    if total_critical > 0:
        overall_status = "failed"
    elif total_warnings > 0:
        overall_status = "warning"
    else:
        overall_status = "passed"

    # Generate recommendations
    recommendations = []
    for stage_name, report in stage_reports.items():
        for check in report.checks:
            if not check.passed and check.suggestion:
                recommendations.append(f"[{stage_name}] {check.suggestion}")

    return GuardrailReport(
        plan_id=f"PLAN-{task_id}" if task_id else "session",
        overall_status=overall_status,
        stage_reports=stage_reports,
        total_critical_failures=total_critical,
        total_warnings=total_warnings,
        recommendations=recommendations,
        timestamp=datetime.now()
    )


def run_forecasting_pipeline(task_id: str, resume: bool = True) -> PipelineState:
    """
    Run the forecasting pipeline: 3 → 3B → 3.5A → 3.5B → 4 → 5

    Args:
        task_id: Task ID to run the pipeline for
        resume: If True, resume from existing checkpoints (default True)

    Returns:
        Final pipeline state
    """
    return run_pipeline_stages(
        ["stage3", "stage3b", "stage3_5a", "stage3_5b", "stage4", "stage5"],
        task_id=task_id,
        resume_from_checkpoint=resume
    )


# ============================================================================
# CONVERSATIONAL ORCHESTRATOR
# ============================================================================

class ConversationalOrchestrator:
    """
    Orchestrates the conversational pipeline experience.

    Combines the conversation agent with pipeline execution.
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation = ConversationHandler(self.session_id)
        self.current_task_id = None
        self.pipeline_state = None

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and return response with any actions taken.
        """
        # Get conversation response
        result = self.conversation.process_message(user_input)

        # Check if we need to run pipeline stages
        if result.get("action") == "run_pipeline" and result.get("task_id"):
            task_id = result["task_id"]
           
            # If task_id is None, it means validation failed
            if task_id is None:
                result["response"] += "\n\n❌ Task not found. Please check available tasks with 'show tasks' or create a custom task."
                return result
            
            self.current_task_id = task_id

            # Add pipeline execution info to response
            result["pipeline_started"] = True
            result["response"] += f"\n\n🚀 Starting pipeline execution for {task_id}..."
            result["response"] += "\n   Stages: 3 → 3B → 3.5A → 3.5B → 4 → 5"

            # Run the forecasting pipeline
            try:
                self.pipeline_state = run_forecasting_pipeline(task_id)
                result["pipeline_completed"] = True
                result["response"] += "\n\n✅ Pipeline completed successfully!"

                # Add summary of results
                if self.pipeline_state.stage5_output:
                    result["response"] += f"\n\n📊 Visualization report created with insights."
                elif self.pipeline_state.stage4_output:
                    metrics = self.pipeline_state.stage4_output.metrics or {}
                    result["response"] += f"\n\n📈 Execution completed. Metrics: {metrics}"

            except Exception as e:
                result["pipeline_completed"] = False
                result["response"] += f"\n\n❌ Pipeline failed: {e}"
                logger.error(f"Pipeline execution failed: {e}")

        elif result.get("action") == "run_stages" and result.get("stages"):
            stages = result["stages"]
            try:
                self.pipeline_state = run_pipeline_stages(stages)
                result["response"] += f"\n\n✅ Completed stages: {stages}"
            except Exception as e:
                result["response"] += f"\n\n❌ Failed to run stages: {e}"

        return result

    def get_status(self) -> str:
        """Get current pipeline status."""
        from tools.conversation_tools import check_pipeline_status
        return check_pipeline_status.invoke({})

    def get_summary(self) -> str:
        """Get a summary of available data and tasks."""
        return get_quick_summary()


# ============================================================================
# QUICK RUN FUNCTIONS
# ============================================================================

def quick_analyze_data() -> Dict[str, Any]:
    """
    Quickly analyze all available data (Stage 1).
    """
    logger.info("Running quick data analysis")
    return run_stage1_quick()


def quick_propose_tasks(user_query: str = None) -> Dict[str, Any]:
    """
    Quickly propose tasks based on available data.
    """
    logger.info("Running quick task proposal")

    # First ensure Stage 1 is done
    summaries = list(SUMMARIES_DIR.glob("*.summary.json"))
    if not summaries:
        run_stage1_quick()

    # Run Stage 2
    if user_query:
        output = run_stage2_for_query(user_query)
    else:
        output = run_stage2()

    return {
        "proposals": [p.model_dump() for p in output.proposals],
        "notes": output.exploration_notes
    }


def quick_run_task(task_id: str) -> Dict[str, Any]:
    """
    Quickly run all stages for a task.
    """
    logger.info(f"Running quick task execution for {task_id}")

    state = run_forecasting_pipeline(task_id)

    return {
        "completed_stages": [s for s, st in state.stages.items() if st.status == StageStatus.COMPLETED],
        "failed_stages": [s for s, st in state.stages.items() if st.status == StageStatus.FAILED],
        "errors": state.errors,
        "success": len(state.errors) == 0
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the orchestrator.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Conversational AI Pipeline Orchestrator")
    parser.add_argument("--mode", choices=["conversation", "analyze", "propose", "run"],
                        default="conversation", help="Execution mode")
    parser.add_argument("--task", type=str, help="Task ID for run mode")
    parser.add_argument("--query", type=str, help="User query for propose mode")
    parser.add_argument("--stages", type=str, help="Comma-separated stages to run")

    args = parser.parse_args()

    if args.mode == "conversation":
        # Run interactive conversation
        orchestrator = ConversationalOrchestrator()

        print("\n" + "="*60)
        print("Conversational AI Pipeline")
        print("="*60)
        print("\nCommands:")
        print("  Type your question or request")
        print("  'status' - Check pipeline status")
        print("  'summary' - Get data summary")
        print("  'quit' - Exit")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'status':
                    print(f"\n{orchestrator.get_status()}")
                    continue

                if user_input.lower() == 'summary':
                    print(f"\n{orchestrator.get_summary()}")
                    continue

                result = orchestrator.process_user_input(user_input)
                print(f"\nAssistant: {result['response']}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break

    elif args.mode == "analyze":
        result = quick_analyze_data()
        print(json.dumps(result, indent=2, default=str))

    elif args.mode == "propose":
        result = quick_propose_tasks(args.query)
        print(json.dumps(result, indent=2, default=str))

    elif args.mode == "run":
        if not args.task:
            print("Error: --task required for run mode")
            return

        if args.stages:
            stages = [s.strip() for s in args.stages.split(",")]
            state = run_pipeline_stages(stages, args.task)
        else:
            state = run_forecasting_pipeline(args.task)

        print(f"Completed stages: {[s for s, st in state.stages.items() if st.status == StageStatus.COMPLETED]}")
        print(f"Errors: {state.errors}")


if __name__ == "__main__":
    main()
