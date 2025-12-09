"""
Stage 6 Tools: Final Report Generation

Tools for generating comprehensive final reports based on task proposals,
execution results, and visualizations.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.tools import tool
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    STAGE2_OUT_DIR, STAGE3_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE6_OUT_DIR,
    DataPassingManager, logger
)


@tool
def load_task_proposal(plan_id: str) -> str:
    """
    Load the original task proposal to understand what was requested.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Task proposal details as formatted string
    """
    try:
        # Extract task ID from plan ID
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id

        # Load Stage 2 task proposals
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if not proposals_path.exists():
            return f"Task proposals not found at: {proposals_path}"

        proposals_data = DataPassingManager.load_artifact(proposals_path)
        proposals = proposals_data.get('proposals', [])

        for proposal in proposals:
            if proposal.get('id') == task_id:
                result = ["=== ORIGINAL TASK PROPOSAL ===\n"]
                result.append(f"ID: {proposal.get('id')}")
                result.append(f"Title: {proposal.get('title')}")
                result.append(f"Category: {proposal.get('category')}")
                result.append(f"Problem Statement: {proposal.get('problem_statement')}")
                result.append(f"Target Column: {proposal.get('target_column')}")
                result.append(f"Datasets: {', '.join(proposal.get('datasets_involved', []))}")
                result.append(f"Feasibility Score: {proposal.get('feasibility_score', 'N/A')}")
                return "\n".join(result)

        return f"Task {task_id} not found in proposals"

    except Exception as e:
        return f"Error loading task proposal: {e}"


@tool
def load_execution_plan(plan_id: str) -> str:
    """
    Load the execution plan from Stage 3.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Execution plan details as formatted string
    """
    try:
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        if not plan_path.exists():
            return f"Execution plan not found at: {plan_path}"

        plan = DataPassingManager.load_artifact(plan_path)

        result = ["=== EXECUTION PLAN ===\n"]
        result.append(f"Plan ID: {plan.get('plan_id')}")
        result.append(f"Goal: {plan.get('goal')}")
        result.append(f"Task Category: {plan.get('task_category')}")
        result.append(f"Target Column: {plan.get('target_column')}")
        result.append(f"Expected Model Types: {', '.join(plan.get('expected_model_types', []))}")
        result.append(f"Evaluation Metrics: {', '.join(plan.get('evaluation_metrics', []))}")
        result.append(f"Forecast Horizon: {plan.get('forecast_horizon')}")
        result.append(f"Forecast Type: {plan.get('forecast_type')}")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading execution plan: {e}"


@tool
def load_execution_results(plan_id: str) -> str:
    """
    Load the execution results from Stage 4.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Execution results as formatted string
    """
    try:
        result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
        if not result_path.exists():
            return f"Execution results not found at: {result_path}"

        exec_result = DataPassingManager.load_artifact(result_path)

        result = ["=== EXECUTION RESULTS ===\n"]
        result.append(f"Status: {exec_result.get('status')}")
        result.append(f"Summary: {exec_result.get('summary')}")

        metrics = exec_result.get('metrics', {})
        if metrics:
            result.append("\nPerformance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    result.append(f"  - {key.upper()}: {value:.4f}")
                else:
                    result.append(f"  - {key.upper()}: {value}")

        if 'artifacts' in exec_result:
            result.append(f"\nArtifacts: {exec_result['artifacts']}")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading execution results: {e}"


@tool
def load_prediction_data(plan_id: str) -> str:
    """
    Load and summarize the prediction data from Stage 4.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Prediction data summary as formatted string
    """
    try:
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        if not predictions_path.exists():
            return f"Prediction data not found at: {predictions_path}"

        df = pd.read_parquet(predictions_path)

        result = ["=== PREDICTION DATA SUMMARY ===\n"]
        result.append(f"Total Predictions: {len(df)}")
        result.append(f"Columns: {list(df.columns)}")

        # Find prediction and actual columns
        pred_cols = [c for c in df.columns if 'predict' in c.lower() or 'forecast' in c.lower()]
        actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'target' in c.lower()]

        if pred_cols and actual_cols:
            pred_col, actual_col = pred_cols[0], actual_cols[0]
            result.append(f"\nPrediction Column: {pred_col}")
            result.append(f"Actual Column: {actual_col}")

            # Statistics on predictions
            result.append(f"\nPrediction Statistics:")
            result.append(f"  Min: {df[pred_col].min():.4f}")
            result.append(f"  Max: {df[pred_col].max():.4f}")
            result.append(f"  Mean: {df[pred_col].mean():.4f}")
            result.append(f"  Std: {df[pred_col].std():.4f}")

            # Check for forecast data
            if 'prediction_type' in df.columns:
                forecast_count = len(df[df['prediction_type'] == 'forecast'])
                test_count = len(df[df['prediction_type'] == 'test'])
                result.append(f"\nForecast Predictions: {forecast_count}")
                result.append(f"Test Predictions: {test_count}")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading prediction data: {e}"


@tool
def load_visualization_report(plan_id: str) -> str:
    """
    Load the visualization report from Stage 5.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Visualization report details as formatted string
    """
    try:
        viz_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json"
        if not viz_path.exists():
            return f"Visualization report not found at: {viz_path}"

        viz_report = DataPassingManager.load_artifact(viz_path)

        result = ["=== VISUALIZATION REPORT ===\n"]
        result.append(f"Summary: {viz_report.get('summary')}")

        visualizations = viz_report.get('visualizations', [])
        result.append(f"\nVisualizations Created: {len(visualizations)}")
        for i, viz in enumerate(visualizations, 1):
            result.append(f"\n{i}. {viz.get('title', 'Untitled')}")
            result.append(f"   Type: {viz.get('plot_type')}")
            result.append(f"   File: {viz.get('filename')}")
            result.append(f"   Description: {viz.get('description', 'N/A')}")

        insights = viz_report.get('insights', [])
        if insights:
            result.append("\nKey Insights:")
            for insight in insights:
                result.append(f"  - {insight}")

        # Check for task answer
        task_answer = viz_report.get('task_answer', {})
        if task_answer:
            result.append("\nTask Answer Available: Yes")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading visualization report: {e}"


@tool
def load_task_answer(plan_id: str) -> str:
    """
    Load the task answer file from Stage 5 if it exists.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)

    Returns:
        Task answer content
    """
    try:
        # Extract task ID
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id

        # Try to find task answer file
        answer_path = STAGE5_OUT_DIR / f"task_answer_{plan_id}.txt"
        if answer_path.exists():
            with open(answer_path, 'r') as f:
                return f.read()

        return "Task answer file not found. This will be generated in the final report."

    except Exception as e:
        return f"Error loading task answer: {e}"


@tool
def generate_final_report(
    plan_id: str,
    executive_summary: str,
    methodology: str,
    results_analysis: str,
    conclusions: str,
    recommendations: str
) -> str:
    """
    Generate and save the final comprehensive report.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        executive_summary: Brief overview of the task and key findings
        methodology: Description of the approach and methods used
        results_analysis: Detailed analysis of results based on actual data
        conclusions: Final conclusions answering the original task
        recommendations: Actionable recommendations

    Returns:
        Path to saved report
    """
    try:
        # Extract task ID
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id

        # Build the report
        report_sections = []

        report_sections.append("═" * 80)
        report_sections.append(f"FINAL ANALYSIS REPORT: {task_id}")
        report_sections.append("═" * 80)
        report_sections.append("")

        report_sections.append("EXECUTIVE SUMMARY")
        report_sections.append("-" * 80)
        report_sections.append(executive_summary)
        report_sections.append("")

        report_sections.append("METHODOLOGY")
        report_sections.append("-" * 80)
        report_sections.append(methodology)
        report_sections.append("")

        report_sections.append("RESULTS ANALYSIS")
        report_sections.append("-" * 80)
        report_sections.append(results_analysis)
        report_sections.append("")

        report_sections.append("CONCLUSIONS")
        report_sections.append("-" * 80)
        report_sections.append(conclusions)
        report_sections.append("")

        report_sections.append("RECOMMENDATIONS")
        report_sections.append("-" * 80)
        report_sections.append(recommendations)
        report_sections.append("")

        report_sections.append("═" * 80)
        report_sections.append(f"Report generated for {task_id}")
        report_sections.append("═" * 80)

        report_content = "\n".join(report_sections)

        # Save as text file
        txt_path = STAGE6_OUT_DIR / f"{task_id}_final_report.txt"
        with open(txt_path, 'w') as f:
            f.write(report_content)

        # Also save as JSON for programmatic access
        report_data = {
            "task_id": task_id,
            "plan_id": plan_id,
            "executive_summary": executive_summary,
            "methodology": methodology,
            "results_analysis": results_analysis,
            "conclusions": conclusions,
            "recommendations": recommendations,
            "generated_at": str(pd.Timestamp.now())
        }

        json_path = DataPassingManager.save_artifact(
            data=report_data,
            output_dir=STAGE6_OUT_DIR,
            filename=f"{task_id}_final_report.json",
            metadata={"stage": "stage6", "type": "final_report"}
        )

        logger.info(f"Final report saved to {txt_path} and {json_path}")

        return f"Final report saved successfully:\n  Text: {txt_path}\n  JSON: {json_path}"

    except Exception as e:
        return f"Error generating final report: {e}"


# Export tools list
STAGE6_TOOLS = [
    load_task_proposal,
    load_execution_plan,
    load_execution_results,
    load_prediction_data,
    load_visualization_report,
    load_task_answer,
    generate_final_report,
]
