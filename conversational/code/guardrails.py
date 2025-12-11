"""
Guardrails Module: Multi-Layered Validation System

This module implements defense-in-depth guardrails for the conversational AI pipeline.
Each stage has dedicated validation to ensure data quality, safety, and accuracy.

Architecture:
- Stage 1 → Guardrail 1 → Stage 2 → Guardrail 2 → ... → Stage 5 → Guardrail 5
- Moderate validation strictness (>70% nulls = critical, 35-70% = warning)
- Continue on failures (don't halt pipeline)
- Retry guardrails once on crash

Author: AI Pipeline Team
"""

import math
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR,
    STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR,
    STAGE4_OUT_DIR, STAGE5_OUT_DIR, logger
)
from code.models import (
    Stage1Output, Stage2Output, Stage3Plan, ExecutionResult,
    VisualizationReport, PipelineState
)


# ============================================================================
# BASE MODELS
# ============================================================================

class GuardrailCheckResult(BaseModel):
    """Single guardrail check result"""
    check_name: str = Field(description="Name of the check")
    check_type: str = Field(description="Type: data_quality, safety, business_logic, accuracy")
    passed: bool = Field(description="Whether the check passed")
    severity: str = Field(description="Severity: critical, warning, info")
    message: str = Field(description="Human-readable message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    suggestion: Optional[str] = Field(default=None, description="Suggestion for fixing the issue")
    timestamp: datetime = Field(default_factory=datetime.now)


class StageGuardrailReport(BaseModel):
    """Guardrail report for a single stage"""
    stage_name: str = Field(description="Stage name (stage1, stage2, etc.)")
    overall_status: str = Field(description="Overall status: passed, warning, failed")
    checks: List[GuardrailCheckResult] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now)

    # NEW: Actionable feedback for agents
    requires_retry: bool = Field(default=False, description="Whether stage should retry")
    feedback_for_agent: Optional[str] = Field(default=None, description="Detailed feedback for agent to fix issues")
    failed_checks_summary: List[str] = Field(default_factory=list, description="Summary of failed checks")


class GuardrailReport(BaseModel):
    """Consolidated guardrail report for entire pipeline"""
    plan_id: str = Field(description="Plan ID or session ID")
    overall_status: str = Field(description="Overall status: passed, warning, failed")
    stage_reports: Dict[str, StageGuardrailReport] = Field(default_factory=dict)
    total_critical_failures: int = Field(default=0)
    total_warnings: int = Field(default=0)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# BASE GUARDRAIL CLASS
# ============================================================================

class BaseGuardrail(ABC):
    """Abstract base class for all guardrails"""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.checks: List[GuardrailCheckResult] = []

    @abstractmethod
    def validate(self, stage_output: Any, pipeline_state: PipelineState) -> StageGuardrailReport:
        """
        Validate stage output and return report.

        Args:
            stage_output: Output from the stage
            pipeline_state: Current pipeline state

        Returns:
            StageGuardrailReport with validation results
        """
        pass

    def add_check(
        self,
        check_name: str,
        check_type: str,
        passed: bool,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        """Helper to add a check result"""
        self.checks.append(
            GuardrailCheckResult(
                check_name=check_name,
                check_type=check_type,
                passed=passed,
                severity=severity,
                message=message,
                details=details,
                suggestion=suggestion,
                timestamp=datetime.now()
            )
        )

    def generate_report(self) -> StageGuardrailReport:
        """Generate consolidated report from all checks"""
        critical_failures = [c for c in self.checks if not c.passed and c.severity == "critical"]
        warnings = [c for c in self.checks if not c.passed and c.severity == "warning"]

        if critical_failures:
            overall_status = "failed"
        elif warnings:
            overall_status = "warning"
        else:
            overall_status = "passed"

        # Generate actionable feedback for agent
        requires_retry = len(critical_failures) > 0
        feedback_parts = []
        failed_summaries = []

        for check in critical_failures:
            failed_summaries.append(f"{check.check_name}: {check.message}")
            if check.suggestion:
                feedback_parts.append(f"- {check.check_name}: {check.suggestion}")

        feedback_for_agent = None
        if feedback_parts:
            feedback_for_agent = (
                f"GUARDRAIL FEEDBACK - Critical issues detected in {self.stage_name}:\n" +
                "\n".join(feedback_parts) +
                "\n\nPlease address these issues and regenerate the output."
            )

        return StageGuardrailReport(
            stage_name=self.stage_name,
            overall_status=overall_status,
            checks=self.checks,
            execution_time_ms=0.0,  # Will be set by orchestrator
            timestamp=datetime.now(),
            requires_retry=requires_retry,
            feedback_for_agent=feedback_for_agent,
            failed_checks_summary=failed_summaries
        )


# ============================================================================
# STAGE 1 GUARDRAIL: Data Summarization Validation
# ============================================================================

class Stage1Guardrail(BaseGuardrail):
    """
    Validates Stage 1 (Data Summarization) outputs.

    Checks:
    - All datasets have summaries generated
    - Summaries contain required fields
    - Data quality within acceptable ranges (moderate strictness)
    - No corrupted or empty summaries
    """

    def validate(self, stage_output: Stage1Output, pipeline_state: PipelineState) -> StageGuardrailReport:
        logger.info(f"[Guardrail] Validating {self.stage_name}")

        # Check 1: Summaries exist
        if not stage_output or not stage_output.summaries:
            self.add_check(
                "summaries_exist", "data_quality", False, "critical",
                "No dataset summaries generated",
                suggestion="Verify DATA_DIR contains valid CSV files and Stage 1 executed successfully"
            )
            return self.generate_report()

        self.add_check(
            "summaries_exist", "data_quality", True, "info",
            f"Generated {len(stage_output.summaries)} dataset summaries"
        )

        # Check 2: Schema completeness for each summary
        required_fields = ['filename', 'shape', 'columns', 'null_fractions', 'dtypes']
        for summary in stage_output.summaries:
            missing = [f for f in required_fields if not hasattr(summary, f) or getattr(summary, f) is None]
            if missing:
                self.add_check(
                    f"schema_{summary.filename}", "data_quality", False, "critical",
                    f"Summary missing required fields: {missing}",
                    details={"filename": summary.filename, "missing_fields": missing}
                )
            else:
                self.add_check(
                    f"schema_{summary.filename}", "data_quality", True, "info",
                    f"Summary schema complete for {summary.filename}"
                )

        # Check 3: Data quality thresholds (MODERATE strictness)
        for summary in stage_output.summaries:
            if not hasattr(summary, 'null_fractions') or not summary.null_fractions:
                continue

            # CRITICAL: >70% nulls (severe)
            critical_null_cols = [col for col, frac in summary.null_fractions.items() if frac > 0.70]
            if critical_null_cols:
                self.add_check(
                    f"quality_critical_{summary.filename}", "data_quality", False, "critical",
                    f"Severe null columns (>70%) in {summary.filename}: {critical_null_cols[:5]}",
                    details={"filename": summary.filename, "columns": critical_null_cols},
                    suggestion="Consider removing these columns or applying imputation before analysis"
                )

            # WARNING: 35-70% nulls (moderate)
            warning_null_cols = [
                col for col, frac in summary.null_fractions.items()
                if 0.35 < frac <= 0.70
            ]
            if warning_null_cols:
                self.add_check(
                    f"quality_warning_{summary.filename}", "data_quality", False, "warning",
                    f"Moderate null columns (35-70%) in {summary.filename}: {warning_null_cols[:5]}",
                    details={"filename": summary.filename, "columns": warning_null_cols}
                )

        # Check 4: Empty datasets
        for summary in stage_output.summaries:
            if hasattr(summary, 'shape') and summary.shape:
                rows, cols = summary.shape
                if rows == 0:
                    self.add_check(
                        f"empty_dataset_{summary.filename}", "data_quality", False, "critical",
                        f"Dataset {summary.filename} has 0 rows",
                        suggestion="Remove empty datasets or investigate why no data was loaded"
                    )
                elif cols == 0:
                    self.add_check(
                        f"no_columns_{summary.filename}", "data_quality", False, "critical",
                        f"Dataset {summary.filename} has 0 columns",
                        suggestion="Check data loading process"
                    )

        return self.generate_report()


# ============================================================================
# STAGE 2 GUARDRAIL: Task Proposal Validation
# ============================================================================

class Stage2Guardrail(BaseGuardrail):
    """
    Validates Stage 2 (Task Proposal) outputs.

    Checks:
    - Proposals are feasible with available data
    - Required datasets exist
    - Target columns exist
    - Join plans are valid
    - Task categories are valid
    """

    def validate(self, stage_output: Stage2Output, pipeline_state: PipelineState) -> StageGuardrailReport:
        logger.info(f"[Guardrail] Validating {self.stage_name}")

        # Check 1: Proposals exist
        if not stage_output or not stage_output.proposals:
            self.add_check(
                "proposals_exist", "business_logic", False, "critical",
                "No task proposals generated",
                suggestion="Verify Stage 1 summaries exist and Stage 2 executed successfully"
            )
            return self.generate_report()

        self.add_check(
            "proposals_exist", "business_logic", True, "info",
            f"Generated {len(stage_output.proposals)} task proposals"
        )

        # Check 2: Minimum proposal count
        if len(stage_output.proposals) < 3:
            self.add_check(
                "proposal_count", "business_logic", False, "warning",
                f"Only {len(stage_output.proposals)} proposals generated (expected at least 3)",
                suggestion="Review dataset quality and availability"
            )

        # Get available datasets
        available_datasets = set()
        if SUMMARIES_DIR.exists():
            available_datasets = {f.stem.replace('.summary', '') for f in SUMMARIES_DIR.glob("*.summary.json")}

        # Check 3: Validate each proposal
        valid_categories = ['forecasting', 'regression', 'classification', 'clustering']

        for i, proposal in enumerate(stage_output.proposals):
            proposal_id = getattr(proposal, 'id', f'proposal_{i}')

            # Check dataset availability
            if hasattr(proposal, 'dataset') and proposal.dataset:
                if proposal.dataset not in available_datasets:
                    self.add_check(
                        f"dataset_{proposal_id}", "data_quality", False, "critical",
                        f"Proposal {proposal_id} references non-existent dataset: {proposal.dataset}",
                        suggestion=f"Use one of: {list(available_datasets)[:5]}"
                    )

            # Check category validity
            if hasattr(proposal, 'task_category') and proposal.task_category:
                if proposal.task_category not in valid_categories:
                    self.add_check(
                        f"category_{proposal_id}", "business_logic", False, "warning",
                        f"Proposal {proposal_id} has invalid category: {proposal.task_category}",
                        details={"valid_categories": valid_categories}
                    )

            # Check join plan validity (if multi-dataset)
            if hasattr(proposal, 'join_plan') and proposal.join_plan:
                if not hasattr(proposal.join_plan, 'primary_dataset'):
                    self.add_check(
                        f"join_plan_{proposal_id}", "business_logic", False, "warning",
                        f"Proposal {proposal_id} has incomplete join plan",
                        suggestion="Ensure join_plan specifies primary_dataset and join keys"
                    )

        return self.generate_report()


# ============================================================================
# STAGE 3 GUARDRAIL: Execution Plan Validation
# ============================================================================

class Stage3Guardrail(BaseGuardrail):
    """
    Validates Stage 3 (Execution Planning) outputs.

    Checks:
    - Plan has valid method selections
    - Data split strategy is appropriate
    - File instructions reference existing files
    - Plan is executable
    """

    def validate(self, stage_output: Stage3Plan, pipeline_state: PipelineState) -> StageGuardrailReport:
        logger.info(f"[Guardrail] Validating {self.stage_name}")

        # Check 1: Plan exists
        if not stage_output:
            self.add_check(
                "plan_exists", "business_logic", False, "critical",
                "No execution plan generated",
                suggestion="Verify Stage 3 executed successfully"
            )
            return self.generate_report()

        self.add_check(
            "plan_exists", "business_logic", True, "info",
            f"Execution plan generated: {getattr(stage_output, 'plan_id', 'unknown')}"
        )

        # Check 2: Method count (should have 3 methods)
        if hasattr(stage_output, 'methods') and stage_output.methods:
            method_count = len(stage_output.methods)
            if method_count != 3:
                self.add_check(
                    "method_count", "business_logic", False, "warning",
                    f"Plan has {method_count} methods (expected 3)",
                    suggestion="Stage 3.5A should propose exactly 3 methods for benchmarking"
                )
            else:
                self.add_check(
                    "method_count", "business_logic", True, "info",
                    "Plan has correct number of methods (3)"
                )

        # Check 3: Data split strategy
        if hasattr(stage_output, 'data_split_strategy'):
            split = stage_output.data_split_strategy
            if hasattr(split, 'train_fraction') and hasattr(split, 'test_fraction'):
                total = split.train_fraction + split.test_fraction
                if hasattr(split, 'val_fraction'):
                    total += split.val_fraction

                if not (0.99 <= total <= 1.01):  # Allow small floating point error
                    self.add_check(
                        "data_split", "business_logic", False, "critical",
                        f"Data split fractions don't sum to 1.0 (sum={total:.3f})",
                        suggestion="Ensure train + val + test fractions = 1.0"
                    )
                else:
                    self.add_check(
                        "data_split", "business_logic", True, "info",
                        f"Data split valid (train={split.train_fraction}, test={split.test_fraction})"
                    )

        # Check 4: File instructions validity
        if hasattr(stage_output, 'file_instructions') and stage_output.file_instructions:
            for file_inst in stage_output.file_instructions:
                if hasattr(file_inst, 'filename'):
                    file_path = DATA_DIR / file_inst.filename
                    if not file_path.exists():
                        self.add_check(
                            f"file_{file_inst.filename}", "data_quality", False, "critical",
                            f"Referenced file does not exist: {file_inst.filename}",
                            suggestion=f"Verify file exists in {DATA_DIR}"
                        )

        # Check 5: Target column validity
        if hasattr(stage_output, 'target_column') and stage_output.target_column:
            self.add_check(
                "target_column", "business_logic", True, "info",
                f"Target column specified: {stage_output.target_column}"
            )
        else:
            self.add_check(
                "target_column", "business_logic", False, "warning",
                "No target column specified in plan",
                suggestion="Ensure task has clear prediction target"
            )

        return self.generate_report()


# ============================================================================
# STAGE 4 GUARDRAIL: Execution Results Validation
# ============================================================================

class Stage4Guardrail(BaseGuardrail):
    """
    Validates Stage 4 (Execution) outputs.

    Checks:
    - Metrics are valid (no NaN, Inf)
    - Predictions are reasonable
    - Results file exists
    - No hallucinations in results
    """

    def validate(self, stage_output: ExecutionResult, pipeline_state: PipelineState) -> StageGuardrailReport:
        logger.info(f"[Guardrail] Validating {self.stage_name}")

        # Check 1: Execution result exists
        if not stage_output:
            self.add_check(
                "result_exists", "accuracy", False, "critical",
                "No execution result generated",
                suggestion="Verify Stage 4 executed successfully"
            )
            return self.generate_report()

        # Check 2: Execution status
        if hasattr(stage_output, 'status'):
            from code.models import ExecutionStatus
            if stage_output.status == ExecutionStatus.SUCCESS:
                self.add_check(
                    "execution_status", "accuracy", True, "info",
                    "Execution completed successfully"
                )
            elif stage_output.status == ExecutionStatus.FAILURE:
                self.add_check(
                    "execution_status", "accuracy", False, "critical",
                    "Execution failed",
                    details={"errors": getattr(stage_output, 'errors', [])},
                    suggestion="Check Stage 4 logs for detailed error messages"
                )

        # Check 3: Metrics validity
        if hasattr(stage_output, 'metrics') and stage_output.metrics:
            for metric_name, value in stage_output.metrics.items():
                if value is None:
                    self.add_check(
                        f"metric_{metric_name}", "accuracy", False, "warning",
                        f"Metric {metric_name} is None"
                    )
                elif not isinstance(value, (int, float)):
                    self.add_check(
                        f"metric_{metric_name}", "accuracy", False, "warning",
                        f"Metric {metric_name} has invalid type: {type(value)}"
                    )
                elif not math.isfinite(value):
                    self.add_check(
                        f"metric_{metric_name}", "accuracy", False, "critical",
                        f"Metric {metric_name} is not finite: {value}",
                        suggestion="Check for division by zero or invalid calculations in model training"
                    )
                else:
                    self.add_check(
                        f"metric_{metric_name}", "accuracy", True, "info",
                        f"Metric {metric_name}={value:.4f} is valid"
                    )
        else:
            self.add_check(
                "metrics_exist", "accuracy", False, "critical",
                "No metrics generated in execution result",
                suggestion="Ensure model evaluation completed successfully"
            )

        # Check 4: Results file existence and hallucination detection
        task_id = pipeline_state.selected_task_id
        if task_id:
            plan_id = f"PLAN-{task_id}" if not task_id.startswith("PLAN-") else task_id
            results_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"

            if results_path.exists():
                self.add_check(
                    "results_file", "accuracy", True, "info",
                    f"Results file exists: {results_path.name}"
                )

                # Check 5: Prediction sanity and hallucination detection
                try:
                    df = pd.read_parquet(results_path)
                    if 'predicted' in df.columns:
                        predictions = df['predicted'].dropna()

                        if len(predictions) == 0:
                            self.add_check(
                                "predictions", "accuracy", False, "critical",
                                "No predictions found in results file - possible hallucination",
                                suggestion="Ensure model actually generated predictions. Check if results are fabricated."
                            )
                        else:
                            mean, std = predictions.mean(), predictions.std()

                            # HALLUCINATION CHECK: Compare with actual values if available
                            if 'actual' in df.columns:
                                actual = df['actual'].dropna()
                                if len(actual) > 0:
                                    actual_mean = actual.mean()
                                    actual_std = actual.std()

                                    # Check if predictions are wildly different from actuals
                                    if abs(mean - actual_mean) > 10 * actual_std:
                                        self.add_check(
                                            "hallucination_detection", "accuracy", False, "critical",
                                            f"Predictions appear hallucinated - mean prediction ({mean:.2f}) is {abs(mean - actual_mean)/actual_std:.1f} standard deviations from actual data mean ({actual_mean:.2f})",
                                            suggestion="Verify model is using actual data, not generating random values. Retrain model with correct data."
                                        )

                                    # Check if prediction variance is unrealistic
                                    if std > 0 and actual_std > 0:
                                        variance_ratio = std / actual_std
                                        if variance_ratio > 5 or variance_ratio < 0.1:
                                            self.add_check(
                                                "variance_hallucination", "accuracy", False, "critical",
                                                f"Prediction variance ({std:.2f}) is {variance_ratio:.2f}x actual variance ({actual_std:.2f}) - suggests hallucinated data",
                                                suggestion="Check if model is generating synthetic data instead of real predictions"
                                            )

                            # Check for extreme outliers (beyond 5 sigma)
                            if std > 0:
                                outliers = predictions[
                                    (predictions < mean - 5*std) | (predictions > mean + 5*std)
                                ]
                                if len(outliers) > len(predictions) * 0.05:  # More than 5% outliers
                                    self.add_check(
                                        "prediction_outliers", "accuracy", False, "warning",
                                        f"Found {len(outliers)} extreme outliers ({len(outliers)/len(predictions)*100:.1f}%)",
                                        details={"mean": mean, "std": std, "outlier_count": len(outliers)},
                                        suggestion="Review data preprocessing and model training for anomalies"
                                    )
                                else:
                                    self.add_check(
                                        "prediction_sanity", "accuracy", True, "info",
                                        f"Predictions within reasonable range (mean={mean:.2f}, std={std:.2f})"
                                    )

                            # HALLUCINATION CHECK: Check for constant or near-constant predictions
                            if std < 0.01 * abs(mean):
                                self.add_check(
                                    "constant_predictions", "accuracy", False, "critical",
                                    f"Predictions are nearly constant (std={std:.4f}) - likely hallucinated",
                                    suggestion="Model appears to be outputting same value repeatedly. Retrain with diverse data."
                                )

                except Exception as e:
                    self.add_check(
                        "results_file_read", "accuracy", False, "warning",
                        f"Could not read results file: {e}"
                    )
            else:
                self.add_check(
                    "results_file", "accuracy", False, "critical",
                    "Results file not found - possible fabrication",
                    suggestion=f"Ensure Stage 4 actually saves results to {STAGE4_OUT_DIR}. Check if file path is hallucinated."
                )

        return self.generate_report()


# ============================================================================
# STAGE 5 GUARDRAIL: Visualization Validation
# ============================================================================

class Stage5Guardrail(BaseGuardrail):
    """
    Validates Stage 5 (Visualization) outputs.

    Checks:
    - Plot files exist
    - Visualization report contains insights
    - Task answer is provided
    - Plots are appropriate for task type
    """

    def validate(self, stage_output: VisualizationReport, pipeline_state: PipelineState) -> StageGuardrailReport:
        logger.info(f"[Guardrail] Validating {self.stage_name}")

        # Check 1: Visualization report exists
        if not stage_output:
            self.add_check(
                "report_exists", "accuracy", False, "critical",
                "No visualization report generated",
                suggestion="Verify Stage 5 executed successfully"
            )
            return self.generate_report()

        self.add_check(
            "report_exists", "accuracy", True, "info",
            "Visualization report generated"
        )

        # Check 2: Visualizations exist
        if hasattr(stage_output, 'visualizations') and stage_output.visualizations:
            self.add_check(
                "visualizations_count", "business_logic", True, "info",
                f"Generated {len(stage_output.visualizations)} visualizations"
            )

            # Check each visualization file exists
            for i, viz in enumerate(stage_output.visualizations):
                if hasattr(viz, 'filepath') and viz.filepath:
                    viz_path = Path(viz.filepath)
                    if not viz_path.is_absolute():
                        viz_path = STAGE5_OUT_DIR / viz_path

                    if viz_path.exists():
                        self.add_check(
                            f"viz_file_{i}", "business_logic", True, "info",
                            f"Visualization file exists: {viz_path.name}"
                        )
                    else:
                        self.add_check(
                            f"viz_file_{i}", "business_logic", False, "warning",
                            f"Visualization file not found: {viz.filepath}",
                            suggestion="Check that Stage 5 saved all plot files"
                        )
        else:
            self.add_check(
                "visualizations_exist", "business_logic", False, "warning",
                "No visualizations generated",
                suggestion="Ensure Stage 5 created at least one plot"
            )

        # Check 3: Insights provided and hallucination detection
        if hasattr(stage_output, 'insights') and stage_output.insights:
            if len(stage_output.insights) > 0:
                self.add_check(
                    "insights", "business_logic", True, "info",
                    f"Generated {len(stage_output.insights)} insights"
                )

                # HALLUCINATION CHECK: Validate insights reference actual data
                task_id = pipeline_state.selected_task_id
                if task_id:
                    plan_id = f"PLAN-{task_id}" if not task_id.startswith("PLAN-") else task_id
                    results_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"

                    if results_path.exists():
                        try:
                            import pandas as pd
                            df = pd.read_parquet(results_path)

                            # Check if insights mention specific metrics that don't exist
                            insights_text = " ".join([str(i) for i in stage_output.insights])

                            # Common hallucination patterns
                            hallucination_indicators = [
                                ("MAE" in insights_text or "RMSE" in insights_text or "R2" in insights_text) and pipeline_state.stage4_output and not pipeline_state.stage4_output.metrics,
                                "accuracy" in insights_text.lower() and "actual" not in df.columns,
                            ]

                            if any(hallucination_indicators):
                                self.add_check(
                                    "insight_hallucination", "accuracy", False, "critical",
                                    "Insights reference metrics or data that don't exist in execution results",
                                    suggestion="Ensure insights are grounded in actual execution results, not fabricated"
                                )
                        except Exception as e:
                            pass  # Skip hallucination check if can't read results

            else:
                self.add_check(
                    "insights", "business_logic", False, "warning",
                    "No insights generated",
                    suggestion="Review Stage 5 insight generation"
                )
        else:
            self.add_check(
                "insights", "business_logic", False, "warning",
                "Insights field missing or empty"
            )

        # Check 4: Task answer provided and hallucination check
        if hasattr(stage_output, 'task_answer') and stage_output.task_answer:
            if len(stage_output.task_answer.strip()) > 10:  # Non-trivial answer
                self.add_check(
                    "task_answer", "business_logic", True, "info",
                    "Task answer provided"
                )

                # HALLUCINATION CHECK: Verify answer references actual results
                if pipeline_state.stage4_output and hasattr(pipeline_state.stage4_output, 'metrics'):
                    answer_text = stage_output.task_answer.lower()

                    # Check if answer makes specific claims without evidence
                    if ("predict" in answer_text or "forecast" in answer_text) and not pipeline_state.stage4_output.metrics:
                        self.add_check(
                            "answer_hallucination", "accuracy", False, "critical",
                            "Task answer makes predictions claims without supporting metrics",
                            suggestion="Ensure task answer is based on actual execution results, not fabricated"
                        )

                    # Check if answer cites specific numbers that seem unrealistic
                    import re
                    numbers = re.findall(r'\d+\.?\d*%?', answer_text)
                    if len(numbers) > 10:  # Too many specific numbers might indicate hallucination
                        self.add_check(
                            "excessive_precision", "accuracy", False, "warning",
                            f"Task answer contains {len(numbers)} specific numbers - verify all are from actual results",
                            suggestion="Cross-check all cited numbers against execution results"
                        )

            else:
                self.add_check(
                    "task_answer", "business_logic", False, "warning",
                    "Task answer is too short or empty",
                    suggestion="Provide detailed answer to the original task question"
                )
        else:
            self.add_check(
                "task_answer", "business_logic", False, "warning",
                "No task answer provided",
                suggestion="Ensure Stage 5 generates a comprehensive task answer"
            )

        return self.generate_report()


# ============================================================================
# STAGE 3B GUARDRAIL: DATA PREPARATION VALIDATOR
# ============================================================================

class Stage3bGuardrail(BaseGuardrail):
    """
    Guardrail for Stage 3B: Data Preparation

    Validates that data is properly prepared for modeling with:
    - ZERO null values (critical requirement)
    - Correct data transformations applied
    - Feature engineering completed
    - Data quality meets standards
    """

    def __init__(self):
        super().__init__("stage3b")

    def validate(self, state: PipelineState) -> StageGuardrailReport:
        """Run all validation checks for Stage 3B"""

        # === INPUT VALIDATION ===

        # Check execution plan exists
        if not state.stage3_output:
            self.add_check(
                "execution_plan_exists", "business_logic", False, "critical",
                "Stage 3 execution plan not found - cannot prepare data without plan",
                suggestion="Ensure Stage 3 planning completed successfully before data preparation"
            )
            return self.generate_report()

        # Check prepared data output exists
        if not state.stage3b_output:
            self.add_check(
                "stage3b_output_exists", "data_quality", False, "critical",
                "Stage 3B output not found - data preparation did not complete",
                suggestion="Ensure Stage 3B agent completes data preparation and saves output"
            )
            return self.generate_report()

        prepared_output = state.stage3b_output

        # === PROCESS VALIDATION ===

        # Check prepared file exists
        prepared_file = Path(prepared_output.prepared_file_path)
        if not prepared_file.exists():
            self.add_check(
                "prepared_file_exists", "data_quality", False, "critical",
                f"Prepared data file not found at {prepared_output.prepared_file_path}",
                suggestion="Ensure data preparation saves the prepared dataset to the specified path"
            )
            return self.generate_report()
        else:
            self.add_check(
                "prepared_file_exists", "data_quality", True, "info",
                f"Prepared data file exists at {prepared_output.prepared_file_path}"
            )

        # Check row count is reasonable
        if prepared_output.final_row_count == 0:
            self.add_check(
                "data_not_empty", "data_quality", False, "critical",
                "Prepared dataset has 0 rows - data was completely filtered out",
                suggestion="Review data filtering logic - dataset should not be empty after preparation"
            )
        elif prepared_output.final_row_count < prepared_output.original_row_count * 0.1:
            # Lost >90% of data
            pct_lost = ((prepared_output.original_row_count - prepared_output.final_row_count) /
                       prepared_output.original_row_count * 100)
            self.add_check(
                "excessive_data_loss", "data_quality", False, "warning",
                f"Lost {pct_lost:.1f}% of data during preparation (from {prepared_output.original_row_count} to {prepared_output.final_row_count} rows)",
                suggestion="Review data filters and join logic - significant data loss may indicate issues"
            )
        else:
            self.add_check(
                "row_count_reasonable", "data_quality", True, "info",
                f"Row count: {prepared_output.final_row_count} (from {prepared_output.original_row_count} original)"
            )

        # === OUTPUT VALIDATION (CRITICAL) ===

        # CRITICAL CHECK: Zero null values requirement
        quality_report = prepared_output.data_quality_report
        if quality_report.columns_with_nulls:
            null_summary = ", ".join([f"{col} ({quality_report.null_counts.get(col, 0)} nulls)"
                                     for col in quality_report.columns_with_nulls[:5]])
            if len(quality_report.columns_with_nulls) > 5:
                null_summary += f" and {len(quality_report.columns_with_nulls) - 5} more columns"

            self.add_check(
                "zero_nulls_requirement", "data_quality", False, "critical",
                f"Prepared data contains null values in {len(quality_report.columns_with_nulls)} columns: {null_summary}",
                suggestion="ALL null values must be handled via imputation, deletion, or feature engineering. Review missing value strategy and apply appropriate imputation."
            )
        else:
            self.add_check(
                "zero_nulls_requirement", "data_quality", True, "info",
                "Prepared data has zero null values - meets requirement"
            )

        # Check has_no_nulls flag consistency
        if prepared_output.has_no_nulls != (len(quality_report.columns_with_nulls) == 0):
            self.add_check(
                "null_flag_consistency", "data_quality", False, "warning",
                f"has_no_nulls flag ({prepared_output.has_no_nulls}) doesn't match actual null status",
                suggestion="Ensure has_no_nulls flag is set correctly after null handling"
            )

        # Check ready_for_modeling flag
        if not prepared_output.ready_for_modeling:
            self.add_check(
                "ready_for_modeling", "business_logic", False, "critical",
                "Data marked as NOT ready for modeling",
                suggestion="Review data preparation output and ensure all requirements are met before marking ready for modeling"
            )
        else:
            self.add_check(
                "ready_for_modeling", "business_logic", True, "info",
                "Data marked as ready for modeling"
            )

        # Check data quality warnings
        if quality_report.warnings:
            warning_summary = "; ".join(quality_report.warnings[:3])
            if len(quality_report.warnings) > 3:
                warning_summary += f" and {len(quality_report.warnings) - 3} more warnings"

            self.add_check(
                "data_quality_warnings", "data_quality", False, "warning",
                f"Data quality report contains {len(quality_report.warnings)} warnings: {warning_summary}",
                suggestion="Review data quality warnings and address significant issues"
            )

        # Verify actual data by loading and checking
        try:
            df = pd.read_csv(prepared_file)

            # Check actual null count
            actual_null_count = df.isnull().sum().sum()
            if actual_null_count > 0:
                self.add_check(
                    "actual_null_verification", "data_quality", False, "critical",
                    f"Loaded data contains {actual_null_count} null values despite reporting zero nulls",
                    suggestion="Data file contains nulls - rerun data preparation with proper null handling"
                )
            else:
                self.add_check(
                    "actual_null_verification", "data_quality", True, "info",
                    "Verified: Loaded data has zero null values"
                )

            # Check row count matches
            if len(df) != prepared_output.final_row_count:
                self.add_check(
                    "row_count_match", "data_quality", False, "warning",
                    f"Loaded data has {len(df)} rows but output reports {prepared_output.final_row_count} rows",
                    suggestion="Ensure row count in metadata matches actual saved file"
                )

            # Check column count matches
            if len(df.columns) != quality_report.total_columns:
                self.add_check(
                    "column_count_match", "data_quality", False, "warning",
                    f"Loaded data has {len(df.columns)} columns but output reports {quality_report.total_columns} columns",
                    suggestion="Ensure column count in metadata matches actual saved file"
                )

        except Exception as e:
            self.add_check(
                "data_loading_verification", "data_quality", False, "critical",
                f"Failed to load and verify prepared data: {str(e)}",
                suggestion="Ensure prepared data is saved in valid CSV format and is readable"
            )

        return self.generate_report()


# ============================================================================
# STAGE 3.5A GUARDRAIL: METHOD PROPOSAL VALIDATOR
# ============================================================================

class Stage3_5aGuardrail(BaseGuardrail):
    """
    Guardrail for Stage 3.5A: Method Proposal

    Validates that proposed methods are:
    - Appropriate for the task type (FORECASTING, REGRESSION, CLASSIFICATION, etc.)
    - Using actual column names (no hallucinated columns)
    - Properly structured with implementation code
    - Diverse (baseline, traditional, advanced ML)
    """

    def __init__(self):
        super().__init__("stage3_5a")

    def validate(self, state: PipelineState) -> StageGuardrailReport:
        """Run all validation checks for Stage 3.5A"""

        # === INPUT VALIDATION ===

        # Check execution plan exists
        if not state.stage3_output:
            self.add_check(
                "execution_plan_exists", "business_logic", False, "critical",
                "Stage 3 execution plan not found - cannot propose methods without plan",
                suggestion="Ensure Stage 3 planning completed successfully"
            )
            return self.generate_report()

        # Check prepared data exists
        if not state.stage3b_output:
            self.add_check(
                "prepared_data_exists", "business_logic", False, "critical",
                "Stage 3B prepared data not found - cannot propose methods without prepared data",
                suggestion="Ensure Stage 3B data preparation completed successfully"
            )
            return self.generate_report()

        # Check method proposal output exists
        if not state.stage3_5a_output:
            self.add_check(
                "method_proposal_exists", "data_quality", False, "critical",
                "Stage 3.5A output not found - method proposal did not complete",
                suggestion="Ensure Stage 3.5A agent completes method proposal and saves output"
            )
            return self.generate_report()

        proposal = state.stage3_5a_output
        plan = state.stage3_output

        # === PROCESS VALIDATION ===

        # CRITICAL: Exactly 3 methods required
        if len(proposal.methods_proposed) != 3:
            self.add_check(
                "exactly_three_methods", "business_logic", False, "critical",
                f"Expected exactly 3 methods, but {len(proposal.methods_proposed)} were proposed",
                suggestion="Propose exactly 3 methods: M1 (baseline), M2 (traditional/statistical), M3 (advanced ML)"
            )
        else:
            self.add_check(
                "exactly_three_methods", "business_logic", True, "info",
                "Exactly 3 methods proposed as required"
            )

        # Check method IDs are unique
        method_ids = [m.method_id for m in proposal.methods_proposed]
        if len(method_ids) != len(set(method_ids)):
            self.add_check(
                "unique_method_ids", "data_quality", False, "critical",
                f"Duplicate method IDs found: {method_ids}",
                suggestion="Each method must have a unique ID (e.g., M1, M2, M3)"
            )

        # Check method categories are diverse
        categories = [m.category.lower() for m in proposal.methods_proposed]
        expected_categories = ["baseline", "statistical", "ml"]

        if "baseline" not in categories:
            self.add_check(
                "has_baseline_method", "business_logic", False, "warning",
                "No baseline method proposed - should include a simple baseline",
                suggestion="Include a baseline method (e.g., naive forecast, mean prediction, etc.)"
            )

        # Check each method has implementation code
        for method in proposal.methods_proposed:
            if not method.implementation_code or len(method.implementation_code.strip()) < 50:
                self.add_check(
                    f"method_{method.method_id}_implementation", "business_logic", False, "critical",
                    f"Method {method.method_id} ({method.name}) has insufficient implementation code",
                    suggestion=f"Provide complete, executable implementation code for {method.name}"
                )

        # === OUTPUT VALIDATION (CRITICAL - HALLUCINATION CHECKS) ===

        # Load prepared data to verify column references
        try:
            prepared_file = Path(state.stage3b_output.prepared_file_path)
            if prepared_file.exists():
                df = pd.read_csv(prepared_file)
                actual_columns = set(df.columns)

                # Check target column exists
                if proposal.target_column not in actual_columns:
                    self.add_check(
                        "target_column_hallucination", "accuracy", False, "critical",
                        f"Target column '{proposal.target_column}' does NOT exist in prepared data. Available columns: {sorted(actual_columns)}",
                        suggestion=f"Use only columns that exist in the data. Call get_actual_columns() to see available columns and update target_column."
                    )
                else:
                    self.add_check(
                        "target_column_valid", "accuracy", True, "info",
                        f"Target column '{proposal.target_column}' exists in prepared data"
                    )

                # Check date column if specified
                if proposal.date_column and proposal.date_column not in actual_columns:
                    self.add_check(
                        "date_column_hallucination", "accuracy", False, "critical",
                        f"Date column '{proposal.date_column}' does NOT exist in prepared data. Available columns: {sorted(actual_columns)}",
                        suggestion=f"Use only columns that exist in the data. If no date column exists, set date_column=None and use df.index."
                    )

                # Check feature columns exist
                invalid_features = [col for col in proposal.feature_columns if col not in actual_columns]
                if invalid_features:
                    self.add_check(
                        "feature_columns_hallucination", "accuracy", False, "critical",
                        f"Feature columns do NOT exist in prepared data: {invalid_features}. Available: {sorted(actual_columns)}",
                        suggestion="Use only columns that exist in the data. Call get_actual_columns() to verify column names."
                    )

                # Check if any methods reference non-existent columns in their code
                for method in proposal.methods_proposed:
                    code = method.implementation_code
                    # Look for common column reference patterns
                    import re
                    # Find strings in quotes that might be column names
                    potential_cols = re.findall(r"['\"]([^'\"]+)['\"]", code)
                    hallucinated_in_code = [col for col in potential_cols
                                           if col not in actual_columns
                                           and len(col) > 2 and len(col) < 30
                                           and col[0].isupper()]  # Likely column names

                    if hallucinated_in_code:
                        self.add_check(
                            f"method_{method.method_id}_column_references", "accuracy", False, "warning",
                            f"Method {method.method_id} may reference non-existent columns in code: {hallucinated_in_code[:3]}",
                            suggestion=f"Review {method.name} implementation code and ensure all column references are valid"
                        )

        except Exception as e:
            self.add_check(
                "column_verification", "data_quality", False, "warning",
                f"Could not verify column references: {str(e)}",
                suggestion="Ensure prepared data file is accessible for validation"
            )

        # Check task type appropriateness
        task_category = plan.task_category.lower() if hasattr(plan, 'task_category') else "unknown"

        # Verify methods match task type
        forecasting_keywords = ["arima", "prophet", "lstm", "sarima", "ets", "exponential", "seasonal"]
        classification_keywords = ["logistic", "classifier", "svm", "decision tree", "random forest classifier"]
        regression_keywords = ["linear regression", "ridge", "lasso", "random forest regressor", "xgboost regressor"]

        for method in proposal.methods_proposed:
            method_name_lower = method.name.lower()
            method_desc_lower = method.description.lower()
            combined = method_name_lower + " " + method_desc_lower

            # Check if task is forecasting but method isn't
            if "forecast" in task_category:
                has_forecast_method = any(kw in combined for kw in forecasting_keywords)
                if not has_forecast_method and method.category != "baseline":
                    self.add_check(
                        f"method_{method.method_id}_task_mismatch", "business_logic", False, "warning",
                        f"Task is FORECASTING but method '{method.name}' doesn't appear to be a forecasting method",
                        suggestion="For forecasting tasks, use time series methods like ARIMA, Prophet, LSTM, etc."
                    )

            # Check if task is classification but method isn't
            elif "classification" in task_category:
                has_classification_method = any(kw in combined for kw in classification_keywords)
                if not has_classification_method and method.category != "baseline":
                    self.add_check(
                        f"method_{method.method_id}_task_mismatch", "business_logic", False, "warning",
                        f"Task is CLASSIFICATION but method '{method.name}' doesn't appear to be a classification method",
                        suggestion="For classification tasks, use classifiers like Logistic Regression, Random Forest Classifier, etc."
                    )

        return self.generate_report()


# ============================================================================
# STAGE 3.5B GUARDRAIL: BENCHMARKING VALIDATOR
# ============================================================================

class Stage3_5bGuardrail(BaseGuardrail):
    """
    Guardrail for Stage 3.5B: Method Benchmarking

    Validates that benchmarking results are:
    - Consistent across multiple iterations (not hallucinated)
    - Complete for all proposed methods
    - Using correct evaluation metrics
    - Best method selection is justified
    """

    def __init__(self):
        super().__init__("stage3_5b")

    def validate(self, state: PipelineState) -> StageGuardrailReport:
        """Run all validation checks for Stage 3.5B"""

        # === INPUT VALIDATION ===

        # Check method proposals exist
        if not state.stage3_5a_output:
            self.add_check(
                "method_proposals_exist", "business_logic", False, "critical",
                "Stage 3.5A method proposals not found - cannot benchmark without methods",
                suggestion="Ensure Stage 3.5A method proposal completed successfully"
            )
            return self.generate_report()

        # Check prepared data exists
        if not state.stage3b_output:
            self.add_check(
                "prepared_data_exists", "business_logic", False, "critical",
                "Stage 3B prepared data not found - cannot benchmark without data",
                suggestion="Ensure Stage 3B data preparation completed successfully"
            )
            return self.generate_report()

        # Check benchmarking output exists
        if not state.stage3_5b_output:
            self.add_check(
                "benchmarking_output_exists", "data_quality", False, "critical",
                "Stage 3.5B output not found - benchmarking did not complete",
                suggestion="Ensure Stage 3.5B agent completes benchmarking and saves output"
            )
            return self.generate_report()

        benchmark = state.stage3_5b_output
        proposals = state.stage3_5a_output

        # === PROCESS VALIDATION ===

        # Check all proposed methods were tested
        proposed_method_ids = {m.method_id for m in proposals.methods_proposed}
        tested_method_ids = {m.method_id for m in benchmark.methods_tested}

        untested_methods = proposed_method_ids - tested_method_ids
        if untested_methods:
            self.add_check(
                "all_methods_tested", "business_logic", False, "critical",
                f"Methods were proposed but not tested: {untested_methods}",
                suggestion=f"Benchmark ALL proposed methods: {sorted(proposed_method_ids)}"
            )
        else:
            self.add_check(
                "all_methods_tested", "business_logic", True, "info",
                f"All {len(proposed_method_ids)} proposed methods were tested"
            )

        # Check selected method is valid
        if benchmark.selected_method_id not in tested_method_ids:
            self.add_check(
                "selected_method_valid", "business_logic", False, "critical",
                f"Selected method '{benchmark.selected_method_id}' was not in tested methods: {tested_method_ids}",
                suggestion="Select a method that was actually benchmarked"
            )

        # === OUTPUT VALIDATION (CRITICAL - HALLUCINATION CHECKS) ===

        # Check each method has multiple iterations (consistency check)
        from code.config import BENCHMARK_ITERATIONS
        expected_iterations = BENCHMARK_ITERATIONS

        for method_result in benchmark.methods_tested:
            # Check if method has iterations
            if hasattr(method_result, 'iterations') and method_result.iterations:
                num_iterations = len(method_result.iterations)

                if num_iterations < expected_iterations and method_result.is_valid:
                    self.add_check(
                        f"method_{method_result.method_id}_iterations", "data_quality", False, "warning",
                        f"Method {method_result.method_id} only has {num_iterations} iterations (expected {expected_iterations})",
                        suggestion=f"Run each method {expected_iterations} times for consistency validation"
                    )

                # HALLUCINATION CHECK: Verify consistency across iterations
                if num_iterations >= 2 and method_result.is_valid:
                    # Calculate coefficient of variation for consistency
                    # Lower CV means more consistent (less likely hallucinated)
                    if hasattr(method_result, 'coefficient_of_variation') and method_result.coefficient_of_variation is not None:
                        cv = method_result.coefficient_of_variation

                        # CV > 0.5 (50%) indicates high variability - possible hallucination
                        if cv > 0.5:
                            self.add_check(
                                f"method_{method_result.method_id}_consistency", "accuracy", False, "critical",
                                f"Method {method_result.method_id} has high variability (CV={cv:.2%}) across iterations - results may be hallucinated or unstable",
                                suggestion="Results are inconsistent across runs. Verify method implementation is deterministic or uses proper random seeding. May indicate fabricated results."
                            )
                        elif cv > 0.2:
                            self.add_check(
                                f"method_{method_result.method_id}_consistency", "accuracy", False, "warning",
                                f"Method {method_result.method_id} has moderate variability (CV={cv:.2%}) across iterations",
                                suggestion="Results vary across runs. Consider increasing iterations or checking for randomness in method implementation."
                            )
                        else:
                            self.add_check(
                                f"method_{method_result.method_id}_consistency", "accuracy", True, "info",
                                f"Method {method_result.method_id} has consistent results (CV={cv:.2%})"
                            )

            # Check for failed methods
            if not method_result.is_valid:
                if method_result.failure_reason:
                    self.add_check(
                        f"method_{method_result.method_id}_failure", "business_logic", False, "warning",
                        f"Method {method_result.method_id} failed: {method_result.failure_reason}",
                        suggestion=f"Review and fix the implementation for {method_result.method_id}"
                    )

        # Check that best method has valid results
        best_method_result = next((m for m in benchmark.methods_tested
                                   if m.method_id == benchmark.selected_method_id), None)

        if best_method_result:
            if not best_method_result.is_valid:
                self.add_check(
                    "best_method_invalid", "business_logic", False, "critical",
                    f"Selected best method '{benchmark.selected_method_id}' has invalid results",
                    suggestion="Cannot select a method that failed benchmarking - choose a method with valid results"
                )

            # Check best method has average metrics
            if hasattr(best_method_result, 'average_metrics') and best_method_result.average_metrics:
                # Verify metrics are reasonable (not NaN, not negative for most metrics)
                avg_metrics = best_method_result.average_metrics

                # Check for NaN or None values
                metric_dict = avg_metrics.model_dump() if hasattr(avg_metrics, 'model_dump') else avg_metrics.__dict__
                none_metrics = [k for k, v in metric_dict.items() if v is None or (isinstance(v, float) and math.isnan(v))]

                if none_metrics:
                    self.add_check(
                        "best_method_metrics_complete", "accuracy", False, "warning",
                        f"Best method has incomplete metrics: {none_metrics}",
                        suggestion="Ensure all evaluation metrics are calculated successfully"
                    )
        else:
            self.add_check(
                "best_method_found", "business_logic", False, "critical",
                f"Selected method '{benchmark.selected_method_id}' not found in tested methods",
                suggestion="Ensure selected_method_id matches one of the benchmarked methods"
            )

        # Check selection rationale is provided
        if not benchmark.selection_rationale or len(benchmark.selection_rationale.strip()) < 20:
            self.add_check(
                "selection_rationale", "business_logic", False, "warning",
                "Selection rationale is missing or too brief",
                suggestion="Provide clear rationale explaining why this method was selected (based on metrics)"
            )

        # HALLUCINATION CHECK: Verify rationale mentions actual metrics
        if benchmark.selection_rationale:
            rationale_lower = benchmark.selection_rationale.lower()
            mentions_metrics = any(word in rationale_lower for word in ['mae', 'rmse', 'mape', 'accuracy', 'precision', 'recall', 'f1', 'r2', 'mse'])

            if not mentions_metrics:
                self.add_check(
                    "rationale_references_metrics", "accuracy", False, "warning",
                    "Selection rationale doesn't reference specific metrics",
                    suggestion="Rationale should cite actual metric values (e.g., 'Selected M2 because it achieved lowest MAE of 0.15')"
                )

        return self.generate_report()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "GuardrailCheckResult",
    "StageGuardrailReport",
    "GuardrailReport",
    "BaseGuardrail",
    "Stage1Guardrail",
    "Stage2Guardrail",
    "Stage3Guardrail",
    "Stage3bGuardrail",
    "Stage3_5aGuardrail",
    "Stage3_5bGuardrail",
    "Stage4Guardrail",
    "Stage5Guardrail",
]
