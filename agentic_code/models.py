"""
Pydantic models for all stages of the unified agentic AI pipeline.

Contains data models for Stage 1-5 including dataset summaries, task proposals,
execution plans, results, and visualization reports.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal, Tuple, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field

# ===========================
# Stage 1: Dataset Summarization
# ===========================

LogicalType = Literal[
    "numeric", "integer", "float", "categorical",
    "text", "datetime", "boolean", "unknown"
]


class ColumnSummary(BaseModel):
    """Summary of a single column in a dataset."""
    name: str
    physical_dtype: str
    logical_type: LogicalType
    description: str = Field(
        description="Short natural-language description of what the column represents."
    )
    nullable: bool
    null_fraction: float
    unique_fraction: float
    examples: List[str] = Field(default_factory=list)
    is_potential_key: bool = False


class DatasetSummary(BaseModel):
    """Summary of an entire dataset."""
    dataset_name: str
    path: str
    approx_n_rows: Optional[int] = None
    columns: List[ColumnSummary]
    candidate_primary_keys: List[List[str]] = Field(
        default_factory=list,
        description="Each entry is a list of column names that could form a primary key."
    )
    notes: Optional[str] = None


# ===========================
# Stage 2: Task Proposal
# ===========================

TaskCategory = Literal["predictive", "descriptive", "unsupervised"]


class JoinPlan(BaseModel):
    """Plan for joining datasets."""
    hypothesized_keys: List[List[str]] = Field(
        default_factory=list,
        description="Each inner list is a set of columns that might be join keys between files."
    )
    notes: Optional[str] = None


class TargetSpec(BaseModel):
    """Specification for prediction target."""
    name: Optional[str] = None
    granularity: Optional[List[str]] = None
    horizon: Optional[str] = None  # e.g. '1-year ahead'


class FeaturePlan(BaseModel):
    """Plan for feature engineering."""
    candidates: List[str] = Field(
        default_factory=list,
        description="Column names or wildcard patterns (e.g. 'Area-*')."
    )
    transform_ideas: List[str] = Field(
        default_factory=list,
        description="Free-text feature engineering ideas."
    )
    handling_missingness: Optional[str] = None


class TaskProposal(BaseModel):
    """Proposed analytical task."""
    id: str
    category: TaskCategory
    title: str
    problem_statement: str
    required_files: List[str] = Field(
        default_factory=list,
        description="Dataset filenames needed for this task."
    )
    join_plan: JoinPlan = Field(default_factory=JoinPlan)
    target: Optional[TargetSpec] = None
    feature_plan: FeaturePlan = Field(default_factory=FeaturePlan)
    validation_plan: Optional[str] = None
    quality_checks: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)


class Stage2Output(BaseModel):
    """Output from Stage 2 containing all task proposals."""
    proposals: List[TaskProposal]


# ===========================
# Stage 3: Execution Planning
# ===========================

class ArtifactSpec(BaseModel):
    """Specification for output artifacts."""
    intermediate_table: str = Field(description="Filename for the output table")
    intermediate_format: Literal["parquet", "csv", "feather"] = "parquet"
    expected_columns: List[str] = Field(
        default_factory=list,
        description="List of all columns expected in final table"
    )
    expected_row_count_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Expected min/max row count"
    )


class KeyNormalization(BaseModel):
    """Normalization rules for join keys."""
    column_name: str
    mapping: Dict[str, str] = Field(default_factory=dict)
    format_type: Optional[str] = None
    valid_range: Optional[Tuple[Any, Any]] = None


class FileInstruction(BaseModel):
    """Instructions for loading and preprocessing a file."""
    file_id: str
    original_name: str
    alias: str
    filters: List[str] = Field(default_factory=list)
    rename_columns: Dict[str, str] = Field(default_factory=dict)
    join_keys: List[str] = Field(default_factory=list)
    keep_columns: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class JoinValidation(BaseModel):
    """Validation rules for joins."""
    check_duplicates_on_keys: List[str] = Field(default_factory=list)
    expected_unique: bool = False
    check_row_count_stable: bool = False
    check_no_duplicates_introduced: bool = False
    acceptable_coverage: Optional[float] = None
    max_cardinality_ratio: Optional[float] = None


class JoinStep(BaseModel):
    """A single join operation."""
    step: int
    description: str
    left_table: str
    right_table: Optional[str] = None
    join_type: Literal["base", "inner", "left", "right", "outer"] = "base"
    join_keys: List[str] = Field(default_factory=list)
    expected_cardinality: str = "base"
    validation: JoinValidation = Field(default_factory=JoinValidation)


class FeatureEngineering(BaseModel):
    """Feature engineering specification."""
    feature_name: str
    description: str
    transform: str
    depends_on: List[str] = Field(default_factory=list)
    implementation: Optional[str] = None


class TimeSplit(BaseModel):
    """Time-based train/test split specification."""
    method: Literal["year-based", "date-based", "rolling-window", "none"] = "none"
    train_years: Optional[str] = None
    test_years: Optional[str] = None
    validation_years: Optional[str] = None
    leakage_check: str = "Not applicable"


class CoverageCheck(BaseModel):
    """Data coverage validation check."""
    check: str
    threshold: float
    description: str
    action_if_violation: Optional[str] = None


class CardinalityCheck(BaseModel):
    """Cardinality validation check."""
    check: str
    expected: str
    action_if_violation: str


class ValidationSpec(BaseModel):
    """Validation strategy for the pipeline."""
    time_split: Optional[TimeSplit] = Field(default_factory=lambda: TimeSplit())
    coverage_checks: List[CoverageCheck] = Field(default_factory=list)
    cardinality_checks: List[CardinalityCheck] = Field(default_factory=list)
    additional_checks: List[str] = Field(default_factory=list)


class Stage3Plan(BaseModel):
    """Complete execution plan from Stage 3."""
    # Metadata
    plan_id: str
    selected_task_id: str
    goal: str
    task_category: TaskCategory
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = Field(default="stage3_agent")
    
    # Output specification
    artifacts: ArtifactSpec
    
    # Key normalization
    key_normalization: List[KeyNormalization] = Field(default_factory=list)
    
    # File loading instructions
    file_instructions: List[FileInstruction] = Field(default_factory=list)
    
    # Join strategy
    join_steps: List[JoinStep] = Field(default_factory=list)
    
    # Feature engineering
    feature_engineering: List[FeatureEngineering] = Field(default_factory=list)
    
    # Validation strategy
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    
    # Expected models and metrics
    expected_model_types: List[str] = Field(default_factory=list)
    evaluation_metrics: List[str] = Field(default_factory=list)
    
    # Documentation
    notes: List[str] = Field(default_factory=list)


# ===========================
# Stage 4: Execution Results
# ===========================

class ExecutionResult(BaseModel):
    """Results from Stage 4 execution."""
    plan_id: str
    task_category: TaskCategory
    status: Literal["success", "failure", "partial"]
    outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of output type to file path"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics (if applicable)"
    )
    summary: str
    errors: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ===========================
# Stage 5: Visualization Reports
# ===========================

class VisualizationReport(BaseModel):
    """Report from Stage 5 visualization."""
    plan_id: str
    task_category: TaskCategory
    visualizations: List[str] = Field(
        default_factory=list,
        description="Paths to created visualization files"
    )
    html_report: Optional[str] = Field(
        default=None,
        description="Path to HTML report (if created)"
    )
    summary: str
    insights: List[str] = Field(
        default_factory=list,
        description="Key insights from the visualizations"
    )
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ===========================
# Master Pipeline State
# ===========================

class PipelineState(BaseModel):
    """Unified state for the entire pipeline."""
    current_stage: int = 1  # Current stage (1-5)
    
    # Stage outputs
    dataset_summaries: List[DatasetSummary] = Field(default_factory=list)
    task_proposals: List[TaskProposal] = Field(default_factory=list)
    selected_task_id: Optional[str] = None
    stage3_plan: Optional[Stage3Plan] = None
    execution_result: Optional[ExecutionResult] = None
    visualization_report: Optional[VisualizationReport] = None
    
    # Tracking
    errors: List[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_stages: List[int] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


# ===========================
# Stage 0: Conversation State
# ===========================

class ConversationState(TypedDict):
    """State for conversational interactions."""
    query: str  # Current user question
    conversation_history: List[Dict[str, str]]  # Past Q&A
    available_datasets: List[str]  # Known datasets
    completed_tasks: List[str]  # Tasks already executed
    current_plan: Optional[str]  # Current execution plan
    response: Optional[str]  # Agent's response
