# Conversational AI Forecasting Pipeline

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Pipeline Stages](#pipeline-stages)
- [Safety & Validation](#safety--validation)
- [Checkpointing & State Management](#checkpointing--state-management)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development Guide](#development-guide)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **Conversational AI Forecasting Pipeline** is a production-grade, dataset-agnostic system for automated time series forecasting with a conversational interface. Built on LangGraph, it orchestrates multiple LLM-powered agents across 8 sequential stages, from data analysis through to visualization and reporting, with comprehensive safety measures and validation at every step.

### What Makes This Unique

1. **Dataset-Agnostic**: Automatically analyzes any CSV data, proposes appropriate forecasting tasks, and executes them without hardcoded assumptions
2. **Conversational Interface**: Natural language interaction for data exploration, task creation, and pipeline control
3. **Automated Method Selection**: Proposes 3 forecasting methods, benchmarks them with multiple iterations, and selects the best performer
4. **Production-Ready**: Comprehensive error handling, automatic retries, checkpointing, and validation at every stage
5. **Statistical Guardrails**: Multi-layered validation including correlation analysis, propensity scoring, and residual analysis
6. **Full Transparency**: Complete thought process documentation from data profiling to final predictions

### Use Cases

- **Automated Forecasting**: Time series prediction for sales, production, demand, etc.
- **Data Exploration**: Interactive analysis of datasets with the built-in EDA agent
- **Model Benchmarking**: Systematic comparison of multiple forecasting approaches
- **Research & Development**: Reproducible pipeline for machine learning experiments

---

## Architecture

### System Design

The pipeline follows a **multi-stage orchestration** pattern using **LangGraph** for workflow management:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL INTERFACE                       │
│  (Natural language queries → Pipeline actions)                    │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MASTER ORCHESTRATOR                            │
│  (LangGraph-based state machine with MemorySaver checkpointing)  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌───────────────┐              ┌────────────────┐
│  ANALYSIS     │              │  FORECASTING   │
│  PIPELINE     │              │  PIPELINE      │
│               │              │                │
│  Stage 1      │              │  Stage 3       │
│  Stage 2      │              │  Stage 3B      │
│               │              │  Stage 3.5A    │
│               │              │  Stage 3.5B    │
│               │              │  Stage 4       │
│               │              │  Stage 5       │
│               │              │  Stage 6       │
│               │              │  Stage 7       │
└───────────────┘              └────────────────┘
```

### Core Components

#### 1. Master Orchestrator (`master_orchestrator.py`)
- **Purpose**: Central coordinator for all pipeline stages
- **Technology**: LangGraph StateGraph with MemorySaver checkpointing
- **Responsibilities**:
  - Stage sequencing and conditional routing
  - State management across stages
  - Error handling and retry logic
  - Resumability from checkpoints

#### 2. Configuration System (`config.py`)
- **Purpose**: Centralized configuration management
- **Key Features**:
  - LLM endpoint configuration (PRIMARY and SECONDARY)
  - Directory structure definition
  - Stage-specific parameters (max_rounds, max_tokens)
  - Dynamic metrics configuration
  - Retry policies
  - LangSmith tracing integration

#### 3. Data Models (`models.py`)
- **Purpose**: Type-safe data structures with Pydantic
- **Key Models**:
  - `PipelineState`: Central state object for all stages
  - `DatasetSummary`: Stage 1 output
  - `TaskProposal`: Stage 2 output
  - `ExecutionPlan`: Stage 3 output
  - `PreparedData`: Stage 3B output
  - `MethodProposal`: Stage 3.5A output
  - `TesterOutput`: Stage 3.5B output
  - `ExecutionResult`: Stage 4 output
  - `VisualizationReport`: Stage 5 output

#### 4. Utilities (`utils.py`)
- **Purpose**: Shared functionality across stages
- **Key Features**:
  - LLM wrapper functions
  - File I/O utilities
  - Data validation
  - Logging configuration

#### 5. Data Passing Manager (`config.py`)
- **Purpose**: Robust inter-stage communication
- **Safety Features**:
  - Atomic writes (temp file → rename)
  - SHA256 checksums for integrity
  - Metadata envelopes for all artifacts
  - Parquet + sidecar JSON for data files

---

## Key Features

### 1. **Intelligent Task Proposal**
Stage 2 uses a ReAct agent to explore datasets and propose analytical tasks based on:
- Data characteristics (temporal, categorical, numerical)
- Column relationships
- Feasibility assessment
- Automatically selected evaluation metrics

### 2. **Automated Method Benchmarking**
Stage 3.5B runs each proposed method **3 times** to:
- Ensure consistency (coefficient of variation < 10%)
- Detect hallucinated results
- Select the best performer based on task-appropriate metrics
- Validate data split strategies

### 3. **Dynamic Metrics**
The pipeline does **NOT** hardcode metrics like MAE/RMSE:
- Stage 2 determines appropriate metrics based on task type
- Forecasting → mae, rmse, mape, r2
- Classification → accuracy, precision, recall, f1
- Metrics flow through all subsequent stages

### 4. **Comprehensive Error Handling**
Every stage includes:
- Try-catch error boundaries
- Automatic retry logic (configurable retries per stage)
- Detailed error logging with stack traces
- Graceful degradation with fallback strategies
- No silent failures

### 5. **Checkpointing & Resumability**
- **LangGraph MemorySaver**: Agent-level checkpointing within stages
- **Stage Output Caching**: Completed stages can be skipped on restart
- **Benchmark Checkpoints**: Stage 3.5B saves progress after each method
- **Conversation State**: User sessions persisted to disk

### 6. **Statistical Validation (Stage 7)**
Guardrails agent validates model predictions with:
- **Correlation Analysis**: Checks feature-target relationships
- **Propensity Score Analysis**: Detects selection bias
- **Residual Analysis**: Validates error distribution properties
- **Visualizations**: Distribution plots, scatter plots, error analysis
- **Final Verdict**: PASS/WARNING/FAIL with detailed rationale

### 7. **Conversational Interface**
Natural language interaction for:
- Data exploration with EDA agent
- Task creation from user queries
- Pipeline execution
- Result retrieval
- Status monitoring

---

## Pipeline Stages

### Stage 1: Dataset Analysis (`stage1_agent.py`)

**Purpose**: Automated data profiling and quality assessment

**Process**:
1. Scans `data/` directory for CSV files
2. For each dataset:
   - Loads and analyzes structure
   - Identifies column types (datetime, numeric, categorical)
   - Computes quality metrics (missing values, unique values)
   - Detects potential target columns
   - Identifies datetime columns for forecasting
3. Saves individual summaries as `{dataset}.summary.json`

**Output**: `DatasetSummary` objects for each dataset

**Key Features**:
- **Type Inference**: Automatic datetime column detection
- **Quality Scoring**: 0-100% based on completeness and validity
- **Target Identification**: Statistical analysis to find forecast candidates
- **Error Handling**: Continues processing even if individual files fail

**Files Created**:
```
output/summaries/
  ├── dataset1.summary.json
  ├── dataset2.summary.json
  └── ...
```

---

### Stage 2: Task Proposal (`stage2_agent.py`)

**Purpose**: Intelligent task generation from data analysis

**Process**:
1. Loads all dataset summaries from Stage 1
2. ReAct agent explores data characteristics
3. Proposes 3-5 analytical tasks based on:
   - Available datetime columns
   - Numeric targets suitable for prediction
   - Data quality and completeness
   - Domain-appropriate metrics
4. Each proposal includes:
   - Task ID (TSK-XXX)
   - Title and problem statement
   - Dataset and column mappings
   - Evaluation metrics (dynamically selected)
   - Feasibility score

**Output**: `TaskProposalOutput` with list of `TaskProposal` objects

**Key Features**:
- **ReAct Framework**: Iterative reasoning with tool calls
- **Dynamic Metrics**: Selects appropriate metrics for task type
- **Feasibility Scoring**: Assesses likelihood of success
- **Custom Queries**: Can generate tasks from user natural language queries

**Tools Available**:
- `list_available_datasets`: View analyzed datasets
- `load_dataset_summary`: Get detailed info on a dataset
- `propose_task`: Create a new task proposal
- `save_proposals`: Persist final proposals

**Files Created**:
```
output/stage2_out/
  └── task_proposals.json
```

---

### Stage 3: Execution Planning (`stage3_agent.py`)

**Purpose**: Create detailed execution plan for a specific task

**Process**:
1. Loads task proposal (TSK-XXX)
2. Analyzes data requirements
3. Creates comprehensive execution plan including:
   - Data loading and preparation steps
   - Feature engineering approach
   - Validation strategy (train/val/test split)
   - Forecasting horizon and granularity
   - Expected output format
4. Determines data split strategy (temporal, column-based, hybrid)

**Output**: `ExecutionPlan` (PLAN-TSK-XXX)

**Key Features**:
- **Extended Reasoning**: 50 max rounds for complex planning
- **Strategy Discovery**: Infers split strategy from data structure
- **Forecast Configuration**: Defines horizon and granularity
- **Validation Design**: Specifies metrics and evaluation approach

**Files Created**:
```
output/stage3_out/
  └── PLAN-TSK-XXX.json
```

---

### Stage 3B: Data Preparation (`stage3b_agent.py`)

**Purpose**: Load, clean, and prepare data for modeling

**Process**:
1. Loads raw CSV based on execution plan
2. Performs data cleaning:
   - Handles missing values
   - Converts data types
   - Filters rows/columns as needed
   - Creates datetime index if applicable
3. Validates data quality
4. Saves as Parquet for efficient loading

**Output**: `PreparedData` with metadata

**Key Features**:
- **Robust Parsing**: Handles various CSV formats
- **Type Conversion**: Automatic datetime parsing
- **Quality Checks**: Validates before saving
- **Efficient Storage**: Parquet format with compression

**Files Created**:
```
output/stage3b_data_prep/
  ├── prepared_PLAN-TSK-XXX.parquet
  └── prepared_PLAN-TSK-XXX.meta.json
```

**Validation**:
- Schema validation (required columns present)
- Data integrity checks (no corruption)
- Size validation (non-empty dataset)

---

### Stage 3.5A: Method Proposal (`stage3_5a_agent.py`)

**Purpose**: Propose 3 forecasting methods with full implementations

**Process**:
1. Loads execution plan and prepared data
2. Analyzes data characteristics
3. Proposes **exactly 3 methods** appropriate for the data:
   - Method 1: Simple baseline (e.g., moving average)
   - Method 2: Statistical model (e.g., ARIMA, exponential smoothing)
   - Method 3: Machine learning (e.g., random forest, gradient boosting)
4. For each method, provides:
   - Complete Python implementation
   - Data split strategy
   - Hyperparameters
   - Rationale for selection

**Output**: `MethodProposal` with 3 `Method` objects

**Key Features**:
- **Full Code Generation**: Complete, executable implementations
- **Strategy Specification**: Explicit data split instructions
- **Diversity**: Different approaches for robust comparison
- **Hallucination Prevention**: Verifies column names exist

**Pydantic Validation**:
- Enforces exactly 3 methods (raises error if not)
- Validates required fields (name, code, strategy)
- Type checking for all parameters

**Files Created**:
```
output/stage3_5a_method_proposal/
  └── method_proposal_PLAN-TSK-XXX.json
```

---

### Stage 3.5B: Method Benchmarking (`stage3_5b_agent.py`)

**Purpose**: Test methods and select the best performer

**Process**:
1. Loads method proposals from Stage 3.5A
2. For each of the 3 methods:
   - Runs **3 iterations** with identical setup
   - Executes the exact code from Stage 3.5A
   - Calculates all metrics from execution plan
   - Records results for each iteration
3. Validates consistency:
   - Computes coefficient of variation (CV) across iterations
   - If CV < 10%: Results are valid
   - If CV ≥ 10%: Results may be hallucinated
4. Selects best method based on primary metric
5. Saves comprehensive benchmark results

**Output**: `TesterOutput` with selected method and all results

**Key Features**:
- **Consistency Validation**: 3 iterations to detect unreliable results
- **Hallucination Detection**: Statistical validation of outputs
- **Checkpointing**: Saves progress after each method (resumable)
- **Automatic Retry**: Up to 3 retries on failure
- **Dynamic Metrics**: Uses task-specific evaluation metrics

**Safety Measures**:
- Column name verification before execution
- Code execution sandboxing
- Timeout protection
- Error isolation (one method failure doesn't block others)

**Files Created**:
```
output/stage3_5b_benchmarking/
  ├── tester_PLAN-TSK-XXX.json
  └── checkpoint_PLAN-TSK-XXX.json (intermediate)
```

**Retry Logic**:
```python
max_retries = 3
for attempt in range(1, max_retries + 1):
    try:
        output = run_stage3_5b(plan_id)
        # Success - return
        return output
    except Exception as e:
        if is_retryable(e) and attempt < max_retries:
            # Clean partial outputs
            # Retry
            continue
        else:
            # Fail
            raise
```

---

### Stage 4: Execution (`stage4_agent.py`)

**Purpose**: Execute the winning method and generate predictions

**Process**:
1. **PREFERRED**: Load model checkpoint from Stage 3.5B
   - Guarantees identical results
   - No retraining needed
   - Fast execution
2. **FALLBACK**: Retrain model if no checkpoint
   - Loads prepared data
   - Executes winning method code
   - Uses exact data split strategy
3. Generates **two types of predictions**:
   - **Test Set Predictions**: For validation
   - **Future Forecasts**: If forecast_horizon > 0
4. Validates metrics match Stage 3.5B (within ±5%)
5. Saves predictions and metadata

**Output**: `ExecutionResult` with predictions DataFrame

**Key Features**:
- **Model Persistence**: Loads saved models when available
- **Metric Validation**: Ensures consistency with benchmarking
- **Future Forecasting**: Recursive prediction for N periods
- **Automatic Retry**: Up to 3 attempts with error feedback
- **Detailed Logging**: Full execution trace for debugging

**Safety Measures**:
- Validation that test MAE matches benchmark
- NaN/Inf detection in metrics
- Predictions file existence check
- Error context passed to retry attempts

**Files Created**:
```
output/stage4_out/
  ├── results_PLAN-TSK-XXX.parquet (predictions)
  ├── execution_result_PLAN-TSK-XXX.json (metadata)
  └── workspace/
      └── {task_id}_model.pkl (model checkpoint, if applicable)
```

**Results DataFrame Schema**:
- `date/index`: Time or row identifier
- `actual`: True values (NaN for future forecasts)
- `predicted`: Model predictions
- `prediction_type`: 'test' or 'forecast'
- Original feature columns for context

**Retry Strategy**:
```python
for attempt in range(1, max_retries + 1):
    try:
        # Attempt execution
        result = _attempt_stage4_execution(plan_id, attempt)
        if result.status == SUCCESS:
            return result
        # Continue to next attempt if failed
    except Exception as e:
        # Log error and retry
```

---

### Stage 5: Visualization (`stage5_agent.py`)

**Purpose**: Create visualizations and generate insights

**Process**:
1. Loads task context and execution results
2. **Data Structure Analysis**:
   - Analyzes available columns (categorical, numerical, temporal)
   - Identifies dimensions for grouping/coloring
   - Determines optimal visualization strategies
3. **Creative Visualization**:
   - Full creative freedom based on data structure
   - Uses categorical columns for color-coding
   - Leverages temporal data for trend analysis
   - Creates custom matplotlib/seaborn plots
4. **Insight Generation**:
   - Extracts key findings from results
   - Generates task-specific answer
   - Provides recommendations
5. **Report Assembly**:
   - Combines plots, insights, and task answer
   - Saves comprehensive visualization report

**Output**: `VisualizationReport` with plots and insights

**Key Features**:
- **Data-Driven Design**: Visualizations adapt to data structure
- **Professional Quality**: Large figures, legends, clear labels
- **Mandatory Legends**: Every plot includes explanatory legend
- **Fallback Visualizations**: Basic plots if agent fails

**Visualization Requirements**:
```python
# Every plot must include:
- Figure size: 12-18 inches wide
- Title: 14-16pt, bold, descriptive
- Axis labels with units
- Grid lines (alpha=0.3)
- Legend explaining all visual elements
- Thoughtful color schemes
```

**Tools Available**:
- `get_task_context`: Understand original question
- `load_execution_results`: Get prediction data
- `analyze_data_columns`: Examine data structure
- `create_plot`: Generate custom visualizations
- `generate_insights`: Extract key findings
- `generate_task_answer`: Answer the original question
- `save_visualization_report`: Persist all outputs

**Files Created**:
```
output/stage5_out/
  ├── visualization_report_PLAN-TSK-XXX.json
  └── plots/
      ├── TSK-XXX_plot1.png
      ├── TSK-XXX_plot2.png
      └── ...
```

---

### Stage 6: Final Report (`stage6_agent.py`)

**Purpose**: Generate comprehensive final report synthesizing all stages

**Process**:
1. **Data Gathering**:
   - Loads task proposal (original question)
   - Loads execution plan (methodology)
   - Loads execution results (metrics)
   - Loads prediction data (statistics)
   - Loads visualization report (plots and insights)
2. **Analysis**:
   - Reviews all loaded information
   - Extracts actual performance metrics
   - Identifies key findings
3. **Report Generation**:
   - **Executive Summary**: Brief overview and key finding
   - **Methodology**: Data used, models applied, validation approach
   - **Results Analysis**: Detailed results with actual metrics
   - **Conclusions**: Direct answer to original question
   - **Recommendations**: Next steps and improvements
   - **Thought Process**: Stage-by-stage summary of pipeline

**Output**: Comprehensive final report (JSON)

**Key Features**:
- **Evidence-Based**: Uses ONLY actual data from pipeline
- **No Hallucination**: Never invents metrics or statistics
- **Transparency**: Full thought process documentation
- **Actionable**: Includes recommendations for next steps

**Report Sections**:

1. **Executive Summary**
   - 1-2 sentence overview
   - Key finding
   - Most important metric value

2. **Methodology**
   - Data sources
   - Methods/models applied
   - Validation strategy
   - References ONLY actual methods used

3. **Results Analysis**
   - Actual performance metrics from execution
   - Prediction statistics
   - Visualization descriptions
   - Specific numbers from loaded data

4. **Conclusions**
   - Direct answer to original task question
   - Evidence-based conclusions
   - Success assessment
   - Limitations

5. **Recommendations**
   - Next steps
   - Potential improvements
   - Caveats and concerns

6. **Thought Process** (Critical for transparency)
   - Stage 1: Data profiling summary
   - Stage 2: Task proposal rationale
   - Stage 3: Planning decisions
   - Stage 3B: Data preparation approach
   - Stage 3.5A: Method selection rationale
   - Stage 3.5B: Benchmarking results and winner
   - Stage 4: Execution approach
   - Stage 5: Visualization strategy

**Files Created**:
```
output/stage6_out/
  └── TSK-XXX_final_report.json
```

**Critical Rules**:
- Use ONLY actual data from previous stages
- Never make up metrics or statistics
- Reference only visualizations that were created
- Directly answer the original task question
- State missing data explicitly rather than inventing it

---

### Stage 7: Guardrails Validation (`stage7_guardrails_agent.py`)

**Purpose**: Statistical validation to ensure model predictions are valid

**Process**:
1. Loads predictions and actual values
2. **Correlation Analysis**:
   - Checks prediction-actual correlation
   - Validates feature correlations
   - Result: PASS/WARNING/FAIL
3. **Propensity Score Analysis**:
   - Checks for covariate balance
   - Detects selection bias
   - Result: PASS/WARNING/FAIL
4. **Residual Analysis**:
   - Tests residual normality (Shapiro-Wilk)
   - Detects outliers
   - Checks for systematic bias
   - Result: PASS/WARNING/FAIL
5. **Visualization Creation**:
   - Residual distribution plot
   - Actual vs predicted scatter
   - Error by quintile plot
6. **Overall Validity Determination**:
   - All PASS → Valid model
   - Any FAIL → Invalid model (needs investigation)
   - Only WARNINGs → Needs review before deployment

**Output**: Guardrails validation report with overall validity verdict

**Key Features**:
- **Statistical Rigor**: Multiple validation tests
- **Honest Assessment**: Based on actual test results
- **Actionable Recommendations**: Specific guidance for failures
- **Visualization Support**: Charts to understand validation results

**Validation Tests**:

1. **Correlation Analysis**
   ```python
   - Pearson correlation (actual vs predicted)
   - PASS: r > 0.7
   - WARNING: 0.5 ≤ r ≤ 0.7
   - FAIL: r < 0.5
   ```

2. **Propensity Score Analysis**
   ```python
   - Covariate balance check
   - PASS: Balanced covariates
   - WARNING: Minor imbalance
   - FAIL: Significant bias detected
   ```

3. **Residual Analysis**
   ```python
   - Normality test (Shapiro-Wilk, p > 0.05)
   - Outlier detection (IQR method)
   - Mean residual (should be ~0)
   - PASS: Normal distribution, few outliers, low bias
   - WARNING: Mild deviations
   - FAIL: Significant issues
   ```

**Files Created**:
```
output/stage7_guardrails/
  ├── TSK-XXX_guardrails_report.json
  └── plots/
      ├── TSK-XXX_residual_distribution.png
      ├── TSK-XXX_prediction_scatter.png
      └── TSK-XXX_error_by_quintile.png
```

**Overall Validity**:
- **VALID**: All tests PASS, model is statistically sound
- **NEEDS_REVIEW**: Some WARNINGs, review recommended before use
- **INVALID**: Any FAILs, model may not be causally valid

---

## Safety & Validation

The pipeline implements comprehensive safety measures at every level:

### 1. **Input Validation**

**Stage 1 (Data Analysis)**:
- File existence checks
- CSV format validation
- Size limits (max 100MB default)
- Encoding detection

**Stage 2 (Task Proposal)**:
- Dataset availability verification
- Column existence checks
- Feasibility scoring

**Stage 3 (Planning)**:
- Task ID validation
- Required field presence
- Logical consistency checks

### 2. **Execution Safety**

**Code Execution Sandboxing**:
```python
# All stages that execute generated code use:
- Timeout protection (default: 300 seconds)
- Error isolation
- Resource limits
- Safe namespaces
```

**Column Hallucination Prevention**:
```python
# Stage 3.5A, 3.5B, 4:
1. Call get_actual_columns() FIRST
2. Verify all referenced columns exist
3. Use ONLY columns from actual data
4. Raise error if column missing
```

**Data Integrity**:
```python
# DataPassingManager ensures:
- Atomic writes (temp file → rename)
- SHA256 checksums
- Metadata validation
- Schema enforcement
```

### 3. **Output Validation**

**Pydantic Models**:
- Strict type checking
- Required field enforcement
- Value range validation
- Custom validators

**Stage-Specific Validation**:

**Stage 3.5A**:
```python
# Must propose exactly 3 methods
validator = Field(..., min_items=3, max_items=3)
```

**Stage 3.5B**:
```python
# Consistency validation
cv = std / mean  # Coefficient of variation
if cv >= 0.10:
    raise ValidationError("Inconsistent results - possible hallucination")
```

**Stage 4**:
```python
# Metric validation
if abs(test_mae - benchmark_mae) / benchmark_mae > 0.05:
    raise ValidationError("Metrics don't match benchmark")

# NaN/Inf detection
if mae != mae or mae == float('inf'):
    raise ValidationError("Invalid metric value")
```

**Stage 5**:
```python
# Visualization validation
if not visualizations:
    return fallback_visualizations()

# Mandatory legend check
if 'ax.legend()' not in plot_code:
    add_legend(ax)
```

### 4. **Error Handling**

**Retry Logic** (Stages 3.5B, 4):
```python
MAX_RETRIES = 3
RETRY_STAGES = ["stage3_5b", "stage4"]

for attempt in range(1, max_retries + 1):
    try:
        result = run_stage(plan_id)
        return result  # Success
    except Exception as e:
        if is_retryable(e) and attempt < max_retries:
            logger.warning(f"Retry {attempt}/{max_retries}")
            cleanup_partial_outputs()
            continue
        else:
            raise  # Give up
```

**Error Classification**:
- **Retryable**: Token limits, transient LLM failures
- **Non-retryable**: Invalid data, missing files, logic errors

**Error Propagation**:
```python
# Each stage:
try:
    output = run_stage()
    state.mark_stage_completed(stage_name, output)
except Exception as e:
    state.mark_stage_failed(stage_name, str(e))
    # Pipeline continues or stops based on critical flag
```

### 5. **Data Quality Checks**

**Stage 3B (Data Preparation)**:
```python
# After loading data:
- Check for empty DataFrame
- Validate required columns present
- Check for excessive missing values (>50%)
- Verify datetime columns parseable
- Validate numeric columns are numeric
```

**Stage 4 (Execution)**:
```python
# After predictions:
- Check predictions array shape matches actuals
- Validate no NaN predictions (unless expected)
- Verify predictions in reasonable range
- Check prediction_type column present
```

### 6. **Statistical Validation (Stage 7)**

**Comprehensive Testing**:
1. Correlation analysis (prediction quality)
2. Propensity score analysis (selection bias)
3. Residual analysis (error properties)
4. Overall validity determination

**Pass Criteria**:
- Correlation: r > 0.7
- Normality: Shapiro-Wilk p > 0.05
- Bias: Mean residual ≈ 0
- Outliers: < 5% of predictions

---

## Checkpointing & State Management

The pipeline implements multi-level checkpointing for robustness and resumability:

### 1. **LangGraph MemorySaver (Agent-Level)**

**Purpose**: Resume agent execution within a stage if interrupted

**Implementation**:
```python
# Every stage agent uses:
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Invoked with config:
config = {
    "configurable": {"thread_id": f"stage_{plan_id}"},
    "recursion_limit": RECURSION_LIMIT
}
final_state = graph.invoke(initial_state, config)
```

**Behavior**:
- Saves agent state after each node execution
- Automatically resumes from last checkpoint on crash
- Preserves message history and tool call results
- **Limitation**: In-memory only (not persisted across process restarts)

**Use Cases**:
- Recover from LLM API failures
- Resume after tool execution errors
- Continue after timeout recovery

### 2. **Stage Output Caching (Pipeline-Level)**

**Purpose**: Skip completed stages on pipeline restart

**Implementation**:
```python
def load_cached_state() -> Tuple[PipelineState, str]:
    """Determine which stages are complete and where to resume."""
    state = PipelineState()

    # Check each stage output directory
    if stage1_outputs_exist():
        state.mark_stage_completed("stage1", ...)
        resume_from = "stage2"
    if stage2_outputs_exist():
        state.mark_stage_completed("stage2", ...)
        resume_from = "stage3"
    # ... and so on

    return state, resume_from
```

**Files Checked**:
- Stage 1: `summaries/*.summary.json`
- Stage 2: `stage2_out/task_proposals.json`
- Stage 3: `stage3_out/PLAN-{task_id}.json`
- Stage 3B: `stage3b_data_prep/prepared_{plan_id}.parquet`
- Stage 3.5A: `stage3_5a_method_proposal/method_proposal_{plan_id}.json`
- Stage 3.5B: `stage3_5b_benchmarking/tester_{plan_id}.json`
- Stage 4: `stage4_out/execution_result_{plan_id}.json`
- Stage 5: `stage5_out/visualization_report_{plan_id}.json`
- Stage 6: `stage6_out/{task_id}_final_report.json`
- Stage 7: `stage7_guardrails/{task_id}_guardrails_report.json`

**Behavior**:
- If output exists and is valid → skip stage
- If output missing or corrupted → re-run stage
- User can force re-run with `--force` flag

### 3. **Benchmark Checkpoints (Stage 3.5B)**

**Purpose**: Resume benchmarking from last completed method

**Implementation**:
```python
# Stage 3.5B saves checkpoint after each method
checkpoint_data = {
    "plan_id": plan_id,
    "completed_methods": ["M1", "M2"],
    "results": {
        "M1": {...},
        "M2": {...}
    },
    "timestamp": datetime.now()
}
save_checkpoint(checkpoint_data)

# On resume:
checkpoint = load_checkpoint(plan_id)
if checkpoint:
    completed_methods = checkpoint["completed_methods"]
    # Skip M1, M2, continue with M3
```

**Files**:
```
output/stage3_5b_benchmarking/
  └── checkpoint_PLAN-TSK-XXX.json
```

**Behavior**:
- Saves after each of 3 methods completes
- On retry/restart, loads checkpoint
- Skips already-completed methods
- Prevents redundant computation

**Edge Cases**:
- Partial iteration (method started but not finished): Re-run from start
- Checkpoint corruption: Start from beginning
- Method code changed: Invalidate checkpoint

### 4. **Conversation State Persistence**

**Purpose**: Maintain user conversation history across sessions

**Implementation**:
```python
class ConversationHandler:
    def save_session(self):
        """Save conversation context to disk."""
        session_path = CONVERSATION_STATE_DIR / f"{self.session_id}.json"
        DataPassingManager.save_artifact(
            data=self.context.model_dump(),
            output_dir=CONVERSATION_STATE_DIR,
            filename=f"{self.session_id}.json"
        )

    @classmethod
    def load_session(cls, session_id: str):
        """Resume existing conversation."""
        handler = cls(session_id=session_id)
        session_path = CONVERSATION_STATE_DIR / f"{session_id}.json"
        if session_path.exists():
            data = DataPassingManager.load_artifact(session_path)
            handler.context = ConversationContext(**data)
        return handler
```

**Files**:
```
output/conversation_state/
  ├── session_20250101_120000.json
  ├── session_20250101_140000.json
  └── ...
```

**Behavior**:
- Auto-saves after each message exchange
- Persists message history
- Maintains context for multi-turn conversations
- Allows session resumption

### 5. **State Machine Checkpointing**

**Purpose**: Master orchestrator state persistence

**Implementation**:
```python
class PipelineState(BaseModel):
    """Central state object passed between stages."""
    selected_task_id: Optional[str] = None
    stages: Dict[str, StageState] = {}  # Per-stage status
    stage1_output: Optional[Stage1Output] = None
    stage2_output: Optional[TaskProposalOutput] = None
    # ... outputs for each stage

    def mark_stage_started(self, stage_name: str):
        self.stages[stage_name] = StageState(
            status=StageStatus.IN_PROGRESS,
            started_at=datetime.now()
        )

    def mark_stage_completed(self, stage_name: str, output: Any):
        self.stages[stage_name].status = StageStatus.COMPLETED
        self.stages[stage_name].completed_at = datetime.now()
        self.stages[stage_name].output = output
```

**Behavior**:
- Tracks status of all stages
- Stores outputs for inter-stage dependencies
- Enables conditional routing (skip failed dependencies)
- Provides status reporting

### 6. **Resumability Workflow**

**Example: Full Pipeline Restart After Stage 3.5B Failure**

```python
# Initial run
$ python run_conversational.py --mode run --task TSK-001

# Stages complete:
# ✓ Stage 3 (planning)
# ✓ Stage 3B (data prep)
# ✓ Stage 3.5A (method proposal)
# ✗ Stage 3.5B (benchmarking) - FAILED after Method 1

# On restart:
$ python run_conversational.py --mode run --task TSK-001

# Pipeline detects:
1. Stage 3 output exists → SKIP
2. Stage 3B output exists → SKIP
3. Stage 3.5A output exists → SKIP
4. Stage 3.5B checkpoint exists (M1 complete) → RESUME
   - Load checkpoint
   - Skip Method 1
   - Continue with Methods 2, 3
5. Stage 4 not started → RUN
6. Stage 5 not started → RUN
```

**Benefits**:
- **Time Savings**: Don't re-run expensive stages
- **Fault Tolerance**: Recover from crashes
- **Development Speed**: Quick iterations during debugging
- **Cost Efficiency**: Fewer LLM API calls

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- LLM API endpoint (OpenAI-compatible)
- 8GB+ RAM recommended
- 1GB+ disk space for outputs

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Capstone_Project/conversational

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

#### 1. LLM Endpoint Setup

Set environment variables for your LLM provider:

```bash
# For local LLM (e.g., LM Studio, Ollama)
export LLM_BASE_URL="http://localhost:8001/v1"
export LLM_API_KEY="EMPTY"

# For OpenAI
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."

# For custom endpoint
export LLM_BASE_URL="https://your-endpoint.com/v1"
export LLM_API_KEY="your-api-key"
```

**Dual LLM Configuration**:
- **PRIMARY_LLM**: Complex reasoning (Qwen/Qwen2.5-32B-Instruct)
- **SECONDARY_LLM**: Tool-calling agents (Qwen/Qwen3-32B)

Edit `code/config.py` to change models:
```python
PRIMARY_LLM_CONFIG = {
    "base_url": LLM_BASE_URL,
    "api_key": LLM_API_KEY,
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "temperature": 0.1,
    "max_tokens": 8192,
}
```

#### 2. Directory Structure Setup

```bash
# Data directory (place your CSV files here)
mkdir -p data

# Output directory (created automatically)
# conversational/output/
#   ├── summaries/
#   ├── stage2_out/
#   ├── stage3_out/
#   ├── stage3b_data_prep/
#   ├── stage3_5a_method_proposal/
#   ├── stage3_5b_benchmarking/
#   ├── stage4_out/
#   ├── stage5_out/
#   ├── stage6_out/
#   ├── stage7_guardrails/
#   └── conversation_state/
```

#### 3. Optional: LangSmith Tracing

For debugging and monitoring:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="conversational-pipeline"
```

### Verification

```bash
# Check configuration
python run_conversational.py --config

# Should show:
# - Project root
# - Data directory
# - LLM configuration
# - Stage parameters
```

---

## Usage

### Quick Start

```bash
# 1. Place CSV files in data/ directory
cp your_data.csv conversational/data/

# 2. Run interactive mode
python run_conversational.py

# 3. Follow prompts:
#    - Data will be auto-analyzed
#    - Tasks will be proposed
#    - Ask questions or request forecasting
```

### Command-Line Interface

#### 1. **Analyze Data** (Stage 1)

```bash
python run_conversational.py --mode analyze
```

Output:
```
Analyzed 2 datasets:
  agriculture_data.csv:
    Rows: 1000, Columns: 15
    Quality: 95.0%

Datetime columns (suitable for forecasting):
  - agriculture_data.csv: Year

Potential target columns:
  - agriculture_data.csv: Production
  - agriculture_data.csv: Area
```

#### 2. **Generate Task Proposals** (Stage 2)

```bash
# General proposals
python run_conversational.py --mode propose

# Custom query
python run_conversational.py --mode propose --query "Can I forecast rice production?"
```

Output:
```
Generated 3 proposals:

  TSK-001: Forecast Annual Rice Production
    Category: forecasting
    Target: Production
    Feasibility: 0.92
    Description: Predict rice production for next 5 years...
```

#### 3. **Run Forecasting Pipeline** (Stages 3→3B→3.5A→3.5B→4→5)

```bash
python run_conversational.py --mode run --task TSK-001
```

Progress:
```
Running forecasting pipeline: 3 → 3B → 3.5A → 3.5B → 4 → 5

[Stage 3] Creating execution plan... ✓
[Stage 3B] Preparing data... ✓
[Stage 3.5A] Proposing 3 methods... ✓
[Stage 3.5B] Benchmarking methods (3 iterations each)... ✓
  Method 1: MAE=245.67 (CV=2.3%)
  Method 2: MAE=198.43 (CV=1.8%) ← WINNER
  Method 3: MAE=312.89 (CV=4.1%)
[Stage 4] Executing winner... ✓
[Stage 5] Creating visualizations... ✓

Completed stages: [stage3, stage3b, stage3_5a, stage3_5b, stage4, stage5]
Metrics: {"MAE": 198.43, "RMSE": 256.71, "R2": 0.89}
```

#### 4. **Run Full Pipeline** (Stages 1→2→3→...→7)

```bash
python run_conversational.py --mode full --task TSK-001
```

Includes all stages plus:
- Stage 6: Final report generation
- Stage 7: Guardrails validation

#### 5. **Run Specific Stages**

```bash
# Run only data prep and execution
python run_conversational.py --mode run --task TSK-001 --stages "stage3b,stage4"

# Re-run just visualization
python run_conversational.py --mode run --task TSK-001 --stages "stage5"
```

#### 6. **Check Status**

```bash
python run_conversational.py --status
```

Output:
```
Pipeline Status:

Completed Stages:
  - stage1 (2 datasets analyzed)
  - stage2 (3 tasks proposed)

Available Tasks:
  - TSK-001: Forecast Annual Rice Production
  - TSK-002: Predict Crop Area Changes
  - TSK-003: Analyze Seasonal Patterns
```

### Conversational Interface

```bash
python run_conversational.py
```

**Example Session**:

```
You: What data is available?

Agent: I found 2 datasets in the data/ directory:
  1. agriculture_data.csv (1000 rows, 15 columns)
  2. sales_data.csv (5000 rows, 8 columns)

Both have been analyzed. Would you like to explore either dataset?

You: Can we forecast rice production?

Agent: Yes! I found a suitable task:
  TSK-001: Forecast Annual Rice Production
  Dataset: agriculture_data.csv
  Target: Production column
  Feasibility: 92%

Would you like me to run the forecasting pipeline for this task?

You: Yes, please run it.

Agent: Starting forecasting pipeline...
[Pipeline executes Stages 3→3B→3.5A→3.5B→4→5]
...
Pipeline complete! Results:
  - Best method: ARIMA model
  - Test MAE: 198.43
  - R² Score: 0.89
  - Future forecasts: 5 years generated

You can find visualizations in output/stage5_out/plots/

You: Show me the results summary

Agent: [Loads and displays execution results and visualizations]
```

**Available Commands**:
- "What data is available?"
- "Analyze [dataset_name]"
- "Can we forecast [target]?"
- "Run task [TSK-XXX]"
- "Show results for [TSK-XXX]"
- "List all tasks"
- "What's the pipeline status?"

### Web Interface

```bash
# Start web server
python server.py

# Open browser to http://localhost:8000
```

**Features**:
- Interactive chat interface
- Real-time pipeline execution
- Visualization display
- Task management
- Download results

---

## Configuration

### Environment Variables

```bash
# LLM Configuration (Required)
export LLM_BASE_URL="http://localhost:8001/v1"
export LLM_API_KEY="EMPTY"

# Project Root (Optional - defaults to conversational/)
export PROJECT_ROOT="/path/to/conversational"

# LangSmith Tracing (Optional)
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="conversational-pipeline"

# Debug Mode (Optional)
export DEBUG=true
```

### config.py Settings

Edit `code/config.py` to customize pipeline behavior:

#### LLM Configuration

```python
# Dual LLM setup
PRIMARY_LLM_CONFIG = {
    "base_url": LLM_BASE_URL,
    "api_key": LLM_API_KEY,
    "model": "Qwen/Qwen2.5-32B-Instruct",  # Complex reasoning
    "temperature": 0.1,
    "max_tokens": 8192,
}

SECONDARY_LLM_CONFIG = {
    "base_url": LLM_BASE_URL,
    "api_key": LLM_API_KEY,
    "model": "Qwen/Qwen3-32B",  # Tool-calling agents
    "temperature": 0.0,
    "max_tokens": 8192,
}
```

#### Stage Parameters

```python
# Max reasoning rounds per stage
STAGE_MAX_ROUNDS = {
    "stage2": 20,    # Task proposal
    "stage3": 50,    # Planning (extended for complex scenarios)
    "stage3_5a": 20, # Method proposal
    "stage3_5b": 15, # Benchmarking
    "stage4": 30,    # Execution
    "stage5": 25,    # Visualization
}

# Retry configuration
MAX_RETRIES = 3
RETRY_STAGES = ["stage3_5b", "stage4"]

# Benchmarking
BENCHMARK_ITERATIONS = 3  # Iterations per method
CONSISTENCY_THRESHOLD = 0.10  # Max CV for valid results
```

#### Directory Structure

```python
# All paths relative to PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
STAGE2_OUT_DIR = OUTPUT_DIR / "stage2_out"
STAGE3_OUT_DIR = OUTPUT_DIR / "stage3_out"
# ... etc
```

#### Validation Thresholds

```python
# Stage 4 metric validation (vs Stage 3.5B)
METRIC_TOLERANCE = 0.05  # ±5%

# Stage 7 guardrails
CORRELATION_THRESHOLD = 0.7  # Min correlation for PASS
NORMALITY_PVALUE = 0.05     # Shapiro-Wilk significance
MAX_OUTLIER_PERCENTAGE = 0.05  # Max 5% outliers
```

#### Dynamic Metrics

```python
def get_task_appropriate_metrics(task_category: str, description: str) -> List[str]:
    """Select metrics based on task type."""
    if task_category == "forecasting":
        return ["mae", "rmse", "mape", "r2"]
    elif task_category == "classification":
        return ["accuracy", "precision", "recall", "f1"]
    else:
        # Infer from description
        return infer_metrics(description)
```

### Data Passing Configuration

```python
# DataPassingManager settings
class DataPassingManager:
    CHECKSUM_ENABLED = True  # SHA256 validation
    ATOMIC_WRITES = True     # Temp file → rename
    METADATA_ENVELOPE = True  # Wrap artifacts with _meta
```

---

## Development Guide

### Adding a New Stage

**Step 1: Define Output Model** (`models.py`)

```python
class StageXOutput(BaseModel):
    """Output for Stage X."""
    field1: str
    field2: int
    result_data: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True
```

**Step 2: Create Stage Agent** (`code/stageX_agent.py`)

```python
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

def stageX_node(state: PipelineState) -> PipelineState:
    """LangGraph node for Stage X."""
    logger.info("Running Stage X")

    # Load dependencies from state
    prev_output = state.stageY_output

    # Run stage logic
    output = run_stageX(state.selected_task_id, prev_output)

    # Save to disk
    DataPassingManager.save_artifact(
        data=output.model_dump(),
        output_dir=STAGEX_OUT_DIR,
        filename=f"stageX_result_{state.selected_task_id}.json"
    )

    # Update state
    state.stageX_output = output
    state.mark_stage_completed("stageX", output)

    return state

def run_stageX(task_id: str, dependency: StageYOutput) -> StageXOutput:
    """Main execution logic for Stage X."""
    # Implementation
    return StageXOutput(...)
```

**Step 3: Add Tools** (`tools/stageX_tools.py`)

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description for LLM."""
    # Implementation
    return result

STAGEX_TOOLS = [my_tool, another_tool]
```

**Step 4: Update Master Orchestrator** (`master_orchestrator.py`)

```python
# Add to STAGE_ORDER
STAGE_ORDER = [
    "stage1", "stage2", "stage3", "stage3b",
    "stage3_5a", "stage3_5b", "stageX", "stage4", "stage5"
]

# Add node
STAGE_NODES = {
    # ... existing
    "stageX": stageX_node,
}

# Update routing logic if needed
def route_after_stageX(state: PipelineState) -> str:
    if state.stageX_output.should_continue:
        return "stage4"
    else:
        return END
```

**Step 5: Update Config** (`config.py`)

```python
# Add output directory
STAGEX_OUT_DIR = OUTPUT_DIR / "stageX_out"

# Add to stage contracts
StageTransition.STAGE_CONTRACTS = {
    # ... existing
    "stageX": {
        "inputs": ["stageY_output"],
        "outputs": ["stageX_result_{task_id}.json"],
    }
}
```

### Creating Custom Tools

**Pattern 1: Simple Function Tool**

```python
@tool
def simple_tool(param: str) -> str:
    """Description visible to LLM."""
    result = process(param)
    return result
```

**Pattern 2: Stateful Tool with Context**

```python
class ToolContext:
    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self.data = load_prepared_data(plan_id)

def create_tools(context: ToolContext) -> List[Tool]:
    @tool
    def analyze_data() -> Dict:
        """Analyze the prepared data."""
        return {
            "rows": len(context.data),
            "columns": list(context.data.columns)
        }

    return [analyze_data]
```

**Pattern 3: Error-Handling Tool**

```python
@tool
def risky_tool(param: str) -> str:
    """Tool that may fail."""
    try:
        result = dangerous_operation(param)
        return f"Success: {result}"
    except Exception as e:
        logger.error(f"Tool failed: {e}")
        return f"Error: {str(e)}"
```

### Extending Data Models

**Adding Fields to Existing Models**:

```python
# models.py
class TaskProposal(BaseModel):
    # Existing fields
    task_id: str
    title: str

    # New fields
    priority: int = Field(default=1, ge=1, le=5)
    tags: List[str] = Field(default_factory=list)
```

**Custom Validators**:

```python
from pydantic import validator

class MyModel(BaseModel):
    value: float

    @validator("value")
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
```

### Testing

**Unit Testing a Stage**:

```python
# test_stageX.py
import pytest
from code.stageX_agent import run_stageX
from code.models import StageYOutput, StageXOutput

def test_stageX_basic():
    # Prepare input
    input_data = StageYOutput(
        field1="test",
        field2=42
    )

    # Run stage
    output = run_stageX("TSK-999", input_data)

    # Assertions
    assert isinstance(output, StageXOutput)
    assert output.field1 is not None
```

**Integration Testing**:

```bash
# Test full pipeline
python run_conversational.py --mode run --task TSK-001

# Check outputs exist
ls output/stage3_out/PLAN-TSK-001.json
ls output/stage4_out/execution_result_PLAN-TSK-001.json
```

### Debugging

**Enable Verbose Logging**:

```python
# config.py or utils.py
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**LangSmith Tracing**:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="debug-session"

python run_conversational.py --mode run --task TSK-001
```

**Inspecting Checkpoints**:

```python
# Load LangGraph checkpoint
from langgraph.checkpoint import MemorySaver

memory = MemorySaver()
# Checkpoints accessible during execution
# (In-memory only, not persisted)
```

**Manual Stage Execution**:

```python
# Run single stage directly
from code.stage4_agent import run_stage4

output = run_stage4("PLAN-TSK-001")
print(output.model_dump_json(indent=2))
```

---

## API Reference

### Core Classes

#### PipelineState

```python
class PipelineState(BaseModel):
    """Central state object for pipeline orchestration."""

    # Task selection
    selected_task_id: Optional[str] = None

    # Stage outputs
    stage1_output: Optional[Stage1Output] = None
    stage2_output: Optional[TaskProposalOutput] = None
    stage3_output: Optional[ExecutionPlan] = None
    stage3b_output: Optional[PreparedData] = None
    stage3_5a_output: Optional[MethodProposal] = None
    stage3_5b_output: Optional[TesterOutput] = None
    stage4_output: Optional[ExecutionResult] = None
    stage5_output: Optional[VisualizationReport] = None
    stage6_output: Optional[Dict] = None
    stage7_output: Optional[Dict] = None

    # Stage tracking
    stages: Dict[str, StageState] = {}

    def mark_stage_started(self, stage_name: str) -> None:
        """Mark stage as in progress."""

    def mark_stage_completed(self, stage_name: str, output: Any) -> None:
        """Mark stage as completed with output."""

    def mark_stage_failed(self, stage_name: str, error: str) -> None:
        """Mark stage as failed with error message."""
```

#### DataPassingManager

```python
class DataPassingManager:
    """Robust inter-stage data transfer."""

    @staticmethod
    def save_artifact(
        data: Dict,
        output_dir: Path,
        filename: str
    ) -> Path:
        """Save JSON artifact with metadata envelope and checksum."""

    @staticmethod
    def load_artifact(filepath: Path) -> Dict:
        """Load and validate artifact."""

    @staticmethod
    def save_dataframe(
        df: pd.DataFrame,
        output_dir: Path,
        base_filename: str
    ) -> Tuple[Path, Path]:
        """Save DataFrame as Parquet with sidecar metadata."""

    @staticmethod
    def load_dataframe(parquet_path: Path) -> pd.DataFrame:
        """Load and validate DataFrame."""
```

### Key Functions

#### run_conversational.py

```python
def main(
    mode: str = "interactive",
    task_id: Optional[str] = None,
    query: Optional[str] = None,
    stages: Optional[str] = None
) -> None:
    """Main entry point for pipeline execution."""
```

#### master_orchestrator.py

```python
def build_graph() -> CompiledGraph:
    """Build LangGraph state machine."""

def run_pipeline(
    task_id: str,
    stages_to_run: Optional[List[str]] = None
) -> PipelineState:
    """Execute pipeline for a task."""

def load_cached_state() -> Tuple[PipelineState, str]:
    """Load existing outputs and determine resume point."""
```

#### Stage Agent Functions

```python
# Pattern for all stages
def stageX_node(state: PipelineState) -> PipelineState:
    """LangGraph node wrapper."""

def run_stageX(...) -> StageXOutput:
    """Direct execution function."""
```

### Tool Functions

#### Stage 3.5B Tools

```python
@tool
def run_benchmark_code(
    method_name: str,
    code: str,
    data_split_strategy: str,
    iteration: int
) -> Dict:
    """Execute benchmarking code and return metrics."""

@tool
def save_model_checkpoint(
    method_name: str,
    model_object: Any
) -> str:
    """Save trained model for later use."""

@tool
def get_actual_columns() -> List[str]:
    """Get actual column names from prepared data."""
```

#### Stage 4 Tools

```python
@tool
def load_model_checkpoint(method_name: str) -> Any:
    """Load saved model from Stage 3.5B."""

@tool
def retrain_model(code: str, strategy: str) -> Any:
    """Retrain model if no checkpoint available."""

@tool
def generate_future_forecasts(
    model: Any,
    data: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """Generate future predictions."""
```

### Utility Functions

#### utils.py

```python
def call_llm(
    messages: List[Dict],
    config: Dict = PRIMARY_LLM_CONFIG,
    json_mode: bool = False
) -> str:
    """Call LLM with messages and get response."""

def validate_columns(
    df: pd.DataFrame,
    required_columns: List[str]
) -> bool:
    """Validate DataFrame has required columns."""

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: List[str]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
```

---

## Troubleshooting

### Common Issues

#### 1. LLM Connection Errors

**Symptom**: `Connection refused` or `API key invalid`

**Solutions**:
```bash
# Check LLM server is running
curl http://localhost:8001/v1/models

# Verify environment variables
echo $LLM_BASE_URL
echo $LLM_API_KEY

# Test with simple request
python -c "from openai import OpenAI; client = OpenAI(base_url='$LLM_BASE_URL', api_key='$LLM_API_KEY'); print(client.models.list())"
```

#### 2. Stage Fails with "Column not found"

**Symptom**: `KeyError: 'column_name'` during Stage 3.5B or 4

**Cause**: Agent hallucinated a column name

**Solution**:
- Stage automatically retries (up to 3 times)
- System prompt forces `get_actual_columns()` tool call
- If persistent, check data preparation in Stage 3B

**Manual Fix**:
```python
# Check actual columns
import pandas as pd
df = pd.read_parquet("output/stage3b_data_prep/prepared_PLAN-TSK-001.parquet")
print(df.columns.tolist())
```

#### 3. Benchmark Results Inconsistent (CV > 10%)

**Symptom**: Stage 3.5B fails with "Coefficient of variation too high"

**Cause**: Method produces non-deterministic results

**Solutions**:
- Check if method sets random seed
- Increase `BENCHMARK_ITERATIONS` in config.py
- Review method implementation in Stage 3.5A output

**Example Fix**:
```python
# In method code, ensure:
np.random.seed(42)
# or
model = RandomForestRegressor(random_state=42)
```

#### 4. Stage 4 Metrics Don't Match Stage 3.5B

**Symptom**: `ValidationError: Test MAE doesn't match benchmark`

**Cause**: Model checkpoint missing or data split differs

**Solutions**:
1. Check checkpoint exists:
   ```bash
   ls output/stage4_out/workspace/*_model.pkl
   ```

2. Verify Stage 3.5B completed successfully:
   ```bash
   cat output/stage3_5b_benchmarking/tester_PLAN-TSK-001.json | grep selected_method
   ```

3. Force retrain instead of checkpoint:
   ```python
   # In stage4_agent.py, comment out checkpoint loading
   # model = load_model_checkpoint(...)
   ```

#### 5. Out of Memory Errors

**Symptom**: `MemoryError` or process killed

**Cause**: Large dataset or memory leak

**Solutions**:
```python
# Reduce data size in Stage 3B
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data

# Use chunking for large datasets
for chunk in pd.read_csv(file, chunksize=10000):
    process(chunk)

# Clear memory between stages
import gc
gc.collect()
```

#### 6. LangGraph Recursion Limit

**Symptom**: `RecursionError: maximum recursion depth exceeded`

**Cause**: Agent stuck in loop or task too complex

**Solutions**:
```python
# Increase limit in config.py
RECURSION_LIMIT = 300  # Default: 200

# Or reduce STAGE_MAX_ROUNDS
STAGE_MAX_ROUNDS = {"stage3": 30}  # Default: 50
```

#### 7. Visualization Fails (Stage 5)

**Symptom**: No plots generated or errors in matplotlib

**Solutions**:
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Use non-interactive backend
export MPLBACKEND=Agg

# Verify output directory writable
ls -ld output/stage5_out/plots/
```

**Fallback**: Stage 5 includes automatic fallback visualizations if agent fails

#### 8. Task Proposals Not Generated (Stage 2)

**Symptom**: Empty task list or generic proposals

**Cause**: Data summaries incomplete or LLM issues

**Solutions**:
1. Check Stage 1 outputs:
   ```bash
   ls output/summaries/
   cat output/summaries/*.summary.json
   ```

2. Verify datetime columns detected:
   ```bash
   grep "datetime_columns" output/summaries/*.summary.json
   ```

3. Re-run Stage 1 with force:
   ```bash
   rm -rf output/summaries/
   python run_conversational.py --mode analyze
   ```

#### 9. Parquet File Corruption

**Symptom**: `ArrowInvalid` or `ParquetFileError`

**Solutions**:
```bash
# Remove corrupted file
rm output/stage3b_data_prep/prepared_PLAN-TSK-001.parquet

# Re-run Stage 3B
python run_conversational.py --mode run --task TSK-001 --stages "stage3b"

# Verify integrity
python -c "import pandas as pd; df = pd.read_parquet('output/stage3b_data_prep/prepared_PLAN-TSK-001.parquet'); print(len(df))"
```

#### 10. Guardrails Validation Fails (Stage 7)

**Symptom**: "INVALID" verdict with residual analysis failure

**Interpretation**:
- **VALID**: Model is statistically sound, safe to use
- **NEEDS_REVIEW**: Some warnings, review recommended
- **INVALID**: Significant issues, investigate before use

**Actions**:
```bash
# Review validation report
cat output/stage7_guardrails/TSK-001_guardrails_report.json

# Check visualizations
open output/stage7_guardrails/plots/TSK-001_residual_distribution.png

# Common issues:
# - Residuals not normal: Model may be biased
# - Low correlation: Model has poor predictive power
# - High outlier percentage: Data quality issues
```

### Debug Checklist

When a stage fails:

1. **Check Logs**:
   ```bash
   # If using systemd or docker
   journalctl -u pipeline-service -n 100

   # Or console output
   python run_conversational.py --mode run --task TSK-001 2>&1 | tee debug.log
   ```

2. **Verify Inputs**:
   ```bash
   # Check previous stage completed
   ls output/stageX_out/

   # Validate JSON structure
   python -m json.tool output/stage3_out/PLAN-TSK-001.json
   ```

3. **Inspect State**:
   ```python
   # Load state manually
   from code.master_orchestrator import load_cached_state
   state, resume_from = load_cached_state()
   print(state.model_dump_json(indent=2))
   ```

4. **Test LLM**:
   ```python
   from code.utils import call_llm
   response = call_llm([{"role": "user", "content": "Hello"}])
   print(response)
   ```

5. **Validate Data**:
   ```python
   import pandas as pd
   df = pd.read_parquet("output/stage3b_data_prep/prepared_PLAN-TSK-001.parquet")
   print(df.info())
   print(df.describe())
   ```

6. **Check Permissions**:
   ```bash
   ls -la output/
   # Should be writable by current user
   ```

7. **Free Disk Space**:
   ```bash
   df -h
   # Ensure sufficient space (at least 1GB free)
   ```

### Performance Optimization

**Slow Stage 3.5B (Benchmarking)**:
```python
# Reduce iterations
BENCHMARK_ITERATIONS = 2  # Default: 3

# Use smaller data sample
# In Stage 3B or method code
df_sample = df.sample(n=1000, random_state=42)
```

**Slow LLM Response**:
```python
# Reduce max_tokens
PRIMARY_LLM_CONFIG["max_tokens"] = 4096  # Default: 8192

# Lower temperature for faster sampling
PRIMARY_LLM_CONFIG["temperature"] = 0.0  # Default: 0.1
```

**Disk I/O Bottleneck**:
```python
# Use faster Parquet compression
import pandas as pd
df.to_parquet(path, compression='snappy')  # Default: 'gzip'
```

### Getting Help

1. **Check Documentation**:
   - [README.md](README.md) - User guide
   - [detailed.md](detailed.md) - Technical deep-dive
   - [CLAUDE.md](../CLAUDE.md) - Project overview

2. **Review Examples**:
   - Example outputs in `output/` directory
   - Sample data in `data/` directory

3. **Enable Tracing**:
   ```bash
   export LANGSMITH_TRACING=true
   # View traces at smith.langchain.com
   ```

4. **File an Issue**:
   - Include error message
   - Attach relevant logs
   - Describe steps to reproduce
   - Specify LLM configuration

---

## License

[Your License Here]

## Acknowledgments

- **LangGraph**: Workflow orchestration framework
- **LangChain**: LLM integration and tooling
- **Pydantic**: Data validation and serialization
- **Pandas/PyArrow**: Data processing and storage

---

**Version**: 1.0
**Last Updated**: 2025-12-19
**Maintained By**: [Your Name/Team]