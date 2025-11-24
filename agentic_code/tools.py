"""
Centralized tool definitions for all stages of the agentic AI pipeline.

Tools are organized by stage and can be imported individually or as groups.
"""

from __future__ import annotations

import json
import io
import contextlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from langchain_core.tools import tool

from .config import (
    DATA_DIR, SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR,
    STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE4_WORKSPACE, STAGE5_WORKSPACE
)
from .models import Stage2Output, Stage3Plan, ExecutionResult
from .utils import (
    list_summary_files as _list_summary_files,
    read_summary_file as _read_summary_file,
    list_data_files as _list_data_files,
    inspect_data_file as _inspect_data_file,
    load_dataframe,
)

# ===========================
# Stage 2: Task Proposal Tools
# ===========================

@tool
def list_summary_files() -> List[str]:
    """List all dataset summary JSON files available from Stage 1.
    
    Returns filenames (not full paths).
    """
    return _list_summary_files()


@tool
def read_summary_file(filename: str) -> str:
    """Read a single dataset summary JSON file and return its contents as a string."""
    return _read_summary_file(filename)


@tool
def python_sandbox(code: str) -> str:
    """Execute arbitrary Python code to help analyze dataset summaries and design tasks.
    
    The code can:
    - import standard libraries like json, math, statistics, pandas
    - access PROJECT_ROOT, DATA_DIR, SUMMARIES_DIR
    - call read_summary_file('<summary-filename>')
    - call list_summary_files() (sandbox helper)
    - open and inspect files directly
    - print intermediate results
    
    Returns whatever is printed to stdout (or an error string).
    """
    def _read_summary_file_py(filename: str) -> str:
        """Helper for sandbox: read a summary file as text."""
        path = SUMMARIES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No such summary file: {filename}")
        return path.read_text()
    
    def _list_summary_files_py() -> List[str]:
        return [p.name for p in SUMMARIES_DIR.glob("*.summary.json")]
    
    PYTHON_GLOBALS: Dict[str, Any] = {
        "__name__": "__agent_python__",
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "SUMMARIES_DIR": SUMMARIES_DIR,
        "read_summary_file": _read_summary_file_py,
        "list_summary_files": _list_summary_files_py,
    }
    
    local_env: Dict[str, Any] = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, PYTHON_GLOBALS, local_env)
    except Exception as e:
        return f"[python_sandbox error] {e}"
    return buf.getvalue() or "[python_sandbox done]"


# Stage 2 tool list
STAGE2_TOOLS = [list_summary_files, read_summary_file, python_sandbox]


# ===========================
# Stage 3: Planning Tools
# ===========================

@tool
def load_task_proposal(task_id: str) -> str:
    """Load a single TaskProposal by ID."""
    path = STAGE2_OUT_DIR / "task_proposals.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find task_proposals.json in {STAGE2_OUT_DIR}")
    
    raw = path.read_text()
    data = json.loads(raw)
    stage2 = Stage2Output.model_validate(data)
    for p in stage2.proposals:
        if p.id == task_id:
            return p.model_dump_json(indent=2)
    raise ValueError(f"No TaskProposal with id={task_id!r} found.")


@tool
def list_data_files() -> List[str]:
    """List available data files."""
    return _list_data_files()


@tool
def inspect_data_file(filename: str, n_rows: int = 10) -> str:
    """Inspect a data file - shows head, dtypes, nulls."""
    return _inspect_data_file(filename, n_rows)


@tool
def python_sandbox_stage3(code: str) -> str:
    """Execute Python code for data exploration in Stage 3."""
    def load_dataframe_helper(filename: str, n_rows: Optional[int] = None):
        return load_dataframe(filename, nrows=n_rows, base_dir=DATA_DIR)
    
    globals_dict = {
        "__name__": "__stage3_sandbox__",
        "pd": pd,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "load_dataframe": load_dataframe_helper,
    }
    
    local_env = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globals_dict, local_env)
    except Exception as e:
        return f"[ERROR] {e}"
    return buf.getvalue() or "[No output]"


@tool
def save_stage3_plan(plan_json: str) -> str:
    """Validate and save a Stage3Plan.
    
    Args:
        plan_json: Complete JSON string of Stage3Plan
        
    Returns:
        Success message with path, or raises ValueError
    """
    # Parse JSON
    try:
        raw_obj = json.loads(plan_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    # Schema validation
    try:
        plan = Stage3Plan.model_validate(raw_obj)
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}") from e

    # Validate files exist
    available_files = set(_list_data_files())
    for fi in plan.file_instructions:
        if fi.original_name not in available_files:
            raise ValueError(f"File {fi.original_name!r} not found in DATA_DIR")

    # For join steps, do basic validation
    if plan.join_steps:
        file_cache = {}
        
        # Load files
        for fi in plan.file_instructions:
            try:
                df = load_dataframe(fi.original_name, nrows=100, base_dir=DATA_DIR)
                file_cache[fi.alias] = df
            except Exception as e:
                raise ValueError(f"Failed to load {fi.original_name}: {e}") from e
        
        # Validate joins
        for idx, js in enumerate(plan.join_steps):
            if js.join_type == "base":
                if js.left_table not in file_cache:
                    raise ValueError(f"Base table {js.left_table!r} not found")
                continue
                
            if js.left_table not in file_cache:
                raise ValueError(f"Left table {js.left_table!r} not found")
            if js.right_table and js.right_table not in file_cache:
                raise ValueError(f"Right table {js.right_table!r} not found")
            
            # Check keys exist
            df_left = file_cache[js.left_table]
            df_right = file_cache.get(js.right_table) if js.right_table else None
            
            missing_left = [k for k in js.join_keys if k not in df_left.columns]
            if missing_left:
                raise ValueError(f"Join {idx}: keys {missing_left} missing in {js.left_table}")
            
            if df_right is not None:
                missing_right = [k for k in js.join_keys if k not in df_right.columns]
                if missing_right:
                    raise ValueError(f"Join {idx}: keys {missing_right} missing in {js.right_table}")

    # Save
    out_path = STAGE3_OUT_DIR / f"{plan.plan_id}.json"
    out_path.write_text(plan.model_dump_json(indent=2))
    
    return f"✅ Plan saved successfully to: {out_path}"


# Stage 3 tool list
STAGE3_TOOLS = [
    load_task_proposal,
    list_data_files,
    inspect_data_file,
    python_sandbox_stage3,
    save_stage3_plan,
]


# ===========================
# Stage 4: Execution Tools
# ===========================

@tool
def list_stage3_plans() -> List[str]:
    """List all available Stage 3 plans."""
    plans = sorted([p.name for p in STAGE3_OUT_DIR.glob("*.json")])
    return plans


@tool
def load_stage3_plan(plan_id: str) -> str:
    """Load a Stage 3 plan by ID.
    
    Args:
        plan_id: Plan identifier (e.g., 'PLAN-TSK-002')
        
    Returns:
        JSON string of the plan
    """
    # Try exact match first
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        # Try finding by pattern
        matches = list(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No plan found matching: {plan_id}")
        plan_path = matches[0]
    
    return plan_path.read_text()


@tool
def execute_python_code(code: str, description: str = "Executing code") -> str:
    """Execute arbitrary Python code for data processing, modeling, and analysis.
    
    This is the primary tool for implementing the Stage 3 plan.
    You can:
    - Load and transform data
    - Perform joins and aggregations
    - Build and train models
    - Generate predictions
    - Calculate metrics
    - Save intermediate and final results
    
    Available in the execution environment:
    - pandas as pd
    - numpy as np
    - sklearn (all modules)
    - json, pathlib.Path
    - DATA_DIR, STAGE4_OUT_DIR, STAGE4_WORKSPACE
    - Helper: load_dataframe(filename) for loading files from DATA_DIR
    
    Args:
        code: Python code to execute
        description: Brief description of what this code does
        
    Returns:
        Output printed to stdout, or error message
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    def load_dataframe_helper(filename: str):
        """Load a dataframe from DATA_DIR."""
        return load_dataframe(filename, base_dir=DATA_DIR)
    
    globals_dict = {
        "__name__": "__stage4_executor__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "STAGE4_OUT_DIR": STAGE4_OUT_DIR,
        "STAGE4_WORKSPACE": STAGE4_WORKSPACE,
        "load_dataframe": load_dataframe_helper,
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "train_test_split": train_test_split,
        "StandardScaler": StandardScaler,
    }
    
    local_env = {}
    buf = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== {description} ===")
            exec(code, globals_dict, local_env)
    except Exception as e:
        return f"[ERROR] {e}\n\nTraceback: {e.__class__.__name__}"
    
    output = buf.getvalue()
    return output if output else "[Code executed successfully, no output]"


@tool
def save_execution_result(result_json: str) -> str:
    """Save the final execution result.
    
    Args:
        result_json: JSON string containing:
            - plan_id: ID of the executed plan
            - task_category: descriptive/predictive/unsupervised
            - status: success/failure/partial
            - outputs: dict with output file paths
            - metrics: dict with performance metrics (if applicable)
            - summary: text summary of results
            - errors: list of any errors encountered
    
    Returns:
        Confirmation message with save path
    """
    from datetime import datetime
    
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    plan_id = result.get("plan_id", "UNKNOWN")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = STAGE4_OUT_DIR / f"execution_{plan_id}_{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2))
    
    return f"✅ Execution result saved to: {output_path}"


# Stage 4 tool list
STAGE4_TOOLS = [
    list_stage3_plans,
    load_stage3_plan,
    list_data_files,
    execute_python_code,
    save_execution_result,
]


# ===========================
# Stage 5: Visualization Tools
# ===========================

@tool
def list_stage4_results() -> List[str]:
    """List all available Stage 4 execution results."""
    results = sorted([p.name for p in STAGE4_OUT_DIR.glob("execution_*.json")])
    return results


@tool
def load_stage4_result(result_id: str) -> str:
    """Load a Stage 4 execution result.
    
    Args:
        result_id: Result filename or pattern
        
    Returns:
        JSON string of the result
    """
    # Try exact match first
    result_path = STAGE4_OUT_DIR / result_id
    if not result_path.exists():
        # Try finding by pattern
        matches = list(STAGE4_OUT_DIR.glob(f"*{result_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No result found matching: {result_id}")
        result_path = matches[0]
    
    return result_path.read_text()


@tool
def load_stage3_plan_viz(plan_id: str) -> str:
    """Load a Stage 3 plan for context.
    
    Args:
        plan_id: Plan identifier
        
    Returns:
        JSON string of the plan
    """
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        matches = list(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No plan found matching: {plan_id}")
        plan_path = matches[0]
    
    return plan_path.read_text()


@tool
def create_visualizations(code: str, description: str = "Creating visualizations") -> str:
    """Execute Python code to create visualizations and reports.
    
    Use this to:
    - Load data from Stage 4 outputs
    - Create plots (matplotlib, seaborn, plotly)
    - Generate summary tables
    - Create HTML reports
    - Save visualizations to STAGE5_OUT_DIR
    
    Available in the environment:
    - pandas as pd
    - numpy as np
    - matplotlib.pyplot as plt
    - seaborn as sns
    - plotly.express as px (if available)
    - plotly.graph_objects as go (if available)
    - json, pathlib.Path
    - STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE5_WORKSPACE
    
    Args:
        code: Python code to execute
        description: Brief description of what this code creates
        
    Returns:
        Output printed to stdout, or error message
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Try to import plotly (optional)
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        has_plotly = True
    except ImportError:
        px = None
        go = None
        has_plotly = False
    
    def load_dataframe_viz(filepath):
        """Load a dataframe from any supported format."""
        filepath = Path(filepath)
        if not filepath.exists():
            # Try relative to STAGE4_OUT_DIR
            filepath = STAGE4_OUT_DIR / filepath.name
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return load_dataframe(filepath, base_dir=STAGE4_OUT_DIR)
    
    globals_dict = {
        "__name__": "__stage5_visualizer__",
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "json": json,
        "Path": Path,
        "STAGE4_OUT_DIR": STAGE4_OUT_DIR,
        "STAGE5_OUT_DIR": STAGE5_OUT_DIR,
        "STAGE5_WORKSPACE": STAGE5_WORKSPACE,
        "load_dataframe": load_dataframe_viz,
    }
    
    if has_plotly:
        globals_dict.update({"px": px, "go": go})
    
    # Use globals_dict for both globals and locals to avoid scope issues
    buf = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== {description} ===")
            # Use same dict for both to keep variables in scope
            exec(code, globals_dict, globals_dict)
            # Close any open plots
            plt.close('all')
    except Exception as e:
        plt.close('all')
        import traceback
        error_details = traceback.format_exc()
        return f"[ERROR] {e}\n\n{error_details}"
    
    output = buf.getvalue()
    return output if output else "[Visualizations created successfully]"


@tool
def save_visualization_report(report_json: str) -> str:
    """Save the final visualization report.
    
    Args:
        report_json: JSON string containing:
            - plan_id: ID of the executed plan
            - task_category: descriptive/predictive/unsupervised
            - visualizations: list of created visualization paths
            - html_report: path to HTML report (if created)
            - summary: text summary of visualizations
            - insights: key insights from the visualizations
    
    Returns:
        Confirmation message with save path
    """
    from datetime import datetime
    
    try:
        report = json.loads(report_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    plan_id = report.get("plan_id", "UNKNOWN")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2))
    
    return f"✅ Visualization report saved to: {output_path}"


# Stage 5 tool list
STAGE5_TOOLS = [
    list_stage4_results,
    load_stage4_result,
    load_stage3_plan_viz,
    create_visualizations,
    save_visualization_report,
]


# ===========================
# Complete Tool Registry
# ===========================

ALL_TOOLS = {
    "stage2": STAGE2_TOOLS,
    "stage3": STAGE3_TOOLS,
    "stage4": STAGE4_TOOLS,
    "stage5": STAGE5_TOOLS,
}
