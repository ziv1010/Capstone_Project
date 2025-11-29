"""
Centralized tool definitions for all stages of the agentic AI pipeline.

Tools are organized by stage and can be imported individually or as groups.
"""

from __future__ import annotations

import json
import io
import contextlib
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from langchain_core.tools import tool

from .config import (
    PROJECT_ROOT,
    OUTPUT_ROOT,
    DATA_DIR,
    SUMMARIES_DIR,
    STAGE2_OUT_DIR,
    STAGE3_OUT_DIR,
    STAGE3B_OUT_DIR,
    STAGE3_5_OUT_DIR,
    STAGE4_OUT_DIR,
    STAGE5_OUT_DIR,
    STAGE4_WORKSPACE,
    STAGE5_WORKSPACE,
)
from .models import Stage2Output, Stage3Plan, ExecutionResult
from .utils import (
    list_summary_files as _list_summary_files,
    read_summary_file as _read_summary_file,
    list_data_files as _list_data_files,
    inspect_data_file as _inspect_data_file,
    load_dataframe,
)

# Shared scratch slot to make the last tool result available to the sandbox
LAST_TOOL_RESULT: Any = None

# ===========================
# Generic / Failsafe Tools
# ===========================

@tool
def failsafe_python(code: str, description: str = "Failsafe scratchpad") -> str:
    """Execute arbitrary Python for diagnostics/debugging.
    
    Environment includes: pd, np, json, Path, DATA_DIR, OUTPUT_ROOT, PROJECT_ROOT,
    load_dataframe(filename, nrows=None), and print output is returned.
    """
    def load_dataframe_helper(filename: str, nrows: Optional[int] = None):
        return load_dataframe(filename, nrows=nrows, base_dir=DATA_DIR)

    globals_dict = {
        "__name__": "__failsafe_scratch__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "PROJECT_ROOT": PROJECT_ROOT,
        "load_dataframe": load_dataframe_helper,
        "description": description,
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== {description} ===")
            exec(code, globals_dict, globals_dict)
    except Exception as e:
        import traceback
        return f"[failsafe_python error] {e}\n{traceback.format_exc()}"

    return buf.getvalue() or "[failsafe_python done]"


@tool
def search(
    query: str,
    within: str = "project",
    file_glob: str = "**/*",
    max_matches: int = 30,
    case_sensitive: bool = False,
) -> str:
    """Search workspace text files for a pattern (regex supported).
    
    Args:
        query: Pattern to search for (regex).
        within: One of 'project', 'output', 'code', 'data', 'all'. Defaults to 'project'.
        file_glob: Glob filter for files (e.g., '*.json', '*.log').
        max_matches: Maximum number of line-level matches to return.
        case_sensitive: Whether the search is case sensitive.
        
    Returns:
        Matched lines with file paths and line numbers, or a message if none found.
    """
    root_map = {
        "project": PROJECT_ROOT,
        "output": OUTPUT_ROOT,
        "code": PROJECT_ROOT / "final_code",
        "data": DATA_DIR,
        "all": PROJECT_ROOT,
    }
    root = root_map.get(within, PROJECT_ROOT)

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error as e:
        return f"[search error] Invalid regex: {e}"

    hits: List[str] = []
    for path in root.rglob(file_glob):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 2_000_000:  # avoid huge files
                continue
        except OSError:
            continue

        try:
            with path.open("r", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    if pattern.search(line):
                        rel = path.relative_to(root)
                        hits.append(f"{rel}:{lineno}: {line.strip()}")
                        if len(hits) >= max_matches:
                            return "\n".join(hits)
        except (OSError, UnicodeDecodeError):
            continue

    return "\n".join(hits) if hits else "[search] No matches found."

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
        # Make prior tool outputs accessible for LLM convenience
        "result": LAST_TOOL_RESULT,
        "last_result": LAST_TOOL_RESULT,
        "last_tool_result": LAST_TOOL_RESULT,
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
STAGE2_TOOLS = [list_summary_files, read_summary_file, python_sandbox, search]


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
    # Parse JSON (with a light sanitization pass for invalid escapes)
    invalid_escape = re.compile(r'\\([^"\\/bfnrtu])')
    sanitized_payload = invalid_escape.sub(r"\1", plan_json)
    try:
        raw_obj = json.loads(sanitized_payload)
    except json.JSONDecodeError as e:
        debug_path = STAGE3_OUT_DIR / "failed_stage3_plan.json"
        debug_path.write_text(plan_json)
        start = max(e.pos - 40, 0)
        end = min(e.pos + 40, len(plan_json))
        snippet = plan_json[start:end]
        raise ValueError(
            f"Invalid JSON: {e}. Saved raw payload to {debug_path}. "
            f"Context: {snippet}"
        ) from e

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
            
            # Case 1: Equijoin with join_keys
            if js.join_keys:
                missing_left = [k for k in js.join_keys if k not in df_left.columns]
                if missing_left:
                    raise ValueError(f"Join {idx}: keys {missing_left} missing in {js.left_table}")
                
                if df_right is not None:
                    missing_right = [k for k in js.join_keys if k not in df_right.columns]
                    if missing_right:
                        raise ValueError(f"Join {idx}: keys {missing_right} missing in {js.right_table}")

            # Case 2: Different keys with left_on/right_on
            if js.left_on:
                missing_left = [k for k in js.left_on if k not in df_left.columns]
                if missing_left:
                    raise ValueError(f"Join {idx}: left_on keys {missing_left} missing in {js.left_table}")

            if js.right_on and df_right is not None:
                missing_right = [k for k in js.right_on if k not in df_right.columns]
                if missing_right:
                    raise ValueError(f"Join {idx}: right_on keys {missing_right} missing in {js.right_table}")
            
            # Ensure at least one join condition
            if not js.join_keys and not (js.left_on and js.right_on):
                raise ValueError(f"Join {idx}: Must specify either join_keys OR (left_on and right_on)")

    # Save
    out_path = STAGE3_OUT_DIR / f"{plan.plan_id}.json"
    out_path.write_text(plan.model_dump_json(indent=2))
    
    return f"✅ Plan saved successfully to: {out_path}"


# Stage 3 tool list
STAGE3_TOOLS = [
    load_task_proposal,
    list_data_files,
    inspect_data_file,
    search,
    python_sandbox_stage3,
    save_stage3_plan,
]


# ===========================
# Stage 3B: Data Preparation Tools
# ===========================

@tool
def load_stage3_plan_for_prep(plan_id: str) -> str:
    """Load a Stage 3 plan for data preparation.
    
    Args:
        plan_id: Plan identifier (e.g., 'PLAN-TSK-001')
        
    Returns:
        JSON string of the execution plan
    """
    from .config import STAGE3_OUT_DIR
    
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        matches = list(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No plan found matching: {plan_id}")
        plan_path = matches[0]
    
    return plan_path.read_text()


@tool
def python_sandbox_stage3b(code: str) -> str:
    """Execute Python code in a sandbox for Stage 3B data exploration.
    
    Args:
        code: Python code to execute. Can access load_dataframe(), DATA_DIR, etc.
        
    Returns:
        Output from the code execution
    """
    from .config import DATA_DIR, STAGE3B_OUT_DIR
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from io import StringIO
    import sys
    import os
    import traceback
    
    # Helper to resolve files relative to DATA_DIR when a bare filename is provided
    def _resolve_path(file: str) -> Path:
        path = Path(file)
        # Try the provided path, then DATA_DIR / <file>, then DATA_DIR / <name>
        candidates = [path]
        if not path.is_absolute():
            candidates.extend([
                DATA_DIR / file,
                STAGE3B_OUT_DIR / file,
            ])
        else:
            candidates.extend([
                DATA_DIR / path.name,
                STAGE3B_OUT_DIR / path.name,
            ])
        for cand in candidates:
            if cand.exists():
                return cand
        return path

    # Safe CSV reader that falls back to DATA_DIR if needed
    _pd_read_csv = pd.read_csv
    def _safe_read_csv(file, *args, **kwargs):
        target = _resolve_path(file)
        if target.exists():
            return _pd_read_csv(target, *args, **kwargs)
        return _pd_read_csv(file, *args, **kwargs)

    def _safe_load_dataframe(file, **kwargs):
        target = _resolve_path(file)
        if target.suffix in {".parquet", ".parq"}:
            return pd.read_parquet(target, **kwargs)
        return _safe_read_csv(target, **kwargs)
    
    # Inject helpers and monkeypatch read_csv so agent code using pd.read_csv works
    pd.read_csv = _safe_read_csv
    
    # Setup environment
    local_env = {
        "pd": pd,
        "np": np,
        "DATA_DIR": DATA_DIR,
        "STAGE3B_OUT_DIR": STAGE3B_OUT_DIR,
        "Path": Path,
        "load_dataframe": _safe_load_dataframe,
    }
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    old_cwd = os.getcwd()
    
    try:
        exec(code, local_env)
        output = captured.getvalue()
        return output if output else "[No output]"
    except Exception as e:
        return f"[ERROR] {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
    finally:
        pd.read_csv = _pd_read_csv
        os.chdir(old_cwd)
        sys.stdout = old_stdout


@tool
def run_data_prep_code(code: str, description: str) -> str:
    """Execute data preparation code and return results.
    
    This is the main tool for data loading, merging, filtering, and transformation.
    The code should save the prepared DataFrame to STAGE3B_OUT_DIR.
    
    Args:
        code: Python code for data preparation
        description: Brief description of what this code does
        
Returns:
        Execution status and preview of prepared data
    """
    from .config import DATA_DIR, STAGE3B_OUT_DIR
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from io import StringIO
    import sys
    import os
    import traceback
    
    print(f"\n=== Running data preparation: {description} ===")

    # Helper to resolve files relative to DATA_DIR when a bare filename is provided
    def _resolve_path(file: str) -> Path:
        path = Path(file)
        candidates = [path]
        if not path.is_absolute():
            candidates.extend([
                DATA_DIR / file,
                STAGE3B_OUT_DIR / file,
            ])
        else:
            candidates.extend([
                DATA_DIR / path.name,
                STAGE3B_OUT_DIR / path.name,
            ])
        for cand in candidates:
            if cand.exists():
                return cand
        return path

    _pd_read_csv = pd.read_csv
    def _safe_read_csv(file, *args, **kwargs):
        target = _resolve_path(file)
        if target.exists():
            return _pd_read_csv(target, *args, **kwargs)
        return _pd_read_csv(file, *args, **kwargs)

    def _safe_load_dataframe(file, **kwargs):
        target = _resolve_path(file)
        if target.suffix in {".parquet", ".parq"}:
            return pd.read_parquet(target, **kwargs)
        return _safe_read_csv(target, **kwargs)

    # Monkeypatch pandas.read_csv so agent code using pd.read_csv(...) works
    pd.read_csv = _safe_read_csv
    
    # Setup environment with helper functions
    local_env = {
        "pd": pd,
        "np": np,
        "DATA_DIR": DATA_DIR,
        "STAGE3B_OUT_DIR": STAGE3B_OUT_DIR,
        "Path": Path,
        "load_dataframe": _safe_load_dataframe,
    }
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    old_cwd = os.getcwd()
    
    try:
        # Run prep code inside STAGE3B_OUT_DIR so relative writes land there
        os.chdir(STAGE3B_OUT_DIR)
        exec(code, local_env)
        output = captured.getvalue()
        
        # Check if prepared_df was created
        if "prepared_df" in local_env:
            df = local_env["prepared_df"]
            preview = f"\n\nPrepared DataFrame Info:\n"
            preview += f"  Shape: {df.shape}\n"
            preview += f"  Columns: {list(df.columns)}\n"
            preview += f"\nFirst 3 rows:\n{df.head(3).to_string()}\n"
            return output + preview
        else:
            return output if output else "[Code executed successfully, no output]"
            
    except Exception as e:
        error_msg = f"[ERROR] {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return error_msg
    finally:
        pd.read_csv = _pd_read_csv
        os.chdir(old_cwd)
        sys.stdout = old_stdout


@tool
def save_prepared_data(
    plan_id: str,
    prepared_file_name: str,
    original_row_count: int,
    prepared_row_count: int,
    columns_created: List[str],
    transformations_applied: List[str],
    data_quality_report: Dict[str, Any]
) -> str:
    """Save prepared data output for Stage 3B.
    
    This should be called after successfully preparing the data.
    The prepared DataFrame should already be saved as parquet/csv.
    
    Args:
        plan_id: Plan ID (e.g., 'PLAN-TSK-001')
        prepared_file_name: Name of the saved prepared data file
        original_row_count: Number of rows before preparation
        prepared_row_count: Number of rows after preparation
        columns_created: List of feature engineering columns created
        transformations_applied: List of transformations (filters, joins, etc.)
        data_quality_report: Data quality metrics
        
    Returns:
        Success message with file path
    """
    from .config import STAGE3B_OUT_DIR
    from .models import PreparedDataOutput
    import json
    from datetime import datetime
    
    output = PreparedDataOutput(
        plan_id=plan_id,
        prepared_file_path=str(STAGE3B_OUT_DIR / prepared_file_name),
        original_row_count=original_row_count,
        prepared_row_count=prepared_row_count,
        columns_created=columns_created,
        transformations_applied=transformations_applied,
        data_quality_report=data_quality_report,
    )
    
    # Save metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = STAGE3B_OUT_DIR / f"prep_{plan_id}_{timestamp}.json"
    output_path.write_text(output.model_dump_json(indent=2))
    
    return f"✅ Prepared data output saved to: {output_path}\n\nsaved::prep_{plan_id}"


# ===========================
# Stage 3.5: Method Testing & Benchmarking Tools
# ===========================

@tool
def load_stage3_plan_for_tester(plan_id: str) -> str:
    """Load a Stage 3 plan for method testing.
    
    Args:
        plan_id: Plan identifier (e.g., 'PLAN-TSK-001')
        
    Returns:
        JSON string of the plan
    """
    from .config import STAGE3_OUT_DIR
    
    plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        # Try finding by pattern
        matches = list(STAGE3_OUT_DIR.glob(f"*{plan_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"No plan found matching: {plan_id}")
        plan_path = matches[0]
    
    return plan_path.read_text()

@tool
def run_benchmark_code(code: str, description: str = "Running benchmark") -> str:
    """Execute Python code for benchmarking forecasting methods.

    The caller is responsible for every modeling choice.
    The code you provide should:
    - Choose and implement the forecasting methods (nothing is pre-selected)
    - Import any trusted libraries it needs, handling missing packages gracefully
    - Design the time-based train/validation/test splits
    - Compute task-appropriate metrics of your choosing
    - Optionally save artifacts (plots, models, CSVs) under STAGE3_5_OUT_DIR

    Predefined objects in the execution environment:
    - pd, np
    - json, Path
    - DATA_DIR, STAGE3_5_OUT_DIR
    - load_dataframe(filename, nrows=None) -> pd.DataFrame
    - time

    Everything else (models, metrics, imports, logic) is fully under the code's control.
    """
    from .config import DATA_DIR, STAGE3_5_OUT_DIR
    import time

    def load_dataframe_helper(filename: str, nrows: Optional[int] = None):
        """Load a dataframe, preferring DATA_DIR then Stage 3B output for prepared parquet."""
        try:
            return load_dataframe(filename, nrows=nrows, base_dir=DATA_DIR)
        except FileNotFoundError:
            prepared_path = STAGE3B_OUT_DIR / filename
            if prepared_path.exists():
                return load_dataframe(prepared_path)
            raise

    globals_dict = {
        "__name__": "__stage3_5_tester__",
        "__builtins__": __builtins__,  # allow normal imports
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "STAGE3_5_OUT_DIR": STAGE3_5_OUT_DIR,
        "load_dataframe": load_dataframe_helper,
        "time": time,
    }

    buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== {description} ===")
            # Use globals_dict for both globals and locals so code can define functions/vars and reuse them
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                exec(code, globals_dict, globals_dict)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"[ERROR] {e}\n\nTraceback:\n{error_details}"

    output = buf.getvalue()
    return output if output else "[Code executed successfully, no output]"


@tool
def python_sandbox_stage3_5(code: str) -> str:
    """Quick Python sandbox for Stage 3.5 data exploration and prep.

    Use this to:
    - Inspect columns, dtypes, and sample rows
    - Prototype lightweight cleaning/prep helpers shared across methods
    - Verify parsing of date/time columns before full benchmarks

    Available:
    - pd, np, json, Path
    - DATA_DIR, STAGE3_5_OUT_DIR
    - load_dataframe(filename, nrows=None)

    Nothing is pre-hardcoded—your code decides what to do.
    """
    from .config import DATA_DIR, STAGE3_5_OUT_DIR

    def load_dataframe_helper(filename: str, nrows: Optional[int] = None):
        try:
            return load_dataframe(filename, nrows=nrows, base_dir=DATA_DIR)
        except FileNotFoundError:
            prepared_path = STAGE3B_OUT_DIR / filename
            if prepared_path.exists():
                return load_dataframe(prepared_path)
            raise

    globals_dict = {
        "__name__": "__stage3_5_sandbox__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "STAGE3_5_OUT_DIR": STAGE3_5_OUT_DIR,
        "load_dataframe": load_dataframe_helper,
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globals_dict, globals_dict)
    except Exception as e:
        return f"[ERROR] {e}"
    return buf.getvalue() or "[No output]"


@tool
def save_tester_output(output_json: Dict[str, Any]) -> str:
    """Save the final tester output with method selection results.
    
    Args:
        output_json: JSON payload containing:
            - plan_id: ID of the plan being tested
            - task_category: predictive/descriptive/unsupervised
            - methods_proposed: list of ForecastingMethod objects
            - benchmark_results: list of BenchmarkResult objects
            - selected_method_id: ID of the winning method
            - selected_method: The complete ForecastingMethod object for the winner
            - selection_rationale: Why this method was selected
            - data_split_strategy: How data was split for testing
            
    Returns:
        Confirmation message with save path
    """
    from .config import STAGE3_5_OUT_DIR
    from .models import TesterOutput
    from datetime import datetime
    
    # Allow lenient inputs: accept dict or JSON string
    if isinstance(output_json, str):
        try:
            output_data = json.loads(output_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    elif isinstance(output_json, dict):
        output_data = output_json
    else:
        raise ValueError("output_json must be a dict or JSON string")
    
    # Validate against schema
    try:
        tester_output = TesterOutput.model_validate(output_data)
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}")
    
    plan_id = tester_output.plan_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    STAGE3_5_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = STAGE3_5_OUT_DIR / f"tester_{plan_id}_{timestamp}.json"
    output_path.write_text(json.dumps(output_data, indent=2))
    
    return f"saved::{output_path.name}"


@tool
def record_thought(thought: str, what_im_about_to_do: str) -> str:
    """Record your reasoning BEFORE taking an action.
    
    Use this to explicitly document your thinking before calling other tools.
    This helps you stay strategic and avoid repeating mistakes.
    
    Args:
        thought: Your current reasoning - what you know, what's uncertain, what you're considering
        what_im_about_to_do: What action you plan to take next and WHY
        
    Returns:
        Confirmation message
        
    Example:
        record_thought(
            thought="I've seen that the export data has yearly columns but the production data only has 2020-2025. "
                    "A cross-file join won't work because there's no common key.",
            what_im_about_to_do="I'll use python_sandbox to test loading just the export data and reshaping it to long format"
        )
    """
    return f"💭 Thought recorded. Proceeding with: {what_im_about_to_do[:80]}..."


@tool
def record_observation(what_happened: str, what_i_learned: str, next_step: str) -> str:
    """Record what you observed and learned AFTER an action.
    
    Use this to reflect on tool results before deciding what to do next.
    This helps you learn from errors and adjust your strategy.
    
    Args:
        what_happened: What the last tool/action resulted in (success, error, unexpected result)
        what_i_learned: Key insight or lesson from this result
        next_step: What you'll do next based on what you learned
        
    Returns:
        Confirmation message
        
    Example:
        record_observation(
            what_happened="run_benchmark_code failed with 'Found array with 0 samples'",
            what_i_learned="The validation set is empty after dropna() - this means my slicing strategy is wrong",
            next_step="I'll inspect the data shape before/after split to understand the actual structure"
        )
    """
    return f"👁️ Observation recorded. Learning: {what_i_learned[:80]}... → Next: {next_step[:60]}..."


# Stage 3B tool list
STAGE3B_TOOLS = [
    record_thought,  # ReAct: explicit reasoning
    record_observation,  # ReAct: reflection
    load_stage3_plan_for_prep,
    list_data_files,
    inspect_data_file,
    python_sandbox_stage3b,
    run_data_prep_code,
    save_prepared_data,
    search,
]

# Stage 3.5 tool list
STAGE3_5_TOOLS = [
    record_thought,  # ReAct: explicit reasoning before action
    record_observation,  # ReAct: reflection after action
    load_stage3_plan_for_tester,
    list_summary_files,  # Access Stage 1 summaries
    read_summary_file,  # Read dataset summaries
    search,
    list_data_files,
    inspect_data_file,
    python_sandbox_stage3_5,
    run_benchmark_code,
    save_tester_output,
]



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
# Failsafe / Debugging Tools
# ===========================

FAILSAFE_TOOLS = [
    failsafe_python,
    search,
    list_data_files,
    inspect_data_file,
]


# ===========================
# Complete Tool Registry
# ===========================

ALL_TOOLS = {
    "stage2": STAGE2_TOOLS,
    "stage3": STAGE3_TOOLS,
    "stage3_5": STAGE3_5_TOOLS,
    "stage4": STAGE4_TOOLS,
    "stage5": STAGE5_TOOLS,
    "failsafe": FAILSAFE_TOOLS,
}


# ===========================
# Stage 0: Conversational Tools
# ===========================

@tool
def trigger_pipeline_stages(
    start_stage: int,
    end_stage: int,
    task_id: Optional[str] = None,
    user_query: Optional[str] = None,
) -> str:
    """Triggers execution of pipeline stages (1-5) based on the conversational query.
    
    Args:
        start_stage: Stage to start from (1-5)
        end_stage: Stage to end at (1-5)
        task_id: Optional task ID for Stages 3+
        user_query: Optional user request to guide proposal generation (Stage 2)
        
    Returns:
        Execution summary string
    """
    from .master_agent import run_partial_pipeline
    
    try:
        # Run the pipeline
        state = run_partial_pipeline(start_stage, end_stage, task_id, user_query=user_query)
        
        # Format summary based on what ran
        summary = []
        summary.append(f"✅ Pipeline stages {start_stage}-{end_stage} completed successfully.")
        
        if state.get("dataset_summaries"):
            summary.append(f"- Generated {len(state['dataset_summaries'])} dataset summaries")
            
        if state.get("task_proposals"):
            summary.append(f"- Generated {len(state['task_proposals'])} task proposals:")
            for p in state['task_proposals']:
                summary.append(f"  * [{p.id}] {p.category}: {p.title}")
            
        if state.get("stage3_plan"):
            summary.append(f"- Created execution plan: {state['stage3_plan'].plan_id}")
            
        if state.get("execution_result"):
            status = state['execution_result'].status
            summary.append(f"- Execution finished with status: {status}")
            
        if state.get("visualization_report"):
            viz_count = len(state['visualization_report'].visualizations)
            summary.append(f"- Created {viz_count} visualizations")
            
        return "\n".join(summary)
        
    except Exception as e:
        return f"❌ Pipeline execution failed: {str(e)}"


@tool
def query_data_capabilities() -> str:
    """Returns a summary of available datasets and what predictions/analyses are possible.
    
    Checks for existing Stage 1 summaries and Stage 2 task proposals.
    If not available, triggers Stage 1-2 automatically.
    
    Returns:
        Text summary of capabilities
    """
    from .master_agent import run_partial_pipeline
    
    # Check if we have summaries
    summaries = _list_summary_files()
    
    if not summaries:
        return "No data summaries found. Please run 'trigger_pipeline_stages(1, 2)' first to analyze the data."
        
    # Read summaries
    summary_texts = []
    for s in summaries:
        try:
            content = _read_summary_file(s)
            data = json.loads(content)
            summary_texts.append(f"- {data.get('filename', s)}: {data.get('description', 'No description')}")
        except:
            summary_texts.append(f"- {s}")
            
    # Check for proposals
    proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
    proposals_text = ""
    
    if proposals_path.exists():
        try:
            data = json.loads(proposals_path.read_text())
            stage2 = Stage2Output.model_validate(data)
            proposals_text = "\n\nAvailable Analysis Tasks:\n"
            for p in stage2.proposals:
                proposals_text += f"- [{p.id}] {p.category}: {p.title}\n"
        except Exception as e:
            proposals_text = f"\n(Could not read existing proposals: {e})"
    else:
        proposals_text = "\n(No specific task proposals generated yet)"
        
    return f"Data Capabilities:\n\nDatasets:\n" + "\n".join(summary_texts) + proposals_text


@tool
def execute_dynamic_analysis(question: str, code: str, description: str) -> str:
    """General-purpose tool for running custom analysis code.
    
    The agent generates code based on the user's question.
    Has access to all data files and can create ad-hoc predictions.
    
    Args:
        question: The user's original question
        code: Python code to execute
        description: Brief description of the analysis
        
    Returns:
        Results as formatted text suitable for conversation
    """
    # Reuse the Stage 4 execution environment but with a focus on immediate text output
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    def load_dataframe_helper(filename: str):
        return load_dataframe(filename, base_dir=DATA_DIR)
    
    globals_dict = {
        "__name__": "__dynamic_analyzer__",
        "pd": pd,
        "np": np,
        "json": json,
        "Path": Path,
        "DATA_DIR": DATA_DIR,
        "load_dataframe": load_dataframe_helper,
        "plt": plt,
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "mean_squared_error": mean_squared_error,
        "train_test_split": train_test_split,
    }
    
    local_env = {}
    buf = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(buf):
            print(f"=== Analysis: {description} ===")
            exec(code, globals_dict, local_env)
    except Exception as e:
        return f"❌ Analysis failed: {e}"
    
    return buf.getvalue() or "[Analysis finished with no output]"


@tool
def get_conversation_context() -> str:
    """Returns current conversation state (query history, completed analyses, cached results).
    
    Returns:
        JSON string of conversation state
    """
    state_path = OUTPUT_ROOT / "conversation_state.json"
    if not state_path.exists():
        return json.dumps({
            "history": [],
            "completed_tasks": [],
            "last_updated": "never"
        })
    return state_path.read_text()


@tool
def save_conversation_state(state_json: str) -> str:
    """Saves current conversation state for persistence.
    
    Args:
        state_json: JSON string of state
        
    Returns:
        Confirmation message
    """
    try:
        # Validate valid JSON
        json.loads(state_json)
        state_path = OUTPUT_ROOT / "conversation_state.json"
        state_path.write_text(state_json)
        return "✅ Conversation state saved"
    except Exception as e:
        return f"❌ Failed to save state: {e}"


# Stage 0 tool list
STAGE0_TOOLS = [
    trigger_pipeline_stages,
    query_data_capabilities,
    execute_dynamic_analysis,
    get_conversation_context,
    save_conversation_state,
]

# Update ALL_TOOLS
ALL_TOOLS["stage0"] = STAGE0_TOOLS
