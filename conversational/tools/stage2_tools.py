"""
Stage 2 Tools: Task Proposal Generation

Tools for exploring dataset summaries and proposing analytical tasks.
"""

import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, SUMMARIES_DIR, STAGE2_OUT_DIR,
    DataPassingManager, logger
)
from code.utils import (
    list_summary_files, read_summary_file,
    execute_python_sandbox, safe_json_dumps
)


@tool
def list_dataset_summaries() -> str:
    """
    List all available dataset summaries from Stage 1.

    Returns list of summary files with basic info about each dataset.
    IMPORTANT: Note the actual data filename to use in proposals!
    """
    try:
        files = list_summary_files(SUMMARIES_DIR)
        if not files:
            return "No dataset summaries found. Run Stage 1 first."

        result = [
            "Available Dataset Summaries:",
            "",
            "Format: summary_file → actual_data_file (rows x cols)"
        ]
        for f in files:
            try:
                summary = read_summary_file(f, SUMMARIES_DIR)
                data = summary.get('data', summary)  # Handle wrapped format
                n_rows = data.get('n_rows', 'unknown')
                n_cols = data.get('n_cols', 'unknown')
                actual_filename = data.get('filename', f.replace('.summary.json', ''))
                result.append(f"  - {f}")
                result.append(f"    → Data file: {actual_filename} ({n_rows} rows, {n_cols} cols)")
            except:
                result.append(f"  - {f}")

        result.append("")
        result.append("*** IMPORTANT: Use the 'Data file' name (e.g., 'file.csv') in required_datasets ***")
        result.append("*** NOT the summary filename (e.g., 'file.summary.json') ***")

        return "\n".join(result)

    except Exception as e:
        return f"Error listing summaries: {e}"


@tool
def read_dataset_summary(filename: str) -> str:
    """
    Read a specific dataset summary.

    Args:
        filename: Name of the summary file (e.g., 'dataset.summary.json')

    Returns:
        Full summary content as formatted string
    """
    try:
        summary = read_summary_file(filename, SUMMARIES_DIR)
        data = summary.get('data', summary)  # Handle wrapped format

        # Get the actual data filename (not summary filename)
        actual_filename = data.get('filename', 'unknown')

        # Format for readability
        result = [
            f"=== Summary: {filename} ===",
            f"",
            f"*** IMPORTANT: Use this filename in required_datasets: {actual_filename} ***",
            f"",
            f"Dataset: {actual_filename}",
            f"Shape: {data.get('n_rows', '?')} rows x {data.get('n_cols', '?')} columns",
            "",
            "Columns:",
        ]

        for col in data.get('columns', []):
            name = col.get('name', 'unknown')
            ltype = col.get('logical_type', 'unknown')
            nulls = col.get('null_fraction', 0)
            result.append(f"  - {name}: {ltype} (nulls: {nulls:.1%})")

            # Add semantic info for categorical columns (helps model understand values)
            if col.get('value_interpretation'):
                result.append(f"    → {col['value_interpretation']}")
            elif col.get('unique_values') and len(col['unique_values']) <= 10:
                result.append(f"    → Values: {', '.join(str(v) for v in col['unique_values'])}")

        if data.get('candidate_keys'):
            result.append(f"\nCandidate Keys: {data.get('candidate_keys')}")

        if data.get('has_target_candidates'):
            result.append(f"Target Candidates: {data.get('has_target_candidates')}")

        if data.get('has_datetime_column'):
            result.append("Has datetime column: Yes")

        result.append("")
        result.append(f"Remember: Use '{actual_filename}' (not '{filename}') in your proposals!")

        return "\n".join(result)

    except FileNotFoundError:
        return f"Summary file not found: {filename}"
    except Exception as e:
        return f"Error reading summary: {e}"


@tool
def explore_data_relationships(summary_files: str = "") -> str:
    """
    Explore potential relationships between datasets for multi-dataset tasks.

    IMPORTANT: Call this to find join keys and create multi-dataset proposals!

    Args:
        summary_files: Comma-separated list of summary filenames.
                       If empty, automatically uses ALL available summaries.
                       NOTE: If filenames contain commas, pass empty string to use all files.

    Returns:
        Analysis of potential joins and relationships between datasets
    """
    try:
        # Get all available summary files first
        all_available = list_summary_files(SUMMARIES_DIR)
        if not all_available:
            return "No summary files found. Run Stage 1 first."
        
        # If no files specified, use all available summaries
        if not summary_files or summary_files.strip() == "":
            files = all_available
            logger.debug(f"Using all {len(files)} available summary files")
        else:
            # Try to match against available files to handle commas in filenames
            # This is more robust than naive comma splitting
            files = []
            remaining = summary_files.strip()
            
            # Sort by length descending to match longer filenames first
            sorted_available = sorted(all_available, key=len, reverse=True)
            
            for avail_file in sorted_available:
                if avail_file in remaining:
                    files.append(avail_file)
                    # Remove the matched filename from remaining string
                    remaining = remaining.replace(avail_file, '', 1)
            
            # If we couldn't match any files, fall back to all files
            if not files:
                logger.warning(f"Could not match any files from input '{summary_files}', using all available")
                files = all_available
            
            logger.debug(f"Matched {len(files)} files from input")

        summaries = {}
        # Map summary filename to actual data filename
        filename_map = {}

        for f in files:
            try:
                summary = read_summary_file(f, SUMMARIES_DIR)
                data = summary.get('data', summary)
                summaries[f] = data
                # Store the actual data filename (CSV, not summary.json)
                filename_map[f] = data.get('filename', f.replace('.summary.json', ''))
            except Exception as e:
                logger.warning(f"Could not read summary file {f}: {e}")
                continue  # Skip this file instead of failing entirely

        if len(summaries) < 2:
            return "Need at least 2 datasets to explore relationships."

        result = [
            "=== Data Relationship Analysis ===",
            "",
            "*** USE THESE FILENAMES IN YOUR PROPOSALS (not .summary.json files): ***"
        ]

        # Show filename mapping
        for summary_file, actual_file in filename_map.items():
            result.append(f"  {summary_file} → {actual_file}")

        result.append("")

        # Find common column names
        all_columns = {}
        for fname, data in summaries.items():
            actual_file = filename_map[fname]
            for col in data.get('columns', []):
                col_name = col.get('name', '').lower()
                if col_name not in all_columns:
                    all_columns[col_name] = []
                all_columns[col_name].append({
                    'summary_file': fname,
                    'actual_file': actual_file,
                    'original_name': col.get('name'),
                    'type': col.get('logical_type'),
                    'unique_fraction': col.get('unique_fraction', 0)
                })

        # Find potential join keys
        result.append("Potential Join Keys (columns in multiple datasets):")
        join_candidates = []
        for col_name, occurrences in all_columns.items():
            if len(occurrences) > 1:
                files_with_col = [o['actual_file'] for o in occurrences]
                types = [o['type'] for o in occurrences]
                result.append(f"  - '{col_name}' found in: {files_with_col}")
                result.append(f"    Types: {types}")

                # Good join key if types match or both categorical/numeric
                max_unique = max(o['unique_fraction'] for o in occurrences)
                if max_unique > 0.1:  # Lower threshold to catch more join candidates
                    join_candidates.append({
                        'column': col_name,
                        'files': files_with_col,
                        'uniqueness': max_unique
                    })

        if join_candidates:
            result.append(f"\nBest join candidates: {[jc['column'] for jc in join_candidates]}")
        else:
            result.append("\nNo obvious join candidates, but consider joining on similar categorical columns.")

        # Suggest specific multi-dataset proposals
        result.append("\n" + "=" * 60)
        result.append("SUGGESTED MULTI-DATASET TASK IDEAS:")
        result.append("=" * 60)

        for i, (f1, d1) in enumerate(summaries.items()):
            for f2, d2 in list(summaries.items())[i+1:]:
                actual_f1 = filename_map[f1]
                actual_f2 = filename_map[f2]
                cols1 = {c['name'].lower(): c for c in d1.get('columns', [])}
                cols2 = {c['name'].lower(): c for c in d2.get('columns', [])}
                common = set(cols1.keys()) & set(cols2.keys())

                result.append(f"\n{actual_f1} + {actual_f2}:")
                if common:
                    result.append(f"  Join on: {list(common)}")
                    # Find numeric columns that could be targets
                    targets1 = [c['name'] for c in d1.get('columns', [])
                               if c.get('logical_type') in ['numeric', 'integer', 'float']]
                    targets2 = [c['name'] for c in d2.get('columns', [])
                               if c.get('logical_type') in ['numeric', 'integer', 'float']]
                    if targets1 or targets2:
                        result.append(f"  Potential targets: {targets1[:3] + targets2[:3]}")
                    result.append(f"  Example proposal: Join {actual_f1} with {actual_f2} on {list(common)[0] if common else 'similar columns'}")
                else:
                    result.append("  No common columns - consider if semantic relationship exists")

        result.append("\n" + "=" * 60)
        result.append("REMEMBER: At least 2 of your 5 proposals MUST use multiple datasets!")
        result.append("=" * 60)

        return "\n".join(result)

    except Exception as e:
        return f"Error exploring relationships: {e}"


@tool
def python_sandbox_stage2(code: str, description: str = "") -> str:
    """
    Execute Python code in a sandboxed environment.

    Available in sandbox:
    - pd (pandas), np (numpy), json
    - DATA_DIR, SUMMARIES_DIR
    - load_dataframe() function

    Args:
        code: Python code to execute
        description: Description of what the code does

    Returns:
        Output from code execution
    """
    try:
        # Add stage-specific imports to sandbox
        additional = {
            'SUMMARIES_DIR': SUMMARIES_DIR,
        }
        return execute_python_sandbox(code, additional, description)
    except Exception as e:
        return f"Sandbox error: {e}"


@tool
def evaluate_forecasting_feasibility(
    target_column: str,
    date_column: str,
    dataset_summaries: str
) -> str:
    """
    Evaluate if a forecasting task is feasible given the data.

    Args:
        target_column: The column to predict
        date_column: The datetime column to use
        dataset_summaries: Comma-separated list of relevant summary files

    Returns:
        Feasibility assessment with score and recommendations
    """
    try:
        files = [f.strip() for f in dataset_summaries.split(',')]

        report = ["=== Forecasting Feasibility Assessment ===\n"]
        issues = []
        score = 100

        for f in files:
            try:
                summary = read_summary_file(f, SUMMARIES_DIR)
                data = summary.get('data', summary)
                columns = {c['name']: c for c in data.get('columns', [])}

                report.append(f"Dataset: {f}")

                # Check target column
                if target_column in columns:
                    target = columns[target_column]
                    report.append(f"  Target '{target_column}': {target.get('logical_type')}")

                    if target.get('logical_type') not in ['integer', 'float', 'numeric']:
                        issues.append(f"Target '{target_column}' is not numeric")
                        score -= 30

                    if target.get('null_fraction', 0) > 0.2:
                        issues.append(f"Target has {target.get('null_fraction'):.1%} nulls")
                        score -= 20
                else:
                    issues.append(f"Target '{target_column}' not found in {f}")
                    score -= 50

                # Check date column
                if date_column in columns:
                    date = columns[date_column]
                    report.append(f"  Date '{date_column}': {date.get('logical_type')}")

                    if date.get('logical_type') != 'datetime':
                        issues.append(f"'{date_column}' is not datetime type")
                        score -= 20
                else:
                    issues.append(f"Date column '{date_column}' not found in {f}")
                    score -= 40

                # Check data size
                n_rows = data.get('n_rows', 0)
                report.append(f"  Rows: {n_rows}")
                if n_rows < 50:
                    issues.append("Very few data points for forecasting")
                    score -= 30
                elif n_rows < 100:
                    issues.append("Limited data points - results may be unreliable")
                    score -= 10

            except Exception as e:
                issues.append(f"Could not read {f}: {e}")
                score -= 25

        report.append("")
        score = max(0, score)

        if issues:
            report.append("Issues Found:")
            for issue in issues:
                report.append(f"  - {issue}")
            report.append("")

        report.append(f"Feasibility Score: {score}/100")

        if score >= 70:
            report.append("VERDICT: Forecasting is feasible")
        elif score >= 40:
            report.append("VERDICT: Forecasting may work with preprocessing")
        else:
            report.append("VERDICT: Forecasting is not recommended")
            report.append("Consider: classification, clustering, or descriptive analysis")

        return "\n".join(report)

    except Exception as e:
        return f"Error assessing feasibility: {e}"


def _sanitize_json_string(json_str: str) -> str:
    """
    Attempt to fix common JSON formatting issues that LLMs make.

    This is dataset-agnostic and fixes structural issues, not content.
    """
    import re

    # Remove any BOM or weird characters at start
    json_str = json_str.strip().lstrip('\ufeff')
    
    # Remove markdown code blocks if present
    if json_str.startswith('```') or json_str.startswith('```json'):
        lines = json_str.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines[-1].startswith('```'):
            lines = lines[:-1]
        json_str = '\n'.join(lines).strip()

    # Fix trailing commas before closing brackets/braces
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # Fix missing commas between array elements (objects)
    # Pattern: } { or }\n{ or }"{
    json_str = re.sub(r'}\s*({)', r'},\1', json_str)
    json_str = re.sub(r'}\s*(")', r'},\1', json_str)
    
    # Fix missing commas between array elements (arrays)
    # Pattern: ] [ or ]\n[
    json_str = re.sub(r']\s*(\[)', r'],\1', json_str)
    
    # Fix missing commas after strings in arrays/objects
    # Pattern: "value" "next" -> "value", "next"
    # Be careful not to match inside strings
    # This is hard with regex, skipping for safety unless specialized library used

    return json_str


def _get_json_error_context(json_str: str, error: json.JSONDecodeError, context_lines: int = 3) -> str:
    """
    Extract context around a JSON parsing error to help LLM debug.

    Args:
        json_str: The JSON string that failed to parse
        error: The JSONDecodeError exception
        context_lines: Number of lines to show before/after error

    Returns:
        Formatted error context with line numbers
    """
    lines = json_str.split('\n')
    error_line = error.lineno - 1  # Convert to 0-indexed
    error_col = error.colno - 1

    # Calculate context window
    start_line = max(0, error_line - context_lines)
    end_line = min(len(lines), error_line + context_lines + 1)

    context = []
    context.append(f"JSON Parse Error: {error.msg}")
    context.append(f"Location: Line {error.lineno}, Column {error.colno}")
    context.append("\nContext:")
    context.append("-" * 60)

    for i in range(start_line, end_line):
        line_num = i + 1
        marker = ">>> " if i == error_line else "    "
        context.append(f"{marker}{line_num:4d} | {lines[i]}")

        # Show error position with arrow
        if i == error_line:
            arrow = " " * (len(marker) + 7 + error_col) + "^--- ERROR HERE"
            context.append(arrow)

    context.append("-" * 60)
    context.append("\nCommon fixes:")
    context.append("1. Check for missing commas between array/object elements")
    context.append("2. Check for trailing commas before closing }, ]")
    context.append("3. Ensure all strings are properly quoted")
    context.append("4. Verify all brackets/braces are balanced")
    context.append("\nPlease fix the JSON and try again.")

    return "\n".join(context)


@tool
def save_task_proposals(proposals_json: str) -> str:
    """
    Save task proposals to the Stage 2 output directory.

    Args:
        proposals_json: JSON string containing task proposals

    Returns:
        Confirmation with saved path
    """
    try:
        logger.info("=" * 40)
        logger.info("save_task_proposals tool called")
        logger.info("=" * 40)

        # Log the first 500 chars of input for debugging
        logger.debug(f"Received proposals_json (first 500 chars): {proposals_json[:500] if proposals_json else 'None'}...")

        # Try to sanitize common JSON issues before parsing
        original_json = proposals_json
        proposals_json = _sanitize_json_string(proposals_json)

        if proposals_json != original_json:
            logger.debug("Applied JSON sanitization fixes")

        # Attempt to parse JSON
        try:
            proposals = json.loads(proposals_json)
        except json.JSONDecodeError as e:
            # If parsing fails, provide detailed error context
            error_context = _get_json_error_context(proposals_json, e, context_lines=5)
            logger.error(f"JSON parsing failed:\n{error_context}")
            return f"JSON_PARSE_ERROR:\n{error_context}"

        # Validate structure
        if 'proposals' not in proposals:
            logger.error("Error: JSON must contain 'proposals' key")
            error_msg = (
                "Error: JSON structure invalid\n"
                "Expected format:\n"
                "{\n"
                '  "proposals": [\n'
                '    { "id": "TSK-001", ... },\n'
                '    { "id": "TSK-002", ... }\n'
                "  ]\n"
                "}\n\n"
                f"Received keys: {list(proposals.keys())}\n"
                "Please wrap your proposals array in an object with 'proposals' key."
            )
            return error_msg

        n_proposals = len(proposals.get('proposals', []))
        logger.info(f"Parsed {n_proposals} proposals from JSON")

        # Log each proposal ID and title
        for p in proposals.get('proposals', []):
            logger.debug(f"  - {p.get('id', 'N/A')}: {p.get('title', 'N/A')}")

        # Save using DataPassingManager for robust saving
        output_path = DataPassingManager.save_artifact(
            data=proposals,
            output_dir=STAGE2_OUT_DIR,
            filename="task_proposals.json",
            metadata={"stage": "stage2", "type": "task_proposals"}
        )

        logger.info(f"✅ Saved {n_proposals} task proposals to: {output_path}")

        # Verify the file was saved
        if output_path.exists():
            logger.debug(f"Verified: File exists at {output_path}")
        else:
            logger.error(f"WARNING: File was not created at {output_path}")

        return f"SUCCESS: Saved {n_proposals} task proposals to: {output_path}"

    except Exception as e:
        logger.error(f"Error saving proposals: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return f"Error saving proposals: {e}"


@tool
def get_proposal_template() -> str:
    """
    Get a template for creating task proposals.

    Returns JSON template with all required fields explained.
    """
    template = {
        "proposals": [
            {
                "id": "TSK-001",
                "category": "forecasting",
                "title": "Short descriptive title",
                "problem_statement": "Detailed description of the analytical problem",
                "required_datasets": ["dataset1.csv"],
                "target_column": "column_to_predict",
                "target_dataset": "dataset1.csv",
                "feature_columns": ["feature1", "feature2"],
                "validation_plan": {
                    "train_fraction": 0.7,
                    "validation_fraction": 0.15,
                    "test_fraction": 0.15,
                    "split_strategy": "temporal",
                    "date_column": "date_column_name"
                },
                "feasibility_score": 0.8,
                "feasibility_notes": "Notes on why this task is feasible",
                "forecast_horizon": 30,
                "forecast_granularity": "daily",
                "forecast_type": "multi_step",
                "evaluation_metrics": ["mae", "rmse", "mape", "r2"]
            }
        ]
    }

    explanation = """
Task Proposal Template:

IMPORTANT JSON RULES:
1. Wrap proposals in an object with "proposals" key
2. Use double quotes for all strings and keys
3. Separate array/object elements with commas
4. NO trailing commas before closing ], }
5. Ensure all brackets are balanced: { }, [ ]

Required fields:
- id: Unique ID like "TSK-001", "TSK-002", etc.
- category: One of: forecasting, regression, classification, clustering, descriptive
- title: Short descriptive title
- problem_statement: What you're trying to predict/analyze
- required_datasets: Array of dataset filenames (actual .csv files, not .summary.json)
- target_column: Column name to predict (must exist in target_dataset)
- target_dataset: Which dataset has the target column
- feature_columns: Array of column names to use as features
- feasibility_score: Number between 0 and 1
- feasibility_notes: Why this task is feasible

Optional fields (only for multi-dataset tasks):
- join_plan: {"datasets": [...], "join_keys": {"ds1": "col", "ds2": "col"}, "join_type": "inner"}

Category-specific fields:
- For forecasting: forecast_horizon, forecast_granularity, forecast_type, evaluation_metrics
- For classification: evaluation_metrics
- For regression: evaluation_metrics
- For clustering: evaluation_metrics

Template:
"""

    return explanation + "\n" + json.dumps(template, indent=2)


# Export tools list
STAGE2_TOOLS = [
    list_dataset_summaries,
    read_dataset_summary,
    explore_data_relationships,
    python_sandbox_stage2,
    evaluate_forecasting_feasibility,
    save_task_proposals,
    get_proposal_template,
]
