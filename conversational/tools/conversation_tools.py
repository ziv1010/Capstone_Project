"""
Conversation Tools: User Interaction

Tools for the conversational agent to interact with users and manage the pipeline.
"""

import json
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    DATA_DIR, SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR,
    STAGE3B_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR,
    STAGE4_OUT_DIR, STAGE5_OUT_DIR,
    CONVERSATION_STATE_DIR, DataPassingManager, logger
)
from code.utils import list_data_files, list_summary_files, load_dataframe


@tool
def get_available_data() -> str:
    """
    Get information about available datasets.

    Returns list of datasets with basic info for user reference.
    """
    try:
        files = list_data_files(DATA_DIR)

        if not files:
            return "No datasets found in the data directory."

        result = ["Available Datasets:"]
        for f in files:
            filepath = DATA_DIR / f
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                result.append(f"  - {f} ({size_mb:.2f} MB)")

        result.append(f"\nTotal: {len(files)} datasets")
        return "\n".join(result)

    except Exception as e:
        return f"Error listing data: {e}"


@tool
def get_summaries() -> str:
    """
    Get summaries of analyzed datasets (from Stage 1).

    Returns overview of what data is available and its characteristics.
    """
    try:
        summary_files = list_summary_files(SUMMARIES_DIR)

        if not summary_files:
            return "No dataset summaries available. Would you like me to analyze the datasets first?"

        result = ["Dataset Summaries:\n"]

        for sf in summary_files:
            try:
                summary = DataPassingManager.load_artifact(SUMMARIES_DIR / sf)
                data = summary.get('data', summary) if isinstance(summary, dict) else summary

                result.append(f"{data.get('filename', sf)}:")
                result.append(f"  - {data.get('n_rows', '?')} rows, {data.get('n_cols', '?')} columns")

                if data.get('has_datetime_column'):
                    result.append("  - Has datetime column (suitable for time series)")

                if data.get('has_target_candidates'):
                    result.append(f"  - Potential targets: {data.get('has_target_candidates')[:3]}")

                quality = data.get('data_quality_score')
                if quality:
                    result.append(f"  - Quality score: {quality:.1%}")

                result.append("")

            except Exception as e:
                result.append(f"{sf}: Error reading - {e}\n")

        return "\n".join(result)

    except Exception as e:
        return f"Error getting summaries: {e}"


@tool
def get_task_proposals() -> str:
    """
    Get proposed analytical tasks (from Stage 2).

    Returns list of tasks that can be executed on the data.
    """
    try:
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"

        if not proposals_path.exists():
            return "No task proposals available. Would you like me to generate some based on your data?"

        data = DataPassingManager.load_artifact(proposals_path)
        proposals = data.get('proposals', data) if isinstance(data, dict) else data

        if not proposals:
            return "No proposals found in file."

        result = ["Proposed Analysis Tasks:\n"]

        for p in proposals:
            result.append(f"{p.get('id')}: {p.get('title')}")
            result.append(f"  Category: {p.get('category')}")
            result.append(f"  Target: {p.get('target_column')}")
            result.append(f"  Feasibility: {p.get('feasibility_score', 'N/A')}")
            result.append(f"  Description: {p.get('problem_statement', 'N/A')[:200]}...")
            result.append("")

        result.append("Use 'run task TSK-XXX' to execute a specific task.")
        return "\n".join(result)

    except Exception as e:
        return f"Error getting proposals: {e}"


@tool
def check_pipeline_status() -> str:
    """
    Check the current status of the pipeline.

    Returns which stages have been completed and what's available.
    """
    try:
        status = {
            "Stage 1 (Summarization)": "not_run",
            "Stage 2 (Task Proposals)": "not_run",
            "Stage 3 (Execution Plan)": "not_run",
            "Stage 3B (Data Preparation)": "not_run",
            "Stage 3.5A (Method Proposals)": "not_run",
            "Stage 3.5B (Benchmarking)": "not_run",
            "Stage 4 (Execution)": "not_run",
            "Stage 5 (Visualization)": "not_run",
        }

        # Check Stage 1
        summaries = list(SUMMARIES_DIR.glob("*.summary.json"))
        if summaries:
            status["Stage 1 (Summarization)"] = f"completed ({len(summaries)} datasets)"

        # Check Stage 2
        if (STAGE2_OUT_DIR / "task_proposals.json").exists():
            status["Stage 2 (Task Proposals)"] = "completed"

        # Check Stage 3+
        plans = list(STAGE3_OUT_DIR.glob("PLAN-*.json"))
        if plans:
            plan_ids = [p.stem for p in plans]
            status["Stage 3 (Execution Plan)"] = f"completed ({plan_ids})"

            for plan_id in plan_ids:
                # Check downstream stages
                if (STAGE3B_OUT_DIR / f"prepared_{plan_id}.parquet").exists():
                    status["Stage 3B (Data Preparation)"] = f"completed ({plan_id})"

                if list(Path(STAGE3_5A_OUT_DIR).glob(f"method_proposal_{plan_id}.json")):
                    status["Stage 3.5A (Method Proposals)"] = f"completed ({plan_id})"

                if list(Path(STAGE3_5B_OUT_DIR).glob(f"tester_{plan_id}.json")):
                    status["Stage 3.5B (Benchmarking)"] = f"completed ({plan_id})"

                if (STAGE4_OUT_DIR / f"execution_result_{plan_id}.json").exists():
                    status["Stage 4 (Execution)"] = f"completed ({plan_id})"

                if list(Path(STAGE5_OUT_DIR).glob(f"visualization_report_{plan_id}.json")):
                    status["Stage 5 (Visualization)"] = f"completed ({plan_id})"

        result = ["Pipeline Status:\n"]
        for stage, state in status.items():
            icon = "Y" if "completed" in state else " "
            result.append(f"  [{icon}] {stage}: {state}")

        return "\n".join(result)

    except Exception as e:
        return f"Error checking status: {e}"


@tool
def evaluate_user_query(query: str) -> str:
    """
    Evaluate if a user's query is feasible with available data.

    Uses smart keyword extraction to find the BEST matching dataset.

    Args:
        query: User's natural language query

    Returns:
        Feasibility assessment with best dataset recommendation
    """
    try:
        import re

        result = ["=== Query Feasibility Analysis ===", f"Query: {query}\n"]

        # Step 1: Extract keywords from query (smart extraction)
        result.append("Step 1: Extracting Keywords from Query")
        query_lower = query.lower()

        # Extract ALL meaningful words (remove common stopwords)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'can',
                     'you', 'me', 'make', 'next', 'years', 'year', 'predict', 'prediction',
                     'forecast', 'forecasting'}

        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        result.append(f"  Extracted keywords: {keywords}")

        # Step 2: Check available data
        result.append("\nStep 2: Scanning Available Datasets")
        summary_files = list_summary_files(SUMMARIES_DIR)

        if not summary_files:
            result.append("  No data summaries available!")
            result.append("\nConclusion: INFEASIBLE - No data has been analyzed yet")
            result.append("Recommendation: Run Stage 1 to analyze available datasets")
            return "\n".join(result)

        result.append(f"  Found {len(summary_files)} datasets to analyze")

        # Step 3: Smart dataset matching
        result.append("\nStep 3: Finding Best Dataset Match")
        dataset_scores = []

        for sf in summary_files:
            try:
                summary = DataPassingManager.load_artifact(SUMMARIES_DIR / sf)
                data = summary.get('data', summary)

                dataset_name = data.get('filename', '')
                columns = data.get('columns', [])

                # Score this dataset based on keyword matches
                score = 0
                matched_keywords = []
                matched_columns = []

                # Check dataset name
                dataset_name_lower = dataset_name.lower()
                for kw in keywords:
                    if kw in dataset_name_lower:
                        score += 3  # Dataset name match is strong signal
                        matched_keywords.append(kw)

                # Check column names
                for col in columns:
                    col_name = col.get('name', '').lower()
                    col_type = col.get('logical_type', '')

                    for kw in keywords:
                        if kw in col_name:
                            score += 2  # Column name match
                            matched_keywords.append(kw)
                            matched_columns.append(col.get('name'))
                            break

                # Bonus for datetime columns (forecasting)
                if data.get('has_datetime_column'):
                    score += 1

                # Bonus for high quality data
                quality = data.get('data_quality_score', 0.5)
                score += quality

                if score > 0:
                    dataset_scores.append({
                        'dataset': dataset_name,
                        'score': score,
                        'matched_keywords': list(set(matched_keywords)),
                        'matched_columns': list(set(matched_columns)),
                        'has_datetime': data.get('has_datetime_column', False),
                        'n_rows': data.get('n_rows', 0),
                        'all_columns': [c.get('name') for c in columns]
                    })

            except Exception as e:
                logger.warning(f"Error analyzing {sf}: {e}")
                continue

        # Sort by score
        dataset_scores.sort(key=lambda x: x['score'], reverse=True)

        if dataset_scores:
            result.append(f"  Ranked {len(dataset_scores)} datasets by relevance:")
            for i, ds in enumerate(dataset_scores[:5], 1):
                result.append(f"    {i}. {ds['dataset']} (score: {ds['score']:.1f})")
                if ds['matched_keywords']:
                    result.append(f"       Matched keywords: {ds['matched_keywords']}")
                if ds['matched_columns']:
                    result.append(f"       Relevant columns: {ds['matched_columns'][:5]}")
        else:
            result.append("  No datasets matched the query keywords")

        # Step 4: Recommendation
        result.append("\nStep 4: Recommendation")

        if dataset_scores:
            best = dataset_scores[0]
            result.append(f"  ✅ FEASIBLE")
            result.append(f"  Best dataset: {best['dataset']}")
            result.append(f"  Rows: {best['n_rows']}")
            result.append(f"  Relevant columns: {best['matched_columns']}")
            result.append(f"  All columns: {best['all_columns'][:10]}")

            if best['has_datetime']:
                result.append("  ✅ Has temporal data - suitable for time series forecasting")

            result.append(f"\n  💡 Use this dataset in your task: {best['dataset']}")
        else:
            result.append("  ⚠️ UNCERTAIN - No clear matches")
            result.append("  Recommendation: Review available datasets manually")

        return "\n".join(result)

    except Exception as e:
        import traceback
        return f"Error evaluating query: {e}\n{traceback.format_exc()}"


@tool
def create_custom_task_from_query(
    query: str,
    dataset: str = None,
    target_column: str = None,
    date_column: str = None
) -> str:
    """
    Create a custom task proposal based on user's query.

    INTELLIGENTLY selects the best dataset and target column if not provided.
    AUTOMATICALLY parses forecast configuration from the query (e.g., "next 5 years").

    Args:
        query: User's forecasting query
        dataset: Dataset filename to use (optional - will auto-select if not provided)
        target_column: Column to predict (optional - will auto-select if not provided)
        date_column: Date column for temporal analysis (optional - will auto-detect)

    Returns:
        Generated task proposal with proper forecast configuration
    """
    try:
        import re
        from code.config import parse_forecast_config_from_query, get_task_appropriate_metrics

        result_messages = []

        # === STEP 1: SMART DATASET SELECTION ===
        if not dataset or not target_column:
            result_messages.append("🔍 Intelligently selecting best dataset for your query...")

            # Extract keywords from query
            query_lower = query.lower()
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'can',
                         'you', 'me', 'make', 'next', 'years', 'year', 'predict', 'prediction',
                         'forecast', 'forecasting'}

            words = re.findall(r'\b\w+\b', query_lower)
            keywords = [w for w in words if w not in stopwords and len(w) > 2]

            result_messages.append(f"  Extracted keywords: {keywords}")

            # Scan all datasets
            summary_files = list_summary_files(SUMMARIES_DIR)
            if not summary_files:
                return "❌ No dataset summaries found. Run Stage 1 first."

            dataset_scores = []

            for sf in summary_files:
                try:
                    summary = DataPassingManager.load_artifact(SUMMARIES_DIR / sf)
                    data = summary.get('data', summary)

                    dataset_name = data.get('filename', '')
                    columns = data.get('columns', [])

                    # Score this dataset
                    score = 0
                    matched_keywords = []
                    matched_columns = []

                    # Check dataset name
                    dataset_name_lower = dataset_name.lower()
                    for kw in keywords:
                        if kw in dataset_name_lower:
                            score += 3
                            matched_keywords.append(kw)

                    # Check column names
                    for col in columns:
                        col_name = col.get('name', '').lower()
                        for kw in keywords:
                            if kw in col_name:
                                score += 2
                                matched_keywords.append(kw)
                                matched_columns.append(col.get('name'))
                                break

                    # Bonus for datetime columns
                    has_datetime = data.get('has_datetime_column', False)
                    if has_datetime:
                        score += 1

                    # Bonus for data quality
                    quality = data.get('data_quality_score', 0.5)
                    score += quality

                    if score > 0:
                        dataset_scores.append({
                            'dataset': dataset_name,
                            'score': score,
                            'matched_keywords': list(set(matched_keywords)),
                            'matched_columns': list(set(matched_columns)),
                            'has_datetime': has_datetime,
                            'columns': columns
                        })

                except Exception as e:
                    logger.warning(f"Error analyzing {sf}: {e}")
                    continue

            # Sort by score
            dataset_scores.sort(key=lambda x: x['score'], reverse=True)

            if not dataset_scores:
                return "❌ No datasets matched your query. Try being more specific."

            # Select best dataset
            best = dataset_scores[0]
            dataset = best['dataset']

            result_messages.append(f"  ✅ Selected dataset: {dataset} (score: {best['score']:.1f})")
            result_messages.append(f"     Matched keywords: {best['matched_keywords']}")

            # === STEP 2: SMART TARGET COLUMN SELECTION ===
            if not target_column:
                # Find best target column from matched columns
                if best['matched_columns']:
                    # Prefer numeric columns
                    numeric_matches = []
                    for col_name in best['matched_columns']:
                        col_info = next((c for c in best['columns'] if c.get('name') == col_name), None)
                        if col_info and col_info.get('logical_type') in ['integer', 'real', 'categorical_numeric']:
                            numeric_matches.append(col_name)

                    if numeric_matches:
                        target_column = numeric_matches[0]
                    else:
                        target_column = best['matched_columns'][0]

                    result_messages.append(f"  ✅ Selected target column: {target_column}")
                else:
                    return f"❌ Could not identify target column in {dataset}. Please specify manually."

            # === STEP 3: AUTO-DETECT DATE COLUMN ===
            if not date_column and best['has_datetime']:
                for col in best['columns']:
                    col_name = col.get('name', '')
                    if any(keyword in col_name.lower() for keyword in ['year', 'date', 'time', 'period']):
                        date_column = col_name
                        result_messages.append(f"  ✅ Detected date column: {date_column}")
                        break

        # === STEP 4: GENERATE TASK PROPOSAL ===
        import time
        task_id = f"TSK-{int(time.time()) % 10000:04d}"

        # Parse forecast configuration from query
        forecast_config = parse_forecast_config_from_query(query)

        # Get appropriate metrics
        metrics = get_task_appropriate_metrics("forecasting", query)

        proposal = {
            "id": task_id,
            "category": "forecasting",
            "title": f"Custom Forecasting: {target_column}",
            "problem_statement": query,
            "required_datasets": [dataset],
            "target_column": target_column,
            "target_dataset": dataset,
            "feature_columns": [],
            "validation_plan": {
                "train_fraction": 0.7,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "split_strategy": "temporal" if date_column else "random",
                "date_column": date_column
            },
            "feasibility_score": 0.7,
            "feasibility_notes": f"Custom task created from user query: {query}",

            # DYNAMIC forecast configuration
            "forecast_horizon": forecast_config["forecast_horizon"],
            "forecast_granularity": forecast_config["forecast_granularity"],
            "forecast_type": forecast_config["forecast_type"],

            # DYNAMIC metrics
            "evaluation_metrics": metrics
        }

        # Save to proposals
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if proposals_path.exists():
            existing = DataPassingManager.load_artifact(proposals_path)
            proposals = existing.get('proposals', [])
        else:
            proposals = []

        proposals.append(proposal)

        DataPassingManager.save_artifact(
            data={"proposals": proposals},
            output_dir=STAGE2_OUT_DIR,
            filename="task_proposals.json"
        )

        result_messages.extend([
            "",
            "=== Custom Task Created ===",
            f"Task ID: {task_id}",
            f"Dataset: {dataset}",
            f"Target: {target_column}",
            f"Date Column: {date_column or 'None (will use index)'}",
            "",
            "Forecast Configuration:",
            f"  Horizon: {forecast_config['forecast_horizon']} {forecast_config['forecast_granularity']}(s)",
            f"  Type: {forecast_config['forecast_type']}",
            f"  Metrics: {', '.join(metrics)}",
            "",
            f"✅ Proposal saved. Use: 'run task {task_id}'"
        ])

        return "\n".join(result_messages)

    except Exception as e:
        import traceback
        return f"❌ Error creating task: {e}\n{traceback.format_exc()}"


@tool
def get_execution_results(plan_id: str = None) -> str:
    """
    Get results from executed tasks.

    Args:
        plan_id: Specific plan ID. If not provided, shows all results.

    Returns:
        Execution results summary
    """
    try:
        if plan_id:
            result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
            if not result_path.exists():
                return f"No results found for {plan_id}"

            result_data = DataPassingManager.load_artifact(result_path)

            result = [
                f"=== Execution Results: {plan_id} ===",
                f"Status: {result_data.get('status')}",
                f"Summary: {result_data.get('summary')}",
                "",
                "Metrics:",
            ]

            for metric, value in result_data.get('metrics', {}).items():
                result.append(f"  - {metric}: {value}")

            return "\n".join(result)

        else:
            # List all results
            results = list(STAGE4_OUT_DIR.glob("execution_result_*.json"))

            if not results:
                return "No execution results available."

            output = ["=== All Execution Results ===\n"]

            for r in results:
                try:
                    data = DataPassingManager.load_artifact(r)
                    plan_id = r.stem.replace("execution_result_", "")
                    output.append(f"{plan_id}:")
                    output.append(f"  Status: {data.get('status')}")
                    output.append(f"  Metrics: {data.get('metrics', {})}")
                    output.append("")
                except:
                    continue

            return "\n".join(output)

    except Exception as e:
        return f"Error getting results: {e}"


@tool
def get_visualizations(plan_id: str = None) -> str:
    """
    Get visualization reports and plot locations.

    Args:
        plan_id: Specific plan ID. If not provided, shows all.

    Returns:
        Visualization report
    """
    try:
        if plan_id:
            report_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json"
            plots = list(STAGE5_OUT_DIR.glob(f"{plan_id}_*.png"))

            result = [f"=== Visualizations: {plan_id} ===\n"]

            if plots:
                result.append("Generated Plots:")
                for p in plots:
                    result.append(f"  - {p.name}")

            if report_path.exists():
                report = DataPassingManager.load_artifact(report_path)
                result.append(f"\nInsights:")
                for insight in report.get('insights', []):
                    result.append(f"  - {insight}")

            return "\n".join(result)

        else:
            # List all visualizations
            reports = list(STAGE5_OUT_DIR.glob("visualization_report_*.json"))
            plots = list(STAGE5_OUT_DIR.glob("*.png"))

            result = ["=== All Visualizations ===\n"]
            result.append(f"Total plots: {len(plots)}")
            result.append(f"Reports: {len(reports)}")

            if plots:
                result.append("\nPlot files:")
                for p in plots[:20]:  # First 20
                    result.append(f"  - {p.name}")

            return "\n".join(result)

    except Exception as e:
        return f"Error getting visualizations: {e}"


# Export tools list
CONVERSATION_TOOLS = [
    get_available_data,
    get_summaries,
    get_task_proposals,
    check_pipeline_status,
    evaluate_user_query,
    create_custom_task_from_query,
    get_execution_results,
    get_visualizations,
]
