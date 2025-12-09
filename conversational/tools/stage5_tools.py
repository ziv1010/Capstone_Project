"""
Stage 5 Tools: Visualization

Tools for creating visualizations and generating insights.
Uses ReAct framework to understand task context and generate meaningful outputs.
"""

import json
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    STAGE3_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE5_WORKSPACE,
    STAGE2_OUT_DIR, STAGE3_5A_OUT_DIR, STAGE3_5B_OUT_DIR,
    DataPassingManager, logger
)
from code.utils import load_dataframe


# ============================================================================
# ReAct FRAMEWORK TOOLS
# ============================================================================

@tool
def record_thought_stage5(thought: str) -> str:
    """
    Record your reasoning or thought process (ReAct framework).
    
    Use this to:
    - Understand the original task goal before visualizing
    - Plan which visualizations will best answer the task
    - Reason about what insights would be most valuable
    
    Args:
        thought: Your reasoning or thought
        
    Returns:
        Confirmation of recorded thought
    """
    logger.info(f"[THOUGHT] {thought}")
    return f"[THOUGHT RECORDED] {thought}"


@tool
def get_task_context(plan_id: str) -> str:
    """
    Get the original task goal and context to inform visualization decisions.
    
    Loads the task proposal, execution plan, and method details to understand
    what question the user was trying to answer.
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
    
    Returns:
        Task context including goal, problem statement, and method used
    """
    try:
        context = [f"=== Task Context for {plan_id} ===\n"]
        
        # Extract task ID from plan ID
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id
        
        # 1. Load Stage 2 task proposals to get original goal
        proposals_path = STAGE2_OUT_DIR / "task_proposals.json"
        if proposals_path.exists():
            proposals_data = DataPassingManager.load_artifact(proposals_path)
            proposals = proposals_data.get('proposals', [])
            for proposal in proposals:
                if proposal.get('id') == task_id:
                    context.append("ORIGINAL TASK:")
                    context.append(f"  Title: {proposal.get('title', 'N/A')}")
                    context.append(f"  Category: {proposal.get('category', 'N/A')}")
                    context.append(f"  Problem: {proposal.get('problem_statement', 'N/A')}")
                    context.append(f"  Target Column: {proposal.get('target_column', 'N/A')}")
                    context.append("")
                    break
        
        # 2. Load Stage 3 execution plan for goal
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        if plan_path.exists():
            plan = DataPassingManager.load_artifact(plan_path)
            context.append("EXECUTION PLAN:")
            context.append(f"  Goal: {plan.get('goal', 'N/A')}")
            context.append(f"  Target: {plan.get('target_column', 'N/A')}")
            context.append(f"  Model Types: {plan.get('expected_model_types', [])}")
            context.append("")
        
        # 3. Load Stage 3.5B tester output for method used
        tester_path = STAGE3_5B_OUT_DIR / f"tester_{plan_id}.json"
        if tester_path.exists():
            tester = DataPassingManager.load_artifact(tester_path)
            context.append("SELECTED METHOD:")
            context.append(f"  Method: {tester.get('selected_method_name', 'N/A')}")
            context.append(f"  Rationale: {tester.get('selection_rationale', 'N/A')}")
            context.append("")
        
        # 4. Load Stage 4 execution result for metrics
        result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
        if result_path.exists():
            result = DataPassingManager.load_artifact(result_path)
            metrics = result.get('metrics', {})
            context.append("EXECUTION RESULTS:")
            context.append(f"  Status: {result.get('status', 'N/A')}")
            context.append(f"  Summary: {result.get('summary', 'N/A')}")
            if metrics:
                context.append("  Metrics:")
                for k, v in metrics.items():
                    context.append(f"    - {k}: {v}")
            context.append("")
        
        context.append("--- Use this context to create visualizations that answer the original task ---")
        return "\n".join(context)
        
    except Exception as e:
        return f"Error loading task context: {e}"


@tool
def generate_task_answer(
    plan_id: str,
    key_findings: str,
    answer_to_task: str,
    recommendations: str
) -> str:
    """
    Generate a comprehensive answer to the original task question.
    
    This should be called after analyzing results and creating visualizations.
    It produces a human-readable summary that answers the user's original question.
    
    Args:
        plan_id: Plan ID
        key_findings: Main findings from the analysis (bullet points)
        answer_to_task: Direct answer to the original task question
        recommendations: Recommendations based on results
    
    Returns:
        Formatted task answer ready to be saved
    """
    try:
        # Load task context for title
        task_title = "Analysis Results"
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        if plan_path.exists():
            plan = DataPassingManager.load_artifact(plan_path)
            task_title = plan.get('goal', task_title)
        
        # Load metrics
        metrics_summary = ""
        result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
        if result_path.exists():
            result = DataPassingManager.load_artifact(result_path)
            metrics = result.get('metrics', {})
            if metrics:
                metrics_lines = [f"  • {k.upper()}: {v:.4f}" if isinstance(v, float) else f"  • {k.upper()}: {v}" 
                                for k, v in metrics.items()]
                metrics_summary = "\n".join(metrics_lines)
        
        # Format the answer
        answer = f"""
═══════════════════════════════════════════════════════════════
TASK: {task_title}
═══════════════════════════════════════════════════════════════

📊 KEY FINDINGS:
{key_findings}

✅ ANSWER TO TASK:
{answer_to_task}

📈 MODEL PERFORMANCE:
{metrics_summary if metrics_summary else '  No metrics available'}

💡 RECOMMENDATIONS:
{recommendations}

═══════════════════════════════════════════════════════════════
"""
        
        # Save to file
        answer_path = STAGE5_OUT_DIR / f"task_answer_{plan_id}.txt"
        with open(answer_path, 'w') as f:
            f.write(answer)
        
        logger.info(f"Task answer saved to {answer_path}")
        return f"Task answer generated and saved to: {answer_path}\n\n{answer}"
        
    except Exception as e:
        return f"Error generating task answer: {e}"


@tool
def load_historical_data(plan_id: str) -> str:
    """
    Load prepared historical data from Stage 3b for showing trends.

    This provides the complete historical dataset that was used for training,
    allowing you to show the full timeline in visualizations.

    Args:
        plan_id: Plan ID

    Returns:
        Summary of historical data structure and sample values
    """
    try:
        # Construct path to prepared data
        prepared_path = Path(str(STAGE4_OUT_DIR).replace('stage4_out', 'stage3b_data_prep')) / f"prepared_{plan_id}.parquet"

        if not prepared_path.exists():
            return f"Historical data not found at: {prepared_path}"

        hist_df = pd.read_parquet(prepared_path)

        result = [f"=== Historical Data: {plan_id} ===\n"]
        result.append(f"Shape: {hist_df.shape}")
        result.append(f"Columns: {list(hist_df.columns)}")
        result.append("")

        # Identify year columns
        year_cols = [c for c in hist_df.columns if any(prefix in c for prefix in ['Area-', 'Production-', 'Yield-'])]

        if year_cols:
            result.append("Time series columns found:")
            for col in sorted(year_cols)[:10]:  # Show first 10
                result.append(f"  - {col}: min={hist_df[col].min():.2f}, max={hist_df[col].max():.2f}, sum={hist_df[col].sum():.2f}")
            if len(year_cols) > 10:
                result.append(f"  ... and {len(year_cols)-10} more")
            result.append("")

        # Show how to aggregate
        result.append("Usage example for trend visualization:")
        result.append("```python")
        result.append("# Aggregate by year to show overall trend")
        result.append("year_cols = [c for c in hist_df.columns if 'Area-' in c]")
        result.append("years = [c.split('-')[-1] for c in sorted(year_cols)]")
        result.append("values = [hist_df[c].sum() for c in sorted(year_cols)]")
        result.append("ax.plot(years, values, 'o-', label='Historical', linewidth=3)")
        result.append("```")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading historical data: {e}"


@tool
def load_execution_results(plan_id: str = None) -> str:
    """
    Load Stage 4 execution results for visualization.

    Args:
        plan_id: Plan ID. If not provided, loads most recent.

    Returns:
        Summary of available data for visualization
    """
    try:
        if plan_id:
            result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
            predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        else:
            results = list(STAGE4_OUT_DIR.glob("execution_result_*.json"))
            if not results:
                return "No execution results found. Run Stage 4 first."
            result_path = max(results, key=lambda p: p.stat().st_mtime)
            plan_id = result_path.stem.replace("execution_result_", "")
            predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"

        result = [f"=== Visualization Data: {plan_id} ===\n"]

        # Load execution result
        exec_result = DataPassingManager.load_artifact(result_path)
        result.append(f"Status: {exec_result.get('status')}")
        result.append(f"Metrics: {exec_result.get('metrics', {})}")

        # Load predictions
        if predictions_path.exists():
            df = pd.read_parquet(predictions_path)
            result.append(f"\nData Shape: {df.shape}")
            result.append(f"Columns: {list(df.columns)}")

            # Identify column types
            result.append("\nColumn Analysis:")
            for col in df.columns:
                dtype = df[col].dtype
                n_unique = df[col].nunique()
                result.append(f"  {col}: {dtype} ({n_unique} unique values)")
        else:
            result.append("\nPredictions file not found!")

        return "\n".join(result)

    except Exception as e:
        return f"Error loading results: {e}"


@tool
def analyze_data_columns(plan_id: str) -> str:
    """
    Analyze columns in the results data for visualization planning.

    Categorizes columns as: GIVEN (input), PREDICTED (output), ENGINEERED (features),
    DATETIME, CATEGORICAL.

    Args:
        plan_id: Plan ID

    Returns:
        Column categorization for visualization
    """
    try:
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        if not predictions_path.exists():
            return f"Results file not found: {predictions_path}"

        df = pd.read_parquet(predictions_path)

        # Load plan for context
        plan_path = STAGE3_OUT_DIR / f"{plan_id}.json"
        plan = DataPassingManager.load_artifact(plan_path) if plan_path.exists() else {}

        target_col = plan.get('target_column', '')
        date_col = plan.get('date_column', '')

        result = ["=== Column Analysis for Visualization ===\n"]

        categories = {
            'DATETIME': [],
            'TARGET_ACTUAL': [],
            'PREDICTED': [],
            'FEATURES': [],
            'CATEGORICAL': [],
            'OTHER': []
        }

        for col in df.columns:
            col_lower = col.lower()

            # Check datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]) or col == date_col:
                categories['DATETIME'].append(col)
            # Check prediction columns
            elif 'predict' in col_lower or 'forecast' in col_lower or 'fitted' in col_lower:
                categories['PREDICTED'].append(col)
            # Check actual/target
            elif col == target_col or 'actual' in col_lower or 'target' in col_lower:
                categories['TARGET_ACTUAL'].append(col)
            # Check categorical
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                categories['CATEGORICAL'].append(col)
            # Features and other numeric
            elif pd.api.types.is_numeric_dtype(df[col]):
                if 'lag' in col_lower or 'rolling' in col_lower or 'feature' in col_lower:
                    categories['FEATURES'].append(col)
                else:
                    categories['OTHER'].append(col)
            else:
                categories['OTHER'].append(col)

        for cat, cols in categories.items():
            if cols:
                result.append(f"{cat}:")
                for c in cols:
                    result.append(f"  - {c}")
                result.append("")

        # Visualization recommendations
        result.append("--- Recommended Visualizations ---")

        if categories['PREDICTED'] and categories['TARGET_ACTUAL']:
            result.append("1. Actual vs Predicted scatter plot")
            result.append("2. Residual analysis (histogram and time series)")

        if categories['DATETIME'] and (categories['PREDICTED'] or categories['TARGET_ACTUAL']):
            result.append("3. Time series plot with predictions")

        if categories['CATEGORICAL']:
            result.append("4. Predictions by category (box plots)")

        if categories['FEATURES']:
            result.append("5. Feature importance or correlation heatmap")

        return "\n".join(result)

    except Exception as e:
        return f"Error analyzing columns: {e}"


@tool
def plan_visualization(
    plot_type: str,
    purpose: str,
    columns_to_use: str,
    title: str
) -> str:
    """
    Plan a visualization before creating it (ReAct framework).

    Args:
        plot_type: Type of plot (scatter, line, bar, histogram, heatmap, boxplot)
        purpose: What story this visualization tells
        columns_to_use: Comma-separated column names
        title: Plot title

    Returns:
        Visualization plan
    """
    plan = {
        "plot_type": plot_type,
        "purpose": purpose,
        "columns": [c.strip() for c in columns_to_use.split(',')],
        "title": title,
        "status": "planned"
    }

    result = [
        "=== Visualization Plan ===",
        f"Type: {plot_type}",
        f"Purpose: {purpose}",
        f"Columns: {plan['columns']}",
        f"Title: {title}",
        "",
        "Ready to create with create_plot tool."
    ]

    return "\n".join(result)


@tool
def create_plot(
    plan_id: str,
    plot_code: str,
    filename: str,
    description: str
) -> str:
    """
    Create and save a visualization.

    Args:
        plan_id: Plan ID for loading data
        plot_code: Python code that creates the plot using matplotlib
        filename: Output filename (e.g., 'actual_vs_predicted.png')
        description: Description of what the plot shows

    Returns:
        Confirmation with saved path
    """
    import sys
    from io import StringIO

    try:
        # Load data
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        if not predictions_path.exists():
            return f"Results file not found: {predictions_path}"

        df = pd.read_parquet(predictions_path)

        # Set up namespace
        namespace = {
            'pd': pd,
            'np': np,
            'df': df,
            'Path': Path,
            'STAGE5_OUT_DIR': STAGE5_OUT_DIR,
        }

        # Import visualization libraries
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        namespace['plt'] = plt
        namespace['sns'] = sns

        # Execute plotting code
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exec(plot_code, namespace)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Save figure
        output_path = STAGE5_OUT_DIR / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return f"Plot saved: {output_path}\nDescription: {description}"

    except Exception as e:
        import traceback
        return f"Error creating plot: {e}\n{traceback.format_exc()}"


@tool
def create_standard_plots(plan_id: str) -> str:
    """
    Create a standard set of visualizations for the results.

    Creates: actual vs predicted, residuals, time series, and distribution plots.

    Args:
        plan_id: Plan ID

    Returns:
        Summary of created plots
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        if not predictions_path.exists():
            return f"Results file not found: {predictions_path}"

        df = pd.read_parquet(predictions_path)

        # Find prediction and actual columns
        pred_cols = [c for c in df.columns if 'predict' in c.lower() or 'forecast' in c.lower()]
        actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'target' in c.lower()]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or 'year' in c.lower()]

        # Check if this is a forecasting task
        has_prediction_type = 'prediction_type' in df.columns
        is_forecasting = has_prediction_type and 'forecast' in df['prediction_type'].unique()

        created_plots = []

        # === FORECAST TREND PLOT (for forecasting tasks) ===
        if is_forecasting and date_cols and pred_cols and actual_cols:
            date_col = date_cols[0]
            pred_col, actual_col = pred_cols[0], actual_cols[0]

            # Separate test and forecast data
            test_df = df[df['prediction_type'] == 'test'].copy()
            forecast_df = df[df['prediction_type'] == 'forecast'].copy()

            fig, ax = plt.subplots(figsize=(16, 8))

            # Plot historical actuals
            ax.plot(test_df[date_col], test_df[actual_col],
                    'o-', label='Historical Actual', linewidth=2, markersize=8, color='blue')

            # Plot test predictions
            ax.plot(test_df[date_col], test_df[pred_col],
                    's--', label='Test Predictions', linewidth=2, markersize=6, alpha=0.7, color='green')

            # Plot future forecasts
            ax.plot(forecast_df[date_col], forecast_df[pred_col],
                    'D:', label='Future Forecasts', linewidth=2, markersize=8, color='red')

            # Add vertical line separating history from forecast
            if len(test_df) > 0:
                last_test_date = test_df[date_col].iloc[-1]
                ax.axvline(x=last_test_date, color='gray', linestyle='--',
                          alpha=0.5, linewidth=2, label='Forecast Start')

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Forecast Trend: Historical + Future Predictions', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(STAGE5_OUT_DIR / f'{plan_id}_forecast_trend.png', dpi=150)
            plt.close()
            created_plots.append('forecast_trend.png')

        # 1. Actual vs Predicted scatter (test set only for forecasting tasks)
        if pred_cols and actual_cols:
            pred_col, actual_col = pred_cols[0], actual_cols[0]

            # For forecasting tasks, only use test set (forecasts have no actuals)
            plot_df = df[df['prediction_type'] == 'test'].copy() if is_forecasting else df.copy()

            # Filter out NaN actuals
            plot_df = plot_df.dropna(subset=[actual_col, pred_col])

            if len(plot_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(plot_df[actual_col], plot_df[pred_col], alpha=0.6, s=100)
                ax.plot([plot_df[actual_col].min(), plot_df[actual_col].max()],
                       [plot_df[actual_col].min(), plot_df[actual_col].max()],
                       'r--', label='Perfect Prediction', linewidth=2)
                ax.set_xlabel('Actual', fontsize=12)
                ax.set_ylabel('Predicted', fontsize=12)
                title = 'Test Set Accuracy: Actual vs Predicted' if is_forecasting else 'Actual vs Predicted'
                ax.set_title(title, fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(STAGE5_OUT_DIR / f'{plan_id}_actual_vs_predicted.png', dpi=150)
                plt.close()
                created_plots.append('actual_vs_predicted.png')

            # 2. Residuals histogram (test set only for forecasting tasks)
            residuals_df = df[df['prediction_type'] == 'test'].copy() if is_forecasting else df.copy()
            residuals_df = residuals_df.dropna(subset=[actual_col, pred_col])
            residuals = residuals_df[actual_col] - residuals_df[pred_col]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Residual')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
            plt.tight_layout()
            plt.savefig(STAGE5_OUT_DIR / f'{plan_id}_residuals_histogram.png', dpi=150)
            plt.close()
            created_plots.append('residuals_histogram.png')

        # 3. Time series plot
        if date_cols and (pred_cols or actual_cols):
            date_col = date_cols[0]
            df_sorted = df.sort_values(date_col)

            fig, ax = plt.subplots(figsize=(14, 6))

            if actual_cols:
                ax.plot(df_sorted[date_col], df_sorted[actual_cols[0]], label='Actual', alpha=0.7)
            if pred_cols:
                ax.plot(df_sorted[date_col], df_sorted[pred_cols[0]], label='Predicted', alpha=0.7)

            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title('Time Series: Actual vs Predicted')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(STAGE5_OUT_DIR / f'{plan_id}_time_series.png', dpi=150)
            plt.close()
            created_plots.append('time_series.png')

        # 4. Error distribution by time
        if date_cols and pred_cols and actual_cols:
            date_col = date_cols[0]
            df_sorted = df.sort_values(date_col)
            df_sorted['residual'] = df_sorted[actual_cols[0]] - df_sorted[pred_cols[0]]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df_sorted[date_col], df_sorted['residual'], alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Date')
            ax.set_ylabel('Residual')
            ax.set_title('Residuals Over Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(STAGE5_OUT_DIR / f'{plan_id}_residuals_time.png', dpi=150)
            plt.close()
            created_plots.append('residuals_time.png')

        result = [
            f"=== Standard Plots Created: {plan_id} ===",
            f"Location: {STAGE5_OUT_DIR}",
            "",
            "Created plots:",
        ]
        for p in created_plots:
            result.append(f"  - {p}")

        if not created_plots:
            result.append("  No plots could be created - check column names")

        return "\n".join(result)

    except Exception as e:
        import traceback
        return f"Error creating plots: {e}\n{traceback.format_exc()}"


@tool
def generate_insights(plan_id: str) -> str:
    """
    Generate insights from the execution results.

    Args:
        plan_id: Plan ID

    Returns:
        List of insights and observations
    """
    try:
        # Load execution result
        result_path = STAGE4_OUT_DIR / f"execution_result_{plan_id}.json"
        predictions_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"

        if not result_path.exists() or not predictions_path.exists():
            return "Results not found."

        exec_result = DataPassingManager.load_artifact(result_path)
        df = pd.read_parquet(predictions_path)

        insights = ["=== Generated Insights ===\n"]

        # Metrics insights
        metrics = exec_result.get('metrics', {})
        if metrics:
            insights.append("Model Performance:")
            if 'mae' in metrics:
                insights.append(f"  - Average prediction error (MAE): {metrics['mae']:.4f}")
            if 'rmse' in metrics:
                insights.append(f"  - Root mean squared error: {metrics['rmse']:.4f}")
            if 'mape' in metrics:
                insights.append(f"  - Percentage error (MAPE): {metrics['mape']:.2f}%")
            if 'r2' in metrics:
                r2 = metrics['r2']
                insights.append(f"  - R-squared: {r2:.4f} ({r2*100:.1f}% variance explained)")

        # Find prediction columns
        pred_cols = [c for c in df.columns if 'predict' in c.lower()]
        actual_cols = [c for c in df.columns if 'actual' in c.lower()]

        if pred_cols and actual_cols:
            pred_col, actual_col = pred_cols[0], actual_cols[0]
            residuals = df[actual_col] - df[pred_col]

            insights.append("\nError Analysis:")
            insights.append(f"  - Mean residual: {residuals.mean():.4f}")
            insights.append(f"  - Residual std: {residuals.std():.4f}")

            if residuals.mean() > residuals.std() * 0.1:
                insights.append("  - Model tends to UNDERPREDICT (positive bias)")
            elif residuals.mean() < -residuals.std() * 0.1:
                insights.append("  - Model tends to OVERPREDICT (negative bias)")
            else:
                insights.append("  - Model predictions are well-centered")

            # Outliers
            outlier_threshold = 3 * residuals.std()
            n_outliers = (abs(residuals) > outlier_threshold).sum()
            insights.append(f"  - Outliers (>3 std): {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

        insights.append("\nData Summary:")
        insights.append(f"  - Total predictions: {len(df)}")
        insights.append(f"  - Date range: {df.select_dtypes(include=['datetime64']).min().min()} to {df.select_dtypes(include=['datetime64']).max().max()}")

        return "\n".join(insights)

    except Exception as e:
        return f"Error generating insights: {e}"


@tool
def save_visualization_report(report_json: str) -> str:
    """
    Save the visualization report.

    Args:
        report_json: JSON string with VisualizationReport structure

    Returns:
        Confirmation with saved path
    """
    try:
        report = json.loads(report_json)

        required = ['plan_id', 'visualizations', 'summary']
        missing = [f for f in required if f not in report]
        if missing:
            return f"Error: Missing required fields: {missing}"

        # Handle insights as string (convert to list)
        if 'insights' in report and isinstance(report['insights'], str):
            # Split by newlines or numbered lines
            insights_str = report['insights']
            insights_list = [line.strip() for line in insights_str.split('\n') if line.strip()]
            # Remove numbering if present
            cleaned = []
            for line in insights_list:
                if line and line[0].isdigit() and '. ' in line[:4]:
                    cleaned.append(line.split('. ', 1)[1])
                else:
                    cleaned.append(line)
            report['insights'] = cleaned

        # Auto-derive filename from filepath for each visualization
        for viz in report.get('visualizations', []):
            if 'filepath' in viz and not viz.get('filename'):
                viz['filename'] = Path(viz['filepath']).name

        plan_id = report['plan_id']
        filename = f"visualization_report_{plan_id}.json"

        output_path = DataPassingManager.save_artifact(
            data=report,
            output_dir=STAGE5_OUT_DIR,
            filename=filename,
            metadata={"stage": "stage5", "type": "visualization_report"}
        )

        return f"Visualization report saved to: {output_path}"

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"
    except Exception as e:
        return f"Error saving report: {e}"


# Export tools list
STAGE5_TOOLS = [
    # ReAct framework tools
    record_thought_stage5,
    get_task_context,
    generate_task_answer,
    # Core visualization tools
    load_historical_data,
    load_execution_results,
    analyze_data_columns,
    plan_visualization,
    create_plot,
    create_standard_plots,
    generate_insights,
    save_visualization_report,
]
