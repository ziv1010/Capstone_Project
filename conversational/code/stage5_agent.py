"""
Stage 5 Agent: Visualization

This agent creates visualizations and generates insights from the results.
"""

import json
from typing import Dict, Any, Optional, Annotated
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    STAGE3_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE5_WORKSPACE,
    SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS, STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.models import VisualizationReport, PipelineState
from tools.stage5_tools import STAGE5_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage5State(BaseModel):
    """State for Stage 5 agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    data_loaded: bool = False
    plots_created: list = []
    insights_generated: list = []
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE5_SYSTEM_PROMPT = """You are a Visualization Agent using the ReAct framework.

## YOUR WORKFLOW (ReAct: THOUGHT → ACTION → OBSERVATION)

### Step 1: UNDERSTAND THE TASK (Required First!)
Before creating ANY visualizations:
1. Call `get_task_context` to understand the ORIGINAL GOAL of this analysis
2. Use `record_thought_stage5` to reason about what visualizations would ANSWER the task

### Step 2: ANALYZE THE DATA
1. Call `load_execution_results` to see available data
2. Call `analyze_data_columns` to understand column types
3. Use `record_thought_stage5` to plan visualizations that address the task goal

### Step 3: CREATE CUSTOM TASK-APPROPRIATE VISUALIZATIONS
**CRITICAL**: You must create visualizations FROM SCRATCH using the `create_plot` tool!

#### Key Principles for Forecasting/Prediction Tasks:
1. **ALWAYS show historical context**: Load the prepared data from stage3b to show past known values
   - Path pattern: replace 'stage4_out' with 'stage3b_data_prep' in results path
   - This allows viewers to see the trend pattern leading up to predictions
2. **Show the complete timeline**: Historical → Test → Forecast
3. **Distinguish different data types** with colors/markers:
   - Historical/training data (what the model learned from)
   - Test predictions (how well the model performs on known data)
   - Future forecasts (what we're predicting)
4. **Make it publication-quality**: Large fonts, clear labels, legends, grid lines
5. **Aggregate appropriately**: If data has multiple categories (crops, regions, etc.),
   aggregate to show overall trends

#### Example Visualization Strategy for Forecasting:
```python
# Load prepared data to get ALL historical values
prepared_path = Path(str(STAGE4_OUT_DIR).replace('stage4_out', 'stage3b_data_prep')) / f"prepared_{plan_id}.parquet"
hist_df = pd.read_parquet(prepared_path)

# Extract all year columns (e.g., Area-2020-21, Area-2021-22, etc.)
year_cols = [c for c in hist_df.columns if 'Area-' in c or 'Production-' in c]

# Aggregate by year to show overall trend
years = []
values = []
for col in sorted(year_cols):
    year = col.split('-')[-1]
    years.append(year)
    values.append(hist_df[col].sum())  # or mean(), depending on context

# Create comprehensive plot
fig, ax = plt.subplots(figsize=(18, 10))
ax.plot(years, values, 'o-', label='Historical (Training)', linewidth=3, markersize=10)
# ... then add test predictions and future forecasts
```

Use `create_plot` to generate custom Python code for each visualization.

#### Types of Visualizations to Create:
1. **Main Forecast/Trend Plot** (MOST IMPORTANT for forecasting):
   - Show complete historical timeline with past known values
   - Overlay test predictions to show model accuracy
   - Extend to future forecasts
   - Use vertical lines or shading to separate past/test/future

2. **Model Accuracy Plots**:
   - Actual vs Predicted scatter (test set only, with perfect prediction line)
   - Show how close predictions are to reality

3. **Error Analysis**:
   - Residuals histogram (distribution of errors)
   - Residuals over time (are errors random or systematic?)
   - Box plots by category if applicable

4. **Additional Context** (if relevant):
   - Predictions by category (if multiple categories exist)
   - Feature importance or correlation heatmaps
   - Confidence intervals or prediction ranges (if available)

### Step 4: GENERATE INSIGHTS & TASK ANSWER
1. Call `generate_insights` to extract key findings
2. Call `generate_task_answer` with:
   - key_findings: Main discoveries from the analysis
   - answer_to_task: Direct answer to the original task question
   - recommendations: What actions to take based on results

### Step 5: SAVE THE REPORT
Call `save_visualization_report` with a JSON containing:
- plan_id: The plan ID
- visualizations: List of plots with {filepath, plot_type, title, description, columns_used}
- insights: Key findings
- summary: Overall assessment
- task_answer: The answer generated in Step 4

## IMPORTANT RULES
1. ALWAYS start with `get_task_context` to understand what we're trying to answer
2. ALWAYS create custom visualizations using `create_plot` - DO NOT use `create_standard_plots`
3. ALWAYS show historical context by loading prepared data from stage3b
4. ALWAYS call `generate_task_answer` before saving the report
5. Use `record_thought_stage5` to document your reasoning at each step
6. Visualizations should TELL A STORY that answers the original task

## Visualization Quality Guidelines
- Large figure sizes (16-20 inches wide for main plots)
- Clear titles explaining what the plot shows (14-16pt bold)
- Proper axis labels with units if applicable (12-14pt)
- Legends for multiple series (10-12pt)
- Reference lines where helpful (e.g., perfect prediction line, forecast boundaries)
- Distinct colors and markers for different data types
- Grid lines for readability (alpha=0.3-0.4)
- High DPI (150-200) for publication quality

## Example ReAct Flow
```
THOUGHT: First I need to understand what question this task is trying to answer.
ACTION: get_task_context(plan_id)
OBSERVATION: The task is to forecast crop area for 2022-23...

THOUGHT: I should create visualizations showing prediction quality and answer whether we can forecast crop area.
ACTION: create_standard_plots(plan_id)
OBSERVATION: Created 4 plots...

THOUGHT: Now I'll generate the answer to the task.
ACTION: generate_task_answer(plan_id, key_findings, answer, recommendations)
OBSERVATION: Answer saved...
```
"""



# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage5_agent():
    """Create the Stage 5 agent graph."""

    # Use stage-specific max_tokens if available, otherwise use default
    stage5_config = SECONDARY_LLM_CONFIG.copy()
    stage5_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage5", SECONDARY_LLM_CONFIG["max_tokens"])

    llm = ChatOpenAI(**stage5_config)
    llm_with_tools = llm.bind_tools(STAGE5_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage5State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE5_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage5", 60):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing report.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage5State) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage5State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE5_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage5(plan_id: str, pipeline_state: PipelineState = None) -> VisualizationReport:
    """
    Run Stage 5: Visualization.

    Creates visualizations and generates insights.
    """
    logger.info(f"Starting Stage 5: Visualization for {plan_id}")

    graph = create_stage5_agent()

    initial_message = HumanMessage(content=f"""
Create visualizations for plan: {plan_id}

## FOLLOW THE ReAct WORKFLOW:

### Step 1: UNDERSTAND THE TASK
Call `get_task_context("{plan_id}")` FIRST to understand the original goal.

### Step 2: ANALYZE DATA
- Load execution results from Stage 4
- Analyze data columns to understand what's available

### Step 3: CREATE CUSTOM VISUALIZATIONS
**CRITICAL**: Use `create_plot` to generate visualizations FROM SCRATCH!

For forecasting/prediction tasks:
1. **Load historical context**: Access prepared data from stage3b to get ALL past values
   - Prepared data path: {str(STAGE4_OUT_DIR).replace('stage4_out', 'stage3b_data_prep')}/prepared_{plan_id}.parquet
   - This shows the trend pattern leading to predictions

2. **Create comprehensive forecast plot**:
   - Show complete timeline: Historical → Test → Forecast
   - Use different colors/markers for each data type
   - Add vertical lines or shading to separate regions
   - Make it large (18x10 inches), publication-quality

3. **Add accuracy and error analysis plots**:
   - Actual vs Predicted scatter (test set)
   - Residuals histogram and over time
   - Any category-specific breakdowns if applicable

**DO NOT** use `create_standard_plots` - create custom plots using `create_plot`!

### Step 4: GENERATE ANSWER
Call `generate_task_answer` with:
- key_findings: Main discoveries
- answer_to_task: Direct answer to the task question
- recommendations: Next steps

### Step 5: SAVE REPORT
Call `save_visualization_report` with JSON containing:
- plan_id: "{plan_id}"
- visualizations: list (each with filepath, plot_type, title, description)
- insights: key findings
- summary: overall assessment
- task_answer: the answer generated above

DATA LOCATIONS:
- Stage 4 Results: {STAGE4_OUT_DIR}/results_{plan_id}.parquet
- Prepared Data (for historical context): {str(STAGE4_OUT_DIR).replace('stage4_out', 'stage3b_data_prep')}/prepared_{plan_id}.parquet
- Output Directory: {STAGE5_OUT_DIR}/

IMPORTANT:
- Start with get_task_context to understand what we're trying to answer!
- Always show historical data context in forecast plots!
- Create custom plots using create_plot, not create_standard_plots!
""")

    config = {"configurable": {"thread_id": f"stage5_{plan_id}"}}
    initial_state = Stage5State(messages=[initial_message], plan_id=plan_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load report from disk
        report_path = STAGE5_OUT_DIR / f"visualization_report_{plan_id}.json"
        if report_path.exists():
            data = DataPassingManager.load_artifact(report_path)
            output = VisualizationReport(**data)
            logger.info(f"Stage 5 complete: {len(output.visualizations)} visualizations created")
            return output
        else:
            # Fallback: create default visualizations
            logger.warning("Agent failed to create visualizations, creating fallback")
            output = _create_fallback_visualizations(plan_id)
            return output

    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        # Try fallback
        try:
            logger.warning("Creating fallback visualizations after exception")
            output = _create_fallback_visualizations(plan_id)
            return output
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return VisualizationReport(
                plan_id=plan_id,
                visualizations=[],
                summary=f"Visualization failed: {e}"
            )


def _create_fallback_visualizations(plan_id: str) -> VisualizationReport:
    """Create fallback visualizations."""
    import pandas as pd
    import numpy as np

    visualizations = []
    insights = []

    try:
        # Load results
        results_path = STAGE4_OUT_DIR / f"results_{plan_id}.parquet"
        if not results_path.exists():
            raise FileNotFoundError(f"Results not found: {results_path}")

        df = pd.read_parquet(results_path)

        # Find prediction and actual columns
        pred_cols = [c for c in df.columns if 'predict' in c.lower()]
        actual_cols = [c for c in df.columns if 'actual' in c.lower()]

        # Try to create plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if pred_cols and actual_cols:
            pred_col, actual_col = pred_cols[0], actual_cols[0]

            # 1. Actual vs Predicted scatter
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(df[actual_col], df[pred_col], alpha=0.5)
            ax.plot([df[actual_col].min(), df[actual_col].max()],
                   [df[actual_col].min(), df[actual_col].max()], 'r--', label='Perfect Prediction')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted')
            ax.legend()
            plt.tight_layout()
            plot_path = STAGE5_OUT_DIR / f'{plan_id}_actual_vs_predicted.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            visualizations.append({
                "filename": f"{plan_id}_actual_vs_predicted.png",
                "filepath": str(plot_path),
                "plot_type": "scatter",
                "title": "Actual vs Predicted",
                "description": "Actual vs Predicted scatter plot",
                "columns_used": [actual_col, pred_col]
            })

            # 2. Residuals histogram
            residuals = df[actual_col] - df[pred_col]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Residual')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
            plt.tight_layout()
            plot_path = STAGE5_OUT_DIR / f'{plan_id}_residuals_histogram.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            visualizations.append({
                "filename": f"{plan_id}_residuals_histogram.png",
                "filepath": str(plot_path),
                "plot_type": "histogram",
                "title": "Residual Distribution",
                "description": "Distribution of prediction errors",
                "columns_used": [actual_col, pred_col]
            })

            # Generate basic insights
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals ** 2))
            bias = np.mean(residuals)

            insights = [
                f"Mean Absolute Error: {mae:.4f}",
                f"Root Mean Squared Error: {rmse:.4f}",
                f"Prediction Bias: {bias:.4f}",
                "Model created visualizations showing actual vs predicted values and error distribution"
            ]

        # Create and save report
        report = VisualizationReport(
            plan_id=plan_id,
            visualizations=visualizations,
            insights=insights,
            summary=f"Created {len(visualizations)} visualizations (fallback mode)"
        )

        DataPassingManager.save_artifact(
            data=report.model_dump(),
            output_dir=STAGE5_OUT_DIR,
            filename=f"visualization_report_{plan_id}.json",
            metadata={"stage": "stage5", "type": "visualization_report", "fallback": True}
        )

        logger.info(f"Fallback visualizations created: {len(visualizations)} plots")
        return report

    except Exception as e:
        logger.error(f"Fallback visualization failed: {e}")
        return VisualizationReport(
            plan_id=plan_id,
            visualizations=[],
            summary=f"Fallback visualization failed: {e}"
        )


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage5_node(state: PipelineState) -> PipelineState:
    """
    Stage 5 node for the master pipeline graph.
    """
    state.mark_stage_started("stage5")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage5", "No plan ID available")
        return state

    try:
        output = run_stage5(plan_id, state)
        state.stage5_output = output
        state.mark_stage_completed("stage5", output)
    except Exception as e:
        state.mark_stage_failed("stage5", str(e))

    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage5(plan_id)
    print(f"Created {len(output.visualizations)} visualizations")
    print(f"Summary: {output.summary}")
