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

STAGE5_SYSTEM_PROMPT = """You are a Visualization Agent with FULL CREATIVE FREEDOM using the ReAct framework.

## YOUR MISSION
Create visualizations that **ANSWER THE ORIGINAL TASK QUESTION**. You analyze the data and decide what plots best tell the story.

## WORKFLOW (ReAct: THOUGHT → ACTION → OBSERVATION)

### Step 1: UNDERSTAND THE TASK (REQUIRED FIRST!)
1. Call `get_task_context(plan_id)` to understand:
   - What question is being answered?
   - What type of analysis was performed?
   - What would the user want to see?

### Step 2: DEEPLY ANALYZE THE DATA STRUCTURE
1. Call `load_execution_results(plan_id)` to examine:
   - What columns exist in the results?
   - What are the data types?
   - What is the shape of the data?

2. Call `analyze_data_columns(plan_id)` to understand:
   - Which columns are categorical (e.g., Crop, Season, Category)?
   - Which columns are numerical (predictions, actuals, metrics)?
   - Which columns represent time or ordering?
   - What relationships exist between columns?

3. **THINK about the data dimensions:**
   - If there are categorical columns: How can I use color, facets, or grouping to show patterns BY category?
   - If there are time columns: How can I show trends over time?
   - If there are multiple categories: Which comparisons would be most insightful?
   - What story does the data tell that generic plots would miss?

### Step 3: CREATE VISUALIZATIONS (YOUR CREATIVE CHOICE!)
**Use `create_plot` to generate custom Python matplotlib/seaborn code.**

Before creating each plot, ask yourself:
- Does this visualization leverage the FULL structure of the data?
- Am I using categorical columns to add meaningful groupings/colors?
- Does this help answer the original task question?
- Would a user learn something NEW from this that they couldn't see from a simple summary?

**Data-Driven Visualization Principles:**
- If data has categories (Crop, Season, Region, etc.) → Consider using them for color-coding, grouping, or faceting
- If data has multiple dimensions → Think about which comparisons are most meaningful
- If predictions exist → Show how they relate to actuals, and WHERE errors occur
- If time exists → Show temporal patterns and trends

### Step 4: GENERATE INSIGHTS & TASK ANSWER
1. Call `generate_insights(plan_id)` to extract key findings
2. Call `generate_task_answer` with:
   - key_findings: What did we discover?
   - answer_to_task: Direct answer to the original question
   - recommendations: What should the user do with these results?

### Step 5: SAVE THE REPORT (CRITICAL - DO NOT SKIP!)
**YOU MUST call `save_visualization_report` at the end!**

This is REQUIRED - without this call, your visualizations will be lost!

Call `save_visualization_report` with JSON containing:
{
  "plan_id": "{plan_id}",
  "visualizations": [
    {"filepath": "/path/to/plot.png", "plot_type": "scatter", "title": "...", "description": "..."}
  ],
  "insights": ["insight 1", "insight 2"],
  "summary": "Brief summary of what was created",
  "task_answer": "The answer from generate_task_answer"
}

## VISUALIZATION QUALITY REQUIREMENTS

### LEGENDS (REQUIRED!)
- Every plot MUST have a legend explaining colors/markers/lines
- Use `ax.legend()` or `plt.legend()` - NEVER skip this

### QUALITY STANDARDS
- Large, readable figure sizes (12-18 inches wide)
- Clear, descriptive titles (14-16pt, bold)
- Proper axis labels with units
- Grid lines for readability (alpha=0.3)
- Thoughtful color choices for categories (use 'tab10', 'Set2', etc.)

## CRITICAL REMINDERS
1. ALWAYS start with `get_task_context` to understand the goal
2. ANALYZE the data structure - look for categories, dimensions, patterns
3. CREATE visualizations that leverage the data's full structure
4. ALWAYS include legends
5. **ALWAYS call `save_visualization_report` at the end - this is MANDATORY!**
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

## YOUR TASK
Analyze the data structure and create visualizations that answer the original task question.
You have full creative freedom - think about what the DATA tells you to visualize.

## WORKFLOW (Follow these steps in order!)

1. **UNDERSTAND** the task:
   Call `get_task_context("{plan_id}")` to learn what question we're answering

2. **ANALYZE** the data deeply:
   - Call `load_execution_results("{plan_id}")` to see the data
   - Call `analyze_data_columns("{plan_id}")` to understand:
     * What categorical columns exist? (e.g., Crop, Season, Category)
     * What numerical columns? (predictions, actuals)
     * What dimensions can you leverage for grouping/coloring?

3. **VISUALIZE** based on data structure:
   - Use `create_plot` to create custom visualizations
   - Think: How can I use the data's categories and dimensions?
   - Think: What story does the data tell?
   - EVERY plot MUST include a legend with ax.legend()

4. **ANSWER** the question:
   Call `generate_task_answer` with findings, answer, and recommendations

5. **SAVE THE REPORT** (CRITICAL!):
   YOU MUST call `save_visualization_report` with a JSON containing all created plots!
   Without this, your work will be LOST!

## DATA LOCATION
- Results: {STAGE4_OUT_DIR}/results_{plan_id}.parquet

## CRITICAL REMINDERS
- Analyze the data STRUCTURE before deciding what to plot
- Use categorical columns for color-coding and grouping
- Include legends in EVERY plot
- **MUST call `save_visualization_report` at the end!**
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

        # Extract task ID from plan_id
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id

        if pred_cols and actual_cols:
            pred_col, actual_col = pred_cols[0], actual_cols[0]

            # 1. Actual vs Predicted scatter
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(df[actual_col], df[pred_col], alpha=0.5, label='Predictions')
            ax.plot([df[actual_col].min(), df[actual_col].max()],
                   [df[actual_col].min(), df[actual_col].max()], 'r--', label='Perfect Prediction')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted')
            ax.legend()
            plt.tight_layout()
            plot_path = STAGE5_OUT_DIR / f'{task_id}_actual_vs_predicted.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            visualizations.append({
                "filename": f"{task_id}_actual_vs_predicted.png",
                "filepath": str(plot_path),
                "plot_type": "scatter",
                "title": "Actual vs Predicted",
                "description": "Actual vs Predicted scatter plot",
                "columns_used": [actual_col, pred_col]
            })

            # 2. Residuals histogram
            residuals = df[actual_col] - df[pred_col]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, label='Residuals')
            ax.axvline(x=0, color='r', linestyle='--', label='Zero Error Line')
            ax.set_xlabel('Residual (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
            ax.legend()
            plt.tight_layout()
            plot_path = STAGE5_OUT_DIR / f'{task_id}_residuals_histogram.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            visualizations.append({
                "filename": f"{task_id}_residuals_histogram.png",
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
