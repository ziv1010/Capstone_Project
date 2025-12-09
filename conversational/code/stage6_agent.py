"""
Stage 6 Agent: Final Report Generation

This agent generates comprehensive final reports by reading task proposals,
execution results, and visualizations, then synthesizing them into a complete
answer to the original task.
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
    STAGE6_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS,
    STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.models import PipelineState
from tools.stage6_tools import STAGE6_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class Stage6State(BaseModel):
    """State for Stage 6 agent."""
    messages: Annotated[list, add_messages] = []
    plan_id: str = ""
    report_generated: bool = False
    iteration: int = 0
    complete: bool = False


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE6_SYSTEM_PROMPT = """You are a Final Report Generation Agent.

## YOUR MISSION
Generate a comprehensive final report that answers the original task question using ONLY the data
from the execution results and visualizations. Do NOT make up values, statistics, or conclusions
that are not supported by the actual data.

## WORKFLOW

### Step 1: GATHER ALL CONTEXT
1. Call `load_task_proposal` to understand the original question
2. Call `load_execution_plan` to understand the approach
3. Call `load_execution_results` to get performance metrics
4. Call `load_prediction_data` to get actual prediction statistics
5. Call `load_visualization_report` to see what visualizations were created
6. Call `load_task_answer` to check if Stage 5 generated an answer

### Step 2: ANALYZE THE DATA
Review all the loaded information carefully:
- What was the original task asking for?
- What methods were used?
- What were the actual performance metrics (MAE, RMSE, R2, etc.)?
- What do the predictions look like (min, max, mean)?
- What insights were generated?

### Step 3: GENERATE COMPREHENSIVE REPORT
Call `generate_final_report` with these sections:

**executive_summary**:
- Brief overview of the task (1-2 sentences)
- Key finding (e.g., "We successfully predicted X with Y accuracy")
- Most important metric value

**methodology**:
- What data was used
- What methods/models were applied
- How the model was validated
- Reference ONLY methods that were actually used (from execution plan)

**results_analysis**:
- Present the actual performance metrics from the execution results
- Describe the predictions (statistics from prediction data)
- Discuss what the visualizations show
- Be specific with numbers - use the ACTUAL values from loaded data
- DO NOT invent metrics or statistics

**conclusions**:
- Answer the original task question directly
- Base conclusions ONLY on the actual results
- State whether the task was successfully completed
- Mention any limitations observed in the data

**recommendations**:
- Suggest next steps based on the results
- Mention potential improvements
- Highlight any concerns or caveats

## CRITICAL RULES
1. **USE ONLY ACTUAL DATA**: Never make up metrics, statistics, or findings
2. **BE SPECIFIC**: Quote exact metric values from the execution results
3. **REFERENCE WHAT EXISTS**: Only mention visualizations that were actually created
4. **ANSWER THE QUESTION**: The report must directly answer the original task
5. **NO HALLUCINATION**: If data is missing, state that explicitly rather than inventing it

## EXAMPLE WORKFLOW
```
1. Load task proposal -> "Predict crop area for 2022-23"
2. Load execution results -> "MAE: 245.67, RMSE: 312.45, R2: 0.89"
3. Load prediction data -> "Mean prediction: 1234.56, 50 forecast points"
4. Generate report with THESE EXACT VALUES in results_analysis
```

## OUTPUT FORMAT
Generate a well-structured, professional report that:
- Uses clear section headers
- Presents findings in a logical order
- Cites actual numbers from the data
- Provides actionable insights
- Directly addresses the original task question
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage6_agent():
    """Create the Stage 6 agent graph."""

    # Use stage-specific max_tokens if available
    stage6_config = SECONDARY_LLM_CONFIG.copy()
    stage6_config["max_tokens"] = STAGE_MAX_TOKENS.get("stage6", SECONDARY_LLM_CONFIG["max_tokens"])

    llm = ChatOpenAI(**stage6_config)
    llm_with_tools = llm.bind_tools(STAGE6_TOOLS, parallel_tool_calls=False)

    def agent_node(state: Stage6State) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=STAGE6_SYSTEM_PROMPT)] + list(messages)

        if state.iteration >= STAGE_MAX_ROUNDS.get("stage6", 30):
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Finalizing report.")],
                "complete": True
            }

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }

    def should_continue(state: Stage6State) -> str:
        """Determine if we should continue or end."""
        if state.complete or state.report_generated:
            return "end"

        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

        return "end"

    builder = StateGraph(Stage6State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(STAGE6_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


def run_stage6(plan_id: str, pipeline_state: PipelineState = None) -> Dict[str, Any]:
    """
    Run Stage 6: Final Report Generation.

    Generates a comprehensive report based on all previous stage outputs.

    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        pipeline_state: Optional pipeline state

    Returns:
        Dictionary with report details
    """
    logger.info(f"Starting Stage 6: Final Report Generation for {plan_id}")

    graph = create_stage6_agent()

    initial_message = HumanMessage(content=f"""
Generate a comprehensive final report for plan: {plan_id}

## INSTRUCTIONS:

### Step 1: GATHER DATA
Load all relevant information:
1. `load_task_proposal("{plan_id}")` - Get the original task
2. `load_execution_plan("{plan_id}")` - Get the execution plan
3. `load_execution_results("{plan_id}")` - Get performance metrics
4. `load_prediction_data("{plan_id}")` - Get prediction statistics
5. `load_visualization_report("{plan_id}")` - Get visualization details
6. `load_task_answer("{plan_id}")` - Check for existing answer

### Step 2: ANALYZE
Review all the data you loaded. Understand:
- What was the original question?
- What methods were used?
- What are the ACTUAL performance metrics?
- What predictions were made?

### Step 3: GENERATE REPORT
Call `generate_final_report` with:
- **executive_summary**: Brief overview and key finding
- **methodology**: Methods used (based on execution plan)
- **results_analysis**: Detailed results with ACTUAL metrics and statistics
- **conclusions**: Answer to the original task question
- **recommendations**: Next steps and improvements

CRITICAL: Use ONLY data from the loaded files. Do NOT invent metrics or statistics.
Be specific and cite actual numbers from the execution results.
""")

    config = {"configurable": {"thread_id": f"stage6_{plan_id}"}}
    initial_state = Stage6State(messages=[initial_message], plan_id=plan_id)

    try:
        final_state = graph.invoke(initial_state, config)

        # Load report from disk
        task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id
        report_path = STAGE6_OUT_DIR / f"{task_id}_final_report.json"

        if report_path.exists():
            data = DataPassingManager.load_artifact(report_path)
            logger.info(f"Stage 6 complete: Report saved to {report_path}")
            return {
                "plan_id": plan_id,
                "report_path": str(report_path),
                "status": "completed",
                **data
            }
        else:
            logger.warning("Report file not created")
            return {
                "plan_id": plan_id,
                "status": "failed",
                "error": "Report file not generated"
            }

    except Exception as e:
        logger.error(f"Stage 6 failed: {e}")
        return {
            "plan_id": plan_id,
            "status": "failed",
            "error": str(e)
        }


# ============================================================================
# PIPELINE NODE FUNCTION
# ============================================================================

def stage6_node(state: PipelineState) -> PipelineState:
    """
    Stage 6 node for the master pipeline graph.
    """
    state.mark_stage_started("stage6")

    plan_id = f"PLAN-{state.selected_task_id}" if state.selected_task_id else None
    if not plan_id:
        state.mark_stage_failed("stage6", "No plan ID available")
        return state

    try:
        output = run_stage6(plan_id, state)

        if output.get('status') == 'completed':
            state.mark_stage_completed("stage6", output)
        else:
            state.mark_stage_failed("stage6", output.get('error', 'Unknown error'))
    except Exception as e:
        state.mark_stage_failed("stage6", str(e))

    return state


if __name__ == "__main__":
    import sys
    plan_id = sys.argv[1] if len(sys.argv) > 1 else "PLAN-TSK-001"
    output = run_stage6(plan_id)
    print(f"Status: {output.get('status')}")
    if output.get('report_path'):
        print(f"Report: {output['report_path']}")
