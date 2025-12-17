"""
Stage 7 Agent: Guardrails Validation

LLM agent that runs statistical validation tests to verify model predictions:
- Correlation analysis
- Propensity score analysis
- Residual analysis
- Creates visualizations and final validity report
"""

import json
from typing import Dict, Any, Optional, Annotated
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    SECONDARY_LLM_CONFIG, STAGE_MAX_TOKENS, STAGE_MAX_ROUNDS,
    STAGE7_OUT_DIR, RECURSION_LIMIT, logger
)
from code.models import PipelineState
from tools.stage7_guardrails_tools import STAGE7_GUARDRAILS_TOOLS


# ============================================================================
# STATE
# ============================================================================

class Stage7State(BaseModel):
    """State for Stage 7 Guardrails agent."""
    messages: Annotated[list, add_messages]
    plan_id: str = ""
    task_id: str = ""
    validation_results: Dict[str, str] = {}
    completed: bool = False

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

STAGE7_SYSTEM_PROMPT = """You are a Model Validation Agent responsible for running guardrails checks.

## YOUR MISSION
Validate machine learning model predictions using statistical tests to ensure:
1. Predictions are causally valid (not just correlated)
2. No selection bias in predictions
3. Model performs consistently across data subgroups
4. Residuals have good statistical properties

## WORKFLOW

### Step 1: LOAD DATA
Call `load_predictions_for_validation` to understand available data.

### Step 2: RUN VALIDATION TESTS
Run ALL of these tests in order:

1. **Correlation Analysis** - `run_correlation_analysis`
   - Checks relationship between predictions and actuals
   - Validates feature correlations are sensible

2. **Propensity Score Analysis** - `run_propensity_score_analysis`
   - Checks for covariate balance
   - Detects if model is biased toward certain data subgroups

3. **Residual Analysis** - `run_residual_analysis`
   - Checks residual normality
   - Detects outliers and systematic bias

### Step 3: CREATE VISUALIZATIONS
Call `create_guardrails_visualization` with these types:
- "residual_distribution" - Shows residual distribution
- "prediction_scatter" - Shows actual vs predicted
- "error_by_quintile" - Shows error distribution by prediction range

### Step 4: GENERATE REPORT
Call `save_guardrails_report` with:
- Result for each test (PASS/WARNING/FAIL)
- Overall assessment explaining validity

## CRITICAL RULES
1. Run ALL validation tests - do not skip any
2. Create at least 2 visualizations
3. Provide honest assessments based on actual test results
4. Generate actionable recommendations for WARNING/FAIL cases
5. Final report must include overall validity determination

## OUTPUT INTERPRETATION
- **PASS**: Model meets statistical validity requirements
- **WARNING**: Minor concerns - review recommended
- **FAIL**: Significant issues - model may not be causally valid

## OVERALL VALIDITY DETERMINATION
- All PASS = Valid model
- Any FAIL = Invalid model (needs investigation)
- Only WARNINGs = Needs review before deployment
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage7_agent():
    """Create the Stage 7 guardrails agent."""
    max_tokens = STAGE_MAX_TOKENS.get("stage7", 8192)
    
    llm = ChatOpenAI(
        base_url=SECONDARY_LLM_CONFIG["base_url"],
        api_key=SECONDARY_LLM_CONFIG["api_key"],
        model=SECONDARY_LLM_CONFIG["model"],
        temperature=0.1,
        max_tokens=max_tokens,
    ).bind_tools(STAGE7_GUARDRAILS_TOOLS)
    
    tool_node = ToolNode(STAGE7_GUARDRAILS_TOOLS)
    
    def should_continue(state: Stage7State):
        """Check if agent should continue or end."""
        if state.completed:
            return END
        
        messages = state.messages
        if not messages:
            return "agent"
        
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Check if we've saved the report
        if isinstance(last_message, AIMessage):
            content = str(last_message.content).lower()
            if "guardrails report saved" in content or "validity:" in content:
                return END
        
        return "agent"
    
    def agent_node(state: Stage7State):
        """Run the agent."""
        messages = state.messages
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def tools_node(state: Stage7State):
        """Execute tools and track results."""
        messages = state.messages
        last_message = messages[-1]
        
        results = []
        validation_results = dict(state.validation_results)
        completed = False
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Find and execute tool
            tool_fn = next((t for t in STAGE7_GUARDRAILS_TOOLS if t.name == tool_name), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_args)
                    
                    # Track validation results
                    if "correlation_analysis" in tool_name:
                        if "✅ PASS" in result:
                            validation_results["correlation"] = "PASS"
                        elif "❌ FAIL" in result:
                            validation_results["correlation"] = "FAIL"
                        else:
                            validation_results["correlation"] = "WARNING"
                    elif "propensity" in tool_name:
                        if "✅ PASS" in result:
                            validation_results["propensity"] = "PASS"
                        elif "❌ FAIL" in result:
                            validation_results["propensity"] = "FAIL"
                        else:
                            validation_results["propensity"] = "WARNING"
                    elif "residual" in tool_name:
                        if "Overall Residual Analysis: PASS" in result:
                            validation_results["residual"] = "PASS"
                        elif "Overall Residual Analysis: FAIL" in result:
                            validation_results["residual"] = "FAIL"
                        else:
                            validation_results["residual"] = "WARNING"
                    
                    results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    ))
                    
                    # Mark completed when report is saved
                    if "save_guardrails_report" in tool_name and "saved" in str(result).lower():
                        logger.info("[Stage7] Report saved - marking completed!")
                        completed = True
                except Exception as e:
                    results.append(ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=tool_call["id"]
                    ))
        
        return {"messages": results, "validation_results": validation_results, "completed": completed}
    
    def should_continue_after_tools(state: Stage7State):
        """Check if we should continue after tools or end."""
        if state.completed:
            logger.info("[Stage7] Completed flag is True - ending!")
            return END
        return "agent"
    
    # Build graph
    workflow = StateGraph(Stage7State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"agent": "agent", "tools": "tools", END: END})
    workflow.add_conditional_edges("tools", should_continue_after_tools, {"agent": "agent", END: END})
    
    return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

def run_stage7_guardrails(plan_id: str) -> Dict[str, Any]:
    """
    Run Stage 7 guardrails validation for a task.
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        
    Returns:
        Guardrails validation results
    """
    logger.info(f"Starting Stage 7: Guardrails Validation for {plan_id}")
    
    task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id
    
    agent = create_stage7_agent()
    
    initial_state = Stage7State(
        messages=[
            SystemMessage(content=STAGE7_SYSTEM_PROMPT),
            HumanMessage(content=f"""Run guardrails validation for plan: {plan_id}

Execute ALL validation tests:
1. Load predictions
2. Correlation analysis
3. Propensity score analysis
4. Residual analysis
5. Create visualizations (residual_distribution, prediction_scatter, error_by_quintile)
6. Save the final guardrails report with overall validity assessment

Be thorough and provide honest assessments.""")
        ],
        plan_id=plan_id,
        task_id=task_id
    )
    
    config = {
        "configurable": {"thread_id": f"stage7_{plan_id}"},
        "recursion_limit": RECURSION_LIMIT
    }
    
    max_rounds = STAGE_MAX_ROUNDS.get("stage7", 50)
    final_state = None
    report_path = STAGE7_OUT_DIR / f"{task_id}_guardrails_report.json"
    
    # Track initial file state to detect when it's saved/modified
    initial_mtime = report_path.stat().st_mtime if report_path.exists() else 0
    
    for step in agent.stream(initial_state, config):
        final_state = step
        max_rounds -= 1
        
        # HARD STOP: Check if report was saved/modified during this run
        if report_path.exists():
            current_mtime = report_path.stat().st_mtime
            if current_mtime > initial_mtime:
                logger.info(f"[Stage7 HARD STOP] Report saved/modified, exiting loop!")
                break
        
        if max_rounds <= 0:
            logger.warning("Stage 7 reached max rounds limit")
            break
    
    # Check for report
    report_path = STAGE7_OUT_DIR / f"{task_id}_guardrails_report.json"
    if report_path.exists():
        from code.config import DataPassingManager
        report = DataPassingManager.load_artifact(report_path)
        logger.info(f"Stage 7 complete: {report.get('validity', 'UNKNOWN')}")
        return report
    
    return {"status": "completed", "plan_id": plan_id, "message": "Guardrails validation completed"}


def stage7_node(state: PipelineState) -> PipelineState:
    """Node function for Stage 7 in pipeline graph."""
    plan_id = f"PLAN-{state.selected_task_id}"
    
    logger.info(f"Running Stage 7 node for {plan_id}")
    
    try:
        result = run_stage7_guardrails(plan_id)
        state.mark_stage_completed("stage7", {"guardrails": result})
    except Exception as e:
        logger.error(f"Stage 7 failed: {e}")
        state.mark_stage_failed("stage7", str(e))
    
    return state


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Stage 7 Guardrails Validation")
    parser.add_argument("--plan-id", type=str, required=True, help="Plan ID (e.g., PLAN-TSK-001)")
    
    args = parser.parse_args()
    
    result = run_stage7_guardrails(args.plan_id)
    print(json.dumps(result, indent=2, default=str))
