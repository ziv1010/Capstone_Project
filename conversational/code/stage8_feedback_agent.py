"""
Stage 8 Agent: Feedback Loop & Auto-Remediation

LLM agent that analyzes guardrails results, diagnoses failures,
and dynamically generates remediation strategies. Triggers re-runs
of earlier stages with fixes applied.
"""

import json
from typing import Dict, Any, Annotated
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
    STAGE8_OUT_DIR, RECURSION_LIMIT, logger
)
from code.models import PipelineState
from tools.stage8_feedback_tools import STAGE8_FEEDBACK_TOOLS


# ============================================================================
# STATE
# ============================================================================

class Stage8State(BaseModel):
    """State for Stage 8 Feedback Loop agent."""
    messages: Annotated[list, add_messages]
    plan_id: str = ""
    task_id: str = ""
    original_score: float = 0.0
    issues_fixed: int = 0
    completed: bool = False

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# SYSTEM PROMPT - Agent reasons about fixes dynamically
# ============================================================================

STAGE8_SYSTEM_PROMPT = """You are a Feedback Loop Agent responsible for diagnosing model failures, applying remediations, and automatically re-running the pipeline with fixes.

## YOUR MISSION
Analyze guardrails results, understand WHY tests failed, generate remediation strategies, and AUTOMATICALLY re-run the pipeline to verify improvements. The improved results are saved as a NEW task version (preserving the original).

## WORKFLOW

### Step 1: ANALYZE FAILURES
Call `load_guardrails_for_feedback` to understand:
- Which tests failed and why
- The specific metrics that caused failure
- Potential remediation strategies

If validity score is >= 75% (HIGH), no remediation is needed. Report success.

### Step 2: DIAGNOSE ROOT CAUSES
For each failure, reason about the root cause:

**Correlation FAIL (< 0.5):**
- Model not learning patterns
- Need better features or different algorithm

**Propensity FAIL (> 40% extreme):**
- Selection bias - model works differently for subgroups
- Data imbalance issues

**IPW FAIL (MAE CV > 0.3):**
- Inconsistent performance across prediction ranges
- Model overfits to certain value ranges

**Residual FAIL:**
- Outliers: Remove extreme values
- Bias: Log-transform target

### Step 3: APPLY REMEDIATIONS
Generate and execute Python code to fix issues:

```python
# Remove outliers (for residual issues)
q1, q3 = df['target'].quantile([0.01, 0.99])
df = df[(df['target'] >= q1) & (df['target'] <= q3)]

# Log transform (for skewed data)
df['target'] = np.log1p(df['target'])

# Add balancing features (for propensity issues)
df['target_decile'] = pd.qcut(df['target'], 10, labels=False, duplicates='drop')
```

### Step 4: UPDATE CONSTRAINTS
Call `update_method_constraints` to guide method selection on re-run.

### Step 5: MARK FOR RE-RUN
Call `mark_for_rerun` with stages: "stage3_5a,stage3_5b,stage4,stage7"

### Step 6: SAVE REPORT
Call `save_feedback_report` with issues found, remediations applied, etc.

### Step 7: CLONE TASK
Call `clone_task_for_rerun` to create a NEW task version:
- TSK-005 → TSK-005-R1
- TSK-005-R1 → TSK-005-R2
This preserves the original results while testing improvements.

### Step 8: CREATE RERUN TASK (Simplified Flow)
Call `create_rerun_task` with the new plan ID and improvement details.
This adds the new task to task_proposals.json and instructs the user to run it via chat.

Arguments:
- new_plan_id: The plan ID returned from clone_task_for_rerun
- original_task_description: Brief description of the original task
- improvement_description: What remediations were applied
- target_column: The target column from the original task

The user will then type "run TSK-xxx-R1" in chat to execute the improved pipeline.

## CODE EXECUTION RULES
1. `df` contains the prepared DataFrame - modify it in place
2. `pd` (pandas) and `np` (numpy) are available
3. Be conservative - don't remove too much data
4. Focus on ONE major fix at a time

## CRITICAL RULES
1. ANALYZE first - understand the specific failures
2. REASON about root causes - don't just apply random fixes
3. If score is HIGH (>= 75%), skip remediation - just report success
4. ALWAYS clone and create rerun task after applying fixes
5. Tell user to run the new task via chat

## DECISION TREE
```
Score >= 75%? → No remediation needed, report success
Score < 75%? → Apply fixes → Clone task → Create rerun task → Tell user to run it
```
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage8_agent():
    """Create the Stage 8 feedback loop agent."""
    max_tokens = STAGE_MAX_TOKENS.get("stage8", 8192)
    
    llm = ChatOpenAI(
        base_url=SECONDARY_LLM_CONFIG["base_url"],
        api_key=SECONDARY_LLM_CONFIG["api_key"],
        model=SECONDARY_LLM_CONFIG["model"],
        temperature=0.2,
        max_tokens=max_tokens,
    ).bind_tools(STAGE8_FEEDBACK_TOOLS)
    
    tool_node = ToolNode(STAGE8_FEEDBACK_TOOLS)
    
    def should_continue(state: Stage8State):
        """Check if agent should continue or end."""
        if state.completed:
            return END
        
        messages = state.messages
        if not messages:
            return "agent"
        
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Check if we've saved the feedback report
        if isinstance(last_message, AIMessage):
            content = str(last_message.content).lower()
            if "feedback report saved" in content or "no remediation needed" in content:
                return END
        
        return "agent"
    
    def agent_node(state: Stage8State):
        """Run the agent."""
        messages = state.messages
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def tools_node(state: Stage8State):
        """Execute tools."""
        messages = state.messages
        last_message = messages[-1]
        
        results = []
        issues_fixed = state.issues_fixed
        original_score = state.original_score
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            tool_fn = next((t for t in STAGE8_FEEDBACK_TOOLS if t.name == tool_name), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_args)
                    
                    # Track issues fixed
                    if "execute_remediation_code" in tool_name and "SUCCESS" in result:
                        issues_fixed += 1
                    
                    # Track original score
                    if "load_guardrails" in tool_name:
                        import re
                        match = re.search(r"Overall Validity: ([\d.]+)%", result)
                        if match:
                            original_score = float(match.group(1))
                    
                    results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    ))
                except Exception as e:
                    results.append(ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=tool_call["id"]
                    ))
        
        return {"messages": results, "issues_fixed": issues_fixed, "original_score": original_score}
    
    # Build graph
    workflow = StateGraph(Stage8State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"agent": "agent", "tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

def run_stage8_feedback(plan_id: str) -> Dict[str, Any]:
    """
    Run Stage 8 feedback loop for a task.
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        
    Returns:
        Feedback loop results
    """
    logger.info(f"Starting Stage 8: Feedback Loop for {plan_id}")
    
    task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id
    
    agent = create_stage8_agent()
    
    initial_state = Stage8State(
        messages=[
            SystemMessage(content=STAGE8_SYSTEM_PROMPT),
            HumanMessage(content=f"""Analyze guardrails results and apply remediations for plan: {plan_id}

Your task:
1. Load and analyze the guardrails report
2. If score is HIGH (>= 75%), report success - no remediation needed
3. If score is LOW/MEDIUM:
   a. Diagnose the failures
   b. Generate and execute remediation code
   c. Update method constraints
   d. Mark stages for re-run
   e. Save the feedback report
   f. CLONE the task to create a new version (preserves original)
   g. CREATE RERUN TASK to register it for user execution

The simplified flow will:
- Create a new task ID (e.g., TSK-005 → TSK-005-R1)
- Add the task to task_proposals.json
- Tell the user to run it via chat (e.g., "run TSK-005-R1")

Be thoughtful - understand WHY tests failed before applying fixes.""")
        ],
        plan_id=plan_id,
        task_id=task_id
    )
    
    config = {
        "configurable": {"thread_id": f"stage8_{plan_id}"},
        "recursion_limit": RECURSION_LIMIT
    }
    
    max_rounds = STAGE_MAX_ROUNDS.get("stage8", 60)
    final_state = None
    
    for step in agent.stream(initial_state, config):
        final_state = step
        max_rounds -= 1
        if max_rounds <= 0:
            logger.warning("Stage 8 reached max rounds limit")
            break
    
    # Check for report
    report_path = STAGE8_OUT_DIR / f"{task_id}_feedback_report.json"
    if report_path.exists():
        from code.config import DataPassingManager
        report = DataPassingManager.load_artifact(report_path)
        logger.info(f"Stage 8 complete: {report.get('remediations_applied', 'none')}")
        return report
    
    return {"status": "completed", "plan_id": plan_id, "message": "Feedback loop completed"}


def stage8_node(state: PipelineState) -> PipelineState:
    """Node function for Stage 8 in pipeline graph."""
    plan_id = f"PLAN-{state.selected_task_id}"
    
    logger.info(f"Running Stage 8 node for {plan_id}")
    
    try:
        result = run_stage8_feedback(plan_id)
        state.mark_stage_completed("stage8", {"feedback": result})
    except Exception as e:
        logger.error(f"Stage 8 failed: {e}")
        state.mark_stage_failed("stage8", str(e))
    
    return state


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Stage 8 Feedback Loop")
    parser.add_argument("--plan-id", type=str, required=True, help="Plan ID (e.g., PLAN-TSK-001)")
    
    args = parser.parse_args()
    
    result = run_stage8_feedback(args.plan_id)
    print(json.dumps(result, indent=2, default=str))
