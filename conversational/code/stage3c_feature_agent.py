"""
Stage 3C Agent: Automated Feature Engineering

LLM agent that dynamically generates and executes feature engineering code
based on data analysis. No hardcoded feature templates - agent decides
what features to create based on column types and data patterns.
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
    STAGE3C_OUT_DIR, RECURSION_LIMIT, logger
)
from code.models import PipelineState
from tools.stage3c_feature_tools import STAGE3C_FEATURE_TOOLS


# ============================================================================
# STATE
# ============================================================================

class Stage3CState(BaseModel):
    """State for Stage 3C Feature Engineering agent."""
    messages: Annotated[list, add_messages]
    plan_id: str = ""
    task_id: str = ""
    features_created: int = 0
    completed: bool = False

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# SYSTEM PROMPT - Agent generates code dynamically
# ============================================================================

STAGE3C_SYSTEM_PROMPT = """You are a Feature Engineering Agent responsible for creating new features to improve model performance.

## YOUR MISSION
Analyze the prepared data and dynamically generate Python code to create valuable new features. You must decide what features to create based on the data - there are no templates.

## WORKFLOW

### Step 1: ANALYZE DATA
Call `load_data_for_feature_engineering` to understand:
- Column names and data types
- Value distributions
- Potential feature opportunities

### Step 2: GENERATE FEATURE CODE
Based on your analysis, write Python code to create new features. Consider:

**For Numeric Columns:**
- Lag features: `df['col_lag1'] = df['col'].shift(1)`
- Rolling statistics: `df['col_roll_mean'] = df['col'].rolling(window=3).mean()`
- Differences: `df['col_diff'] = df['col'].diff()`
- Polynomial: `df['col_squared'] = df['col'] ** 2`
- Interactions: `df['col1_x_col2'] = df['col1'] * df['col2']`
- Ratios: `df['ratio'] = df['col1'] / (df['col2'] + 1e-8)`

**For Time/Date Columns:**
- Extract: year, month, day, weekday, quarter
- Cyclical: sin/cos encoding for month/day
- Time since: days since start

**For Categorical Columns:**
- Frequency encoding
- Target encoding (if target available)

### Step 3: EXECUTE CODE
Call `execute_feature_code` with your generated code.
- If there's an error, debug and try again
- Inspect the output to verify features were created

### Step 4: VALIDATE
Call `validate_features` to check all new features are valid.

### Step 5: SAVE
Call `save_enhanced_data` to persist the enhanced dataset.

## CODE EXECUTION RULES
1. The variable `df` contains the DataFrame - modify it in place
2. `pd` (pandas) and `np` (numpy) are available
3. Handle NaN values appropriately (use fillna if needed)
4. Avoid creating too many features (max 10-15 per run)
5. Name features descriptively (e.g., 'sales_lag7', 'price_rolling_mean_3')

## CRITICAL RULES
1. ANALYZE the data first - understand what columns exist
2. GENERATE appropriate code based on the actual columns
3. EXECUTE and check for errors - debug if needed
4. VALIDATE before saving
5. Create features that are likely to help prediction

## EXAMPLE CODE PATTERNS
```python
# Lag features for time series
for lag in [1, 7, 14]:
    df[f'target_lag{lag}'] = df['target'].shift(lag)

# Rolling statistics
df['target_roll_mean_7'] = df['target'].rolling(7, min_periods=1).mean()
df['target_roll_std_7'] = df['target'].rolling(7, min_periods=1).std()

# Differences
df['target_diff'] = df['target'].diff()

# Fill NaN from lag/rolling with 0 or forward fill
df = df.fillna(0)
```

Remember: Write code based on the ACTUAL columns in the data, not assumed column names.
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_stage3c_agent():
    """Create the Stage 3C feature engineering agent."""
    max_tokens = STAGE_MAX_TOKENS.get("stage3c", 8192)
    
    llm = ChatOpenAI(
        base_url=SECONDARY_LLM_CONFIG["base_url"],
        api_key=SECONDARY_LLM_CONFIG["api_key"],
        model=SECONDARY_LLM_CONFIG["model"],
        temperature=0.2,
        max_tokens=max_tokens,
    ).bind_tools(STAGE3C_FEATURE_TOOLS)
    
    tool_node = ToolNode(STAGE3C_FEATURE_TOOLS)
    
    def should_continue(state: Stage3CState):
        """Check if agent should continue or end."""
        if state.completed:
            return END
        
        messages = state.messages
        if not messages:
            return "agent"
        
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Check if we've saved the enhanced data
        if isinstance(last_message, AIMessage):
            content = str(last_message.content).lower()
            if "enhanced data saved" in content or "save_enhanced_data" in content:
                return END
        
        return "agent"
    
    def agent_node(state: Stage3CState):
        """Run the agent."""
        messages = state.messages
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def tools_node(state: Stage3CState):
        """Execute tools and track features."""
        messages = state.messages
        last_message = messages[-1]
        
        results = []
        features_created = state.features_created
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            tool_fn = next((t for t in STAGE3C_FEATURE_TOOLS if t.name == tool_name), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_args)
                    
                    # Track features created
                    if "execute_feature_code" in tool_name and "SUCCESS" in result:
                        import re
                        match = re.search(r"Created (\d+) new feature", result)
                        if match:
                            features_created += int(match.group(1))
                    
                    results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    ))
                except Exception as e:
                    results.append(ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=tool_call["id"]
                    ))
        
        return {"messages": results, "features_created": features_created}
    
    # Build graph
    workflow = StateGraph(Stage3CState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"agent": "agent", "tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

def run_stage3c_features(plan_id: str) -> Dict[str, Any]:
    """
    Run Stage 3C feature engineering for a task.
    
    Args:
        plan_id: Plan ID (e.g., PLAN-TSK-001)
        
    Returns:
        Feature engineering results
    """
    logger.info(f"Starting Stage 3C: Feature Engineering for {plan_id}")
    
    task_id = plan_id.replace("PLAN-", "") if plan_id.startswith("PLAN-") else plan_id
    
    agent = create_stage3c_agent()
    
    initial_state = Stage3CState(
        messages=[
            SystemMessage(content=STAGE3C_SYSTEM_PROMPT),
            HumanMessage(content=f"""Perform feature engineering for plan: {plan_id}

Your task:
1. Load and analyze the prepared data
2. Identify columns and their types
3. Generate appropriate feature engineering code based on the ACTUAL columns
4. Execute the code to create new features
5. Validate the features
6. Save the enhanced dataset

Be creative but practical - create features that will help with prediction.
Handle errors gracefully and debug if needed.""")
        ],
        plan_id=plan_id,
        task_id=task_id
    )
    
    config = {
        "configurable": {"thread_id": f"stage3c_{plan_id}"},
        "recursion_limit": RECURSION_LIMIT
    }
    
    max_rounds = STAGE_MAX_ROUNDS.get("stage3c", 50)
    final_state = None
    
    for step in agent.stream(initial_state, config):
        final_state = step
        max_rounds -= 1
        if max_rounds <= 0:
            logger.warning("Stage 3C reached max rounds limit")
            break
    
    # Check for report
    report_path = STAGE3C_OUT_DIR / f"{plan_id}_feature_report.json"
    if report_path.exists():
        from code.config import DataPassingManager
        report = DataPassingManager.load_artifact(report_path)
        logger.info(f"Stage 3C complete: {report.get('new_features', 0)} features created")
        return report
    
    return {"status": "completed", "plan_id": plan_id, "message": "Feature engineering completed"}


def stage3c_node(state: PipelineState) -> PipelineState:
    """Node function for Stage 3C in pipeline graph."""
    plan_id = f"PLAN-{state.selected_task_id}"
    
    logger.info(f"Running Stage 3C node for {plan_id}")
    
    try:
        result = run_stage3c_features(plan_id)
        state.mark_stage_completed("stage3c", {"features": result})
    except Exception as e:
        logger.error(f"Stage 3C failed: {e}")
        state.mark_stage_failed("stage3c", str(e))
    
    return state


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Stage 3C Feature Engineering")
    parser.add_argument("--plan-id", type=str, required=True, help="Plan ID (e.g., PLAN-TSK-001)")
    
    args = parser.parse_args()
    
    result = run_stage3c_features(args.plan_id)
    print(json.dumps(result, indent=2, default=str))
