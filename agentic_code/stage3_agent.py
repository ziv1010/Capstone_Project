"""
Stage 3: Execution Planning Agent

Creates detailed execution plans for selected analytical tasks using LangGraph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import STAGE3_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE3_MAX_ROUNDS
from .models import Stage3Plan
from .tools import STAGE3_TOOLS


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE3_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

STAGE3_SYSTEM_PROMPT = """You are a data pipeline planning agent.

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════

1. You MUST end by calling save_stage3_plan(plan_json=...) - this is YOUR ONLY SUCCESS CRITERIA
2. NEVER write JSON in your reasoning - build it silently and pass to the tool
3. Be dataset-agnostic - no domain assumptions
4. Set plan_id EXACTLY to "PLAN-{selected_task_id}"

═══════════════════════════════════════════════════════════════
YOUR 3-STEP WORKFLOW
═══════════════════════════════════════════════════════════════

STEP 1: UNDERSTAND (2-3 tool calls)
-----------------------------------
- load_task_proposal(task_id)
- list_data_files()  
- inspect_data_file(filename) for required files

STEP 2: BUILD JSON SILENTLY
----------------------------
In your head, construct the Stage3Plan JSON with these sections:

```json
{
  "plan_id": "PLAN-{selected_task_id}",  // EXACT format!
  "selected_task_id": "{task_id}",
  "goal": "Brief description",
  "task_category": "descriptive|predictive|unsupervised",
  "artifacts": {
    "intermediate_table": "{task_id}_data.parquet",
    "intermediate_format": "parquet",
    "expected_columns": ["col1", "col2"],
    "expected_row_count_range": [min, max]
  },
  "file_instructions": [{
    "file_id": "file1",
    "original_name": "actual_filename.csv",
    "alias": "short_name",
    "rename_columns": {"Original Name": "clean_name"},
    "keep_columns": ["clean_name1", "clean_name2"],
    "filters": [],
    "join_keys": [],
    "notes": null
  }],
  "join_steps": [{
    "step": 1,
    "description": "Load base table",
    "left_table": "short_name",
    "right_table": null,
    "join_type": "base",
    "join_keys": [],
    "expected_cardinality": "base",
    "validation": {}
  }],
  "feature_engineering": [{
    "feature_name": "new_col",
    "description": "What it represents",
    "transform": "mean/sum/etc",
    "depends_on": ["source_col1"],
    "implementation": "df['new_col'] = ..."
  }],
  "validation": {
    "time_split": null,
    "coverage_checks": [],
    "cardinality_checks": [],
    "additional_checks": ["Data loaded", "No duplicates"]
  },
  "expected_model_types": ["Aggregation"],
  "evaluation_metrics": ["Summary Statistics"],
  "notes": ["Any important context"],
  "key_normalization": []
}
```

STEP 3: SAVE (1 tool call - THIS IS MANDATORY)
-----------------------------------------------
Call: save_stage3_plan(plan_json=<your complete JSON as a string>)

═══════════════════════════════════════════════════════════════
EXAMPLES OF CORRECT BEHAVIOR
═══════════════════════════════════════════════════════════════

GOOD (will succeed):
- Round 1: load_task_proposal, list_files, inspect_file
- Round 2: save_stage3_plan(plan_json="...") ✅

BAD (will fail):
- Round 1-4: Never calls save_stage3_plan ❌
- Round 4: Prints JSON in reasoning instead of calling tool ❌

═══════════════════════════════════════════════════════════════
IMPORTANT TIPS
═══════════════════════════════════════════════════════════════

For WIDE FORMAT data (years as columns like "2018-Quantity", "2019-Quantity"):
- Rename to "quantity_2018", "quantity_2019", etc.
- Keep ALL year columns
- Create aggregate features: mean, growth, trend

For SINGLE FILE descriptive tasks:
- join_steps: ONE entry with join_type="base"
- feature_engineering: simple aggregates (mean, sum, growth)
- validation: method="none" for time_split
- expected_model_types: ["Aggregation", "Visualization"]

Remember: Your ONLY job is to call save_stage3_plan() with valid JSON.
DO NOT explain the plan. DO NOT show JSON to the user. JUST SAVE IT."""


# ===========================
# LangGraph
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(STAGE3_TOOLS)


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls."""
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
stage3_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 3 Runner
# ===========================

def run_stage3(task_id: str, max_rounds: int = STAGE3_MAX_ROUNDS, debug: bool = True) -> Dict:
    """Run Stage 3 planning for a specific task.
    
    Args:
        task_id: Task ID from Stage  2 (e.g., 'TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    system_msg = SystemMessage(content=STAGE3_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"Create Stage 3 plan for '{task_id}'.\n\n"
            f"Steps:\n"
            f"1. load_task_proposal('{task_id}')\n"
            f"2. list_data_files()\n"
            f"3. inspect_data_file() for required files\n"
            f"4. Build Stage3Plan JSON silently\n"
            f"5. save_stage3_plan(plan_json=...) ← MANDATORY FINAL STEP\n\n"
            f"Rules:\n"
            f"- Set plan_id = 'PLAN-{task_id}'\n"
            f"- Never print JSON in reasoning\n"
            f"- Your ONLY success criteria: call save_stage3_plan()\n"
            f"- Complete in 3-4 rounds"
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    if not debug:
        return stage3_app.invoke(state, config={"configurable": {"thread_id": f"stage3-{task_id}"}})

    print("=" * 80)
    print(f"🚀 STAGE 3: Planning for {task_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage3_app.stream(
        state,
        config={"configurable": {"thread_id": f"stage3-{task_id}"}},
        stream_mode="values",
    ):
        msgs = curr_state["messages"]
        new_msgs = msgs[prev_len:]

        for m in new_msgs:
            msg_type = m.__class__.__name__
            if "System" in msg_type:
                print("\n" + "─" * 80)
                print("💻 [SYSTEM]")
                print("─" * 80)
                print(m.content[:500] + "..." if len(m.content) > 500 else m.content)
            elif "Human" in msg_type:
                print("\n" + "─" * 80)
                print("👤 [USER]")
                print("─" * 80)
                print(m.content)
            elif "AI" in msg_type:
                round_num += 1
                print("\n" + "═" * 80)
                print(f"🤖 [AGENT - Round {round_num}]")
                print("═" * 80)
                if m.content:
                    print("\n💭 Reasoning:")
                    content = m.content
                    if len(content) > 1000:
                        print(content[:500] + "\n...[truncated]...\n" + content[-500:])
                    else:
                        print(content)
                
                if hasattr(m, 'tool_calls') and m.tool_calls:
                    print("\n🔧 Tool Calls:")
                    for tc in m.tool_calls:
                        name = tc.get("name", "UNKNOWN")
                        args = tc.get("args", {})
                        print(f"\n  📌 {name}")
                        for k, v in args.items():
                            if isinstance(v, str) and len(v) > 200:
                                print(f"     {k}: {v[:100]}...[truncated]...{v[-100:]}")
                            else:
                                print(f"     {k}: {v}")

        prev_len = len(msgs)
        final_state = curr_state
        
        if round_num >= max_rounds:
            print(f"\n⚠️  Reached max rounds ({max_rounds}). Stopping.")
            break

    print("\n" + "=" * 80)
    print(f"✅ Complete - {round_num} rounds")
    print("=" * 80)
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage3_node(state: dict) -> dict:
    """Stage 3 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state with selected_task_id set
        
    Returns:
        Updated state with stage3_plan populated
    """
    task_id = state.get("selected_task_id")
    if not task_id:
        print("ERROR: No task_id selected for Stage 3")
        state["errors"].append("Stage 3: No task_id selected")
        return state
    
    print(f"\n🎯 Starting Stage 3 for: {task_id}\n")
    
    result = run_stage3(task_id, debug=True)
    
    # Check for saved plan
    plan_file = STAGE3_OUT_DIR / f"PLAN-{task_id}.json"
    if plan_file.exists():
        print(f"\n✅ SUCCESS! Plan saved to: {plan_file}")
        plan_data = json.loads(plan_file.read_text())
        state["stage3_plan"] = Stage3Plan.model_validate(plan_data)
        state["completed_stages"].append(3)
        state["current_stage"] = 4
    else:
        print("\n⚠️  WARNING: Plan not saved. Check logs above.")
        state["errors"].append("Stage 3: Plan not saved")
    
    return state


if __name__ == "__main__":
    # Run Stage 3 standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage3_agent.py <task_id>")
        print("Example: python stage3_agent.py TSK-001")
        sys.exit(1)
    
    task_id = sys.argv[1].strip()
    run_stage3(task_id)
