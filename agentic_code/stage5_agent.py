"""
Stage 5: Visualization Agent

Creates comprehensive visualizations and reports from Stage 4 execution results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import STAGE5_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE5_MAX_ROUNDS
from .models import VisualizationReport
from .tools import STAGE5_TOOLS


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE5_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

STAGE5_SYSTEM_PROMPT = """You are Agent 5: The Visualizer.

Your mission: Create comprehensive, insightful visualizations from Stage 4 execution results.

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════

1. You are FULLY AUTONOMOUS - write visualization code for ANY data type
2. You are DATASET-AGNOSTIC - adapt to any domain
3. CREATE PUBLICATION-QUALITY VISUALS - clear, informative, professional
4. TELL A STORY - your visualizations should reveal insights
5. END BY CALLING save_visualization_report() - this is your success criterion

⚠️ ALWAYS INSPECT DATA STRUCTURE FIRST before creating plots! ⚠️
⚠️ CREATE SEPARATE PNG FILES - one chart per file! ⚠️
⚠️ USE CLEAR LABELS - titles, axes, units, readable fonts! ⚠️

Your success = Creating visualizations and saving report."""


# ===========================
# LangGraph
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """LLM agent step."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(STAGE5_TOOLS)


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
stage5_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 5 Runner
# ===========================

def run_stage5(plan_id: str, max_rounds: int = STAGE5_MAX_ROUNDS, debug: bool = True) -> Dict:
    """Create visualizations for a Stage 4 execution result.
    
    Args:
        plan_id: Plan ID to visualize (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    system_msg = SystemMessage(content=STAGE5_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"Create visualizations for plan: '{plan_id}'\n\n"
            f"Workflow:\n"
            f"1. list_stage4_results() - find the execution result\n"
            f"2. load_stage4_result() - load the result\n"
            f"3. load_stage3_plan() - understand the context\n"
            f"4. create_visualizations() - FIRST inspect data, THEN create charts\n"
            f"5. save_visualization_report() with paths and insights\n\n"
            f"Be autonomous. Create beautiful, insightful visualizations.\n"
            f"Your success = Visualizations created + Report saved."
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    if not debug:
        return stage5_app.invoke(state, config={"configurable": {"thread_id": f"stage5-{plan_id}"}})

    print("=" * 80)
    print(f"🚀 STAGE 5: Visualizing results for {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage5_app.stream(
        state,
        config={"configurable": {"thread_id": f"stage5-{plan_id}"}},
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
                            if isinstance(v, str) and len(v) > 300:
                                print(f"     {k}: {v[:150]}...[truncated]...{v[-150:]}")
                            else:
                                print(f"     {k}: {v}")

        prev_len = len(msgs)
        final_state = curr_state
        
        if round_num >= max_rounds:
            print(f"\n⚠️  Reached max rounds ({max_rounds})")
            break

    print("\n" + "=" * 80)
    print(f"✅ Complete - {round_num} rounds")
    print("=" * 80)
    return final_state


# ===========================
# State Node for Master Graph
# ===========================

def stage5_node(state: dict) -> dict:
    """Stage 5 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state with execution_result set
        
    Returns:
        Updated state with visualization_report populated
    """
    if not state.get("execution_result"):
        print("ERROR: No Stage 4 execution result available for visualization")
        state["errors"].append("Stage 5: No Stage 4 execution result available")
        return state
    
    plan_id = state["execution_result"].plan_id
    print(f"\n🎯 Visualizing results for: {plan_id}\n")
    
    result = run_stage5(plan_id, debug=True)
    
    # Check for visualization reports
    reports = sorted(STAGE5_OUT_DIR.glob(f"visualization_report_{plan_id}*.json"))
    if reports:
        latest = reports[-1]
        print(f"\n✅ Visualization report: {latest}")
        report_data = json.loads(latest.read_text())
        state["visualization_report"] = VisualizationReport.model_validate(report_data)
        state["completed_stages"].append(5)
        print(f"  Visualizations: {len(report_data.get('visualizations', []))}")
        if report_data.get('insights'):
            print(f"  Insights: {len(report_data['insights'])}")
    else:
        print("\n⚠️  No visualization report saved")
        state["errors"].append("Stage 5: No visualization report saved")
    
    return state


if __name__ == "__main__":
    # Run Stage 5 standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage5_agent.py <plan_id>")
        print("Example: python stage5_agent.py PLAN-TSK-001")
        sys.exit(1)
    
    plan_id = sys.argv[1].strip()
    run_stage5(plan_id)
