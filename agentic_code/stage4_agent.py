"""
Stage 4: Execution Agent

Executes Stage 3 plans by writing and running code to process data, build models, etc.
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

from .config import STAGE4_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE4_MAX_ROUNDS
from .models import ExecutionResult
from .tools import STAGE4_TOOLS


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
llm_with_tools = llm.bind_tools(STAGE4_TOOLS, parallel_tool_calls=False)


# ===========================
# System Prompt
# ===========================

STAGE4_SYSTEM_PROMPT = """You are Agent 4: The Executor.

Your mission: Execute the Stage 3 plan flawlessly and autonomously.

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════

1. You are FULLY AUTONOMOUS - write and execute code to handle ANY requirement
2. You are DATASET-AGNOSTIC - no domain assumptions
3. FOLLOW THE PLAN - the Stage 3 plan is your blueprint
4. VERIFY EVERYTHING - check your work at each step
5. END BY CALLING save_execution_result() - this is your success criterion

Your success = Executing the plan and saving results."""


# ===========================
# LangGraph
# ===========================

def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """LLM agent step."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(STAGE4_TOOLS)


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
stage4_app = builder.compile(checkpointer=memory)


# ===========================
# Stage 4 Runner
# ===========================

def run_stage4(plan_id: str, max_rounds: int = STAGE4_MAX_ROUNDS, debug: bool = True) -> Dict:
    """Execute a Stage 3 plan.
    
    Args:
        plan_id: Plan ID from Stage 3 (e.g., 'PLAN-TSK-001')
        max_rounds: Maximum number of agent rounds
        debug: Whether to print debug information
        
    Returns:
        Final state from the graph execution
    """
    system_msg = SystemMessage(content=STAGE4_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"Execute Stage 3 plan: '{plan_id}'\n\n"
            f"Workflow:\n"
            f"1. load_stage3_plan('{plan_id}')\n"
            f"2. Understand the plan structure\n"
            f"3. Execute each step using execute_python_code()\n"
            f"4. Verify results at each stage\n"
            f"5. save_execution_result() with final outputs\n\n"
            f"Be autonomous. Handle any issues. Follow the plan.\n"
            f"Your success = Plan executed + Results saved."
        )
    )

    state: MessagesState = {"messages": [system_msg, human_msg]}

    if not debug:
        return stage4_app.invoke(state, config={"configurable": {"thread_id": f"stage4-{plan_id}"}})

    print("=" * 80)
    print(f"🚀 STAGE 4: Executing plan {plan_id}")
    print("=" * 80)

    final_state = None
    prev_len = 0
    round_num = 0

    for curr_state in stage4_app.stream(
        state,
        config={"configurable": {"thread_id": f"stage4-{plan_id}"}},
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

def stage4_node(state: dict) -> dict:
    """Stage 4 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state with stage3_plan set
        
    Returns:
        Updated state with execution_result populated
    """
    if not state.get("stage3_plan"):
        print("ERROR: No Stage 3 plan available for execution")
        state["errors"].append("Stage 4: No Stage 3 plan available")
        return state
    
    plan_id = state["stage3_plan"].plan_id
    print(f"\n🎯 Executing plan: {plan_id}\n")
    
    result = run_stage4(plan_id, debug=True)
    
    # Check for execution results
    results = sorted(STAGE4_OUT_DIR.glob(f"execution_{plan_id}*.json"))
    if results:
        latest = results[-1]
        print(f"\n✅ Execution result: {latest}")
        result_data = json.loads(latest.read_text())
        state["execution_result"] = ExecutionResult.model_validate(result_data)
        state["completed_stages"].append(4)
        state["current_stage"] = 5
        print(f"  Status: {result_data.get('status', 'N/A')}")
        print(f"  Outputs: {len(result_data.get('outputs', {}))}")
        if result_data.get('metrics'):
            print(f"  Metrics: {result_data['metrics']}")
    else:
        print("\n⚠️  No execution result saved")
        state["errors"].append("Stage 4: No execution result saved")
    
    return state


if __name__ == "__main__":
    # Run Stage 4 standalone
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stage4_agent.py <plan_id>")
        print("Example: python stage4_agent.py PLAN-TSK-001")
        sys.exit(1)
    
    plan_id = sys.argv[1].strip()
    run_stage4(plan_id)
