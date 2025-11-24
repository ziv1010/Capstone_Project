"""
Stage 2: Task Proposal Generation Agent

Uses LangGraph to explore dataset summaries and generate analytical task proposals.
Includes an exploration phase followed by synthesis of final proposals.
"""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict

from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .config import SUMMARIES_DIR, STAGE2_OUT_DIR, PRIMARY_LLM_CONFIG, STAGE2_MAX_EXPLORATION_STEPS
from .models import Stage2Output, TaskProposal
from .tools import STAGE2_TOOLS
from .utils import parse_tool_call, parse_proposals_json


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**PRIMARY_LLM_CONFIG)

# Tools mapped by name
TOOLS_BY_NAME: Dict[str, Any] = {t.name: t for t in STAGE2_TOOLS}


# ===========================
# System Prompt
# ===========================

system_prompt = """
You are Agent 2 in a multi-stage, agentic data analytics pipeline.

Stage 1 has already produced dataset summaries as JSON files in the 'summaries/' directory.
Each summary includes:
- dataset_name, path
- columns (name, physical_dtype, logical_type, null_fraction, unique_fraction, examples, etc.)
- candidate_primary_keys (sometimes empty)
- notes

You have access to THREE TOOLS:

1. list_summary_files()
   - No arguments.
   - Returns a list of summary filenames (ending in .summary.json).

2. read_summary_file(filename: str)
   - Args: { "filename": "<one of the filenames from list_summary_files>" }
   - Returns the raw JSON content of that summary file as a string.

3. python_sandbox(code: str)
   - Args: { "code": "<python code>" }
   - Executes arbitrary Python code to help analyze summaries and design tasks.
   - The code can:
       * import json, math, statistics, pandas, etc.
       * access PROJECT_ROOT, DATA_DIR, SUMMARIES_DIR
       * call read_summary_file('<summary-filename>')
       * call list_summary_files()
       * open and inspect files directly
       * print intermediate results.
   - Returns whatever was printed to stdout, or an error string.

YOUR JOB IN THIS EXPLORATION PHASE:

- Use these tools to deeply understand the available datasets and how they might join.
- Think about potential predictive, descriptive, and unsupervised tasks you could define later.
- But in this phase you ONLY call tools and reason; you DO NOT yet output the final task proposals.

TOOL-CALLING PROTOCOL (EXPLORATION):

- In each step, you conceptually decide on ONE tool to call and produce
  a Python dict literal describing that call, for example:
    {"tool_name": "list_summary_files", "tool_args": {}}

- You MAY add natural language before/after and even wrap the dict in ```python ...``` code fences;
  the orchestration code will try to extract the dict from your message.

- Valid tool_name values in this phase:
    "list_summary_files", "read_summary_file", "python_sandbox".

Later, AFTER this exploration loop finishes, a separate prompt will ask you
to output the final set of TaskProposals based on everything you've learned.
"""


# ===========================
# LangGraph State + Nodes
# ===========================

MAX_STEPS = STAGE2_MAX_EXPLORATION_STEPS


class AgentState(TypedDict):
    """State for Stage 2 exploration agent."""
    messages: List[BaseMessage]
    step: int
    tool_name: Optional[str]
    tool_args: Dict[str, Any]


def agent_llm_node(state: AgentState) -> AgentState:
    """Node that calls the LLM and parses a tool call dict
    (but does NOT execute the tool).
    """
    print(f"\n=== LLM NODE (step={state.get('step', 0)}) ===")
    ai_msg: AIMessage = llm.invoke(state["messages"])

    raw_content = ai_msg.content
    raw = raw_content if isinstance(raw_content, str) else str(raw_content)
    raw = raw.strip()
    print("AI raw:", raw[:400], "..." if len(raw) > 400 else "")

    # Default if parsing fails
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = {}

    try:
        tool_call = parse_tool_call(raw)
        tool_name = tool_call.get("tool_name")
        tool_args = tool_call.get("tool_args", {}) or {}
    except Exception as e:
        print(f"[WARN] Could not parse tool call: {e}")
        # Nudge model next time
        nudge = HumanMessage(
            content=(
                "Your last message did not contain a valid tool call dict.\n"
                "Now respond with ONLY a single Python dict literal of the form:\n"
                '{"tool_name": "<one of: \'list_summary_files\', '
                '\'read_summary_file\', \'python_sandbox\'>", '
                '"tool_args": { ... }}\n'
                "Do not include explanation or markdown."
            )
        )
        state["messages"].append(ai_msg)
        state["messages"].append(nudge)
        state["tool_name"] = None
        state["tool_args"] = {}
        return state

    state["messages"].append(ai_msg)
    state["tool_name"] = tool_name
    state["tool_args"] = tool_args
    return state


def agent_tool_node(state: AgentState) -> AgentState:
    """Node that executes the chosen tool (if valid) and appends the result as a HumanMessage."""
    step = state.get("step", 0)
    print(f"\n=== TOOL NODE (step={step}) ===")

    name = state.get("tool_name")
    args = state.get("tool_args", {}) or {}

    if name not in TOOLS_BY_NAME:
        print(f"[ERROR] Unknown or missing tool_name '{name}'.")
        # Nudge the model again
        msg = HumanMessage(
            content=(
                f"Your last tool_name '{name}' was invalid.\n"
                "Please respond with a valid tool call dict using one of:\n"
                "'list_summary_files', 'read_summary_file', 'python_sandbox'."
            )
        )
        state["messages"].append(msg)
    else:
        tool = TOOLS_BY_NAME[name]
        print(f"Calling tool: {name} with args: {args}")
        try:
            result = tool.invoke(args)
        except Exception as e:
            result = f"[tool execution error] {e}"

        print(
            f"Tool {name} result (truncated):",
            str(result)[:400],
            "..." if len(str(result)) > 400 else "",
        )

        # Feed tool result back to the conversation
        state["messages"].append(
            HumanMessage(
                content=f"Result of tool '{name}' with args {args}:\n{result}"
            )
        )

    state["step"] = step + 1
    # Clear pending tool
    state["tool_name"] = None
    state["tool_args"] = {}
    return state


def continue_or_end(state: AgentState) -> str:
    """Router for LangGraph: either loop again or stop after MAX_STEPS."""
    if state.get("step", 0) >= MAX_STEPS:
        return "end"
    return "continue"


# ===========================
# Build LangGraph
# ===========================

graph = StateGraph(AgentState)

graph.add_node("agent_llm", agent_llm_node)
graph.add_node("agent_tool", agent_tool_node)

graph.set_entry_point("agent_llm")

# LLM -> TOOL
graph.add_edge("agent_llm", "agent_tool")

# TOOL -> LLM (loop) or END
graph.add_conditional_edges(
    "agent_tool",
    continue_or_end,
    {
        "continue": "agent_llm",
        "end": END,
    },
)

exploration_app = graph.compile()


# ===========================
# Final Synthesis
# ===========================

def build_proposals_from_history(messages: List[BaseMessage]) -> tuple[Dict, Stage2Output, Path]:
    """Ask the LLM (once, with at most 1 self-repair) to synthesize final TaskProposals
    as STRICT JSON. Parse, validate, and save as JSON.
    
    Returns:
        proposals_dict (raw JSON dict),
        stage2_output (Pydantic),
        out_path (Path to task_proposals.json).
    """
    final_prompt = """
Now, based on all the dataset summaries and tool outputs in this conversation,
synthesize your final plan of analytic tasks.

You MUST output a SINGLE STRICT JSON object with the following structure:

{
  "proposals": [
    {
      "id": "TSK-001",
      "category": "predictive" | "descriptive" | "unsupervised",
      "title": "short human-readable title",
      "problem_statement": "2–5 sentences explaining the analytic question and why it matters.",
      "required_files": ["Export-of-Rice-Varieties-to-Bangladesh,-2018-19-to-2024-25.csv", "..."],
      "join_plan": {
        "hypothesized_keys": [
          ["state", "year"],
          ["state", "crop", "year"]
        ],
        "notes": "brief commentary about join logic"
      },
      "target": {
        "name": "column name or null",
        "granularity": ["columns that define a prediction unit"] or null,
        "horizon": "forecast horizon like '1-year ahead' or null"
      },
      "feature_plan": {
        "candidates": ["Area-*", "Production-*", "..."],
        "transform_ideas": ["lagged features", "growth rates", "..."],
        "handling_missingness": "brief strategy"
      },
      "validation_plan": "how to evaluate or sanity-check this task",
      "quality_checks": [
        "simple checks to avoid leakage or broken joins",
        "..."
      ],
      "expected_outputs": [
        "tables",
        "plots",
        "model_metrics",
        "cluster_assignments",
        "summary_report"
      ]
    }
    // 2–7 more proposals here
  ]
}

STRICT JSON REQUIREMENTS:
- Use double quotes for all keys and string values.
- Use true / false / null for booleans and missing values.
- No comments, no trailing commas, no Python None/True/False.
- The top-level object MUST have exactly one key: "proposals".

Do NOT wrap your answer in natural language. It is OK (but not required) to wrap
your JSON in ```json ... ``` fences; if you do, the content inside MUST still be
valid JSON.
"""

    # First attempt
    all_messages = messages + [HumanMessage(content=final_prompt)]
    ai_final: AIMessage = llm.invoke(all_messages)

    raw = ai_final.content.strip()
    print("FINAL RAW (1st attempt):", raw[:400], "..." if len(raw) > 400 else "")

    try:
        proposals_dict = parse_proposals_json(raw)
    except Exception as e:
        print("[WARN] First JSON parse failed:", e)

        # Second attempt: ask the model explicitly to fix JSON
        repair_prompt = f"""
You previously tried to output JSON but it was not strictly valid.

Here is what you produced:

<<<
{raw}
>>>

Now, rewrite this into STRICT VALID JSON with the SAME intended content and the SAME "proposals" structure.

Requirements:
- A single JSON object with key "proposals".
- All keys and string values in double quotes.
- true / false / null instead of Python booleans or None.
- No comments, no trailing commas, no extra keys at top level.

Output ONLY the JSON (optionally inside ```json ... ```), with no extra explanation.
"""
        repair_messages = all_messages + [HumanMessage(content=repair_prompt)]
        ai_repair: AIMessage = llm.invoke(repair_messages)
        raw2 = ai_repair.content.strip()
        print("FINAL RAW (repair):", raw2[:400], "..." if len(raw2) > 400 else "")

        proposals_dict = parse_proposals_json(raw2)

    # Validate with Pydantic
    stage2_output = Stage2Output.model_validate(proposals_dict)

    # Save as JSON
    out_path = STAGE2_OUT_DIR / "task_proposals.json"
    out_path.write_text(json.dumps(proposals_dict, indent=2))

    return proposals_dict, stage2_output, out_path


# ===========================
# Main Stage 2 Runner
# ===========================

def run_stage2() -> Stage2Output:
    """Run Stage 2: Task proposal generation.
    
    Returns:
        Stage2Output with all task proposals
    """
    print("\n" + "=" * 80)
    print("STAGE 2: Task Proposal Generation")
    print("=" * 80)
    
    # Run exploration
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Begin by calling list_summary_files to see which summaries exist."),
        ],
        "step": 0,
        "tool_name": None,
        "tool_args": {},
    }

    final_state = exploration_app.invoke(initial_state)
    exploration_messages: List[BaseMessage] = final_state["messages"]
    
    print(f"\n✅ Exploration complete: {len(exploration_messages)} messages")
    
    # Synthesize proposals
    proposals_dict, stage2_output, proposals_path = build_proposals_from_history(exploration_messages)
    
    print(f"\nSaved proposals to: {proposals_path}")
    print(f"Number of proposals: {len(stage2_output.proposals)}")
    for p in stage2_output.proposals:
        print(f"- {p.id}: [{p.category}] {p.title}")
    
    return stage2_output


# ===========================
# State Node for Master Graph
# ===========================

def stage2_node(state: dict) -> dict:
    """Stage 2 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state with task_proposals populated
    """
    stage2_output = run_stage2()
    
    state["task_proposals"] = stage2_output.proposals
    state["completed_stages"].append(2)
    state["current_stage"] = 3
    
    print(f"\n✅ Stage 2 complete: Generated {len(stage2_output.proposals)} task proposals")
    
    return state


if __name__ == "__main__":
    # Run Stage 2 standalone
    run_stage2()
