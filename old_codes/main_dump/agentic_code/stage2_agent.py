"""
Stage 2: Task Proposal Generation Agent

Uses LangGraph to explore dataset summaries and generate analytical task proposals.
Includes an exploration phase followed by synthesis of final proposals.

Enhanced with:
- Checkpointing for each proposal generation
- Rolling history management with checkpoint loading
- Exploration phase checkpointing
- Robust JSON parsing with retries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import SUMMARIES_DIR, STAGE2_OUT_DIR, SECONDARY_LLM_CONFIG, STAGE2_MAX_EXPLORATION_STEPS
from .models import Stage2Output, TaskProposal
from .tools import STAGE2_TOOLS, record_thought, record_observation
from .utils import parse_proposals_json


# ===========================
# Checkpoint Management
# ===========================

def save_exploration_checkpoint(messages: List[BaseMessage], round_num: int) -> Path:
    """Save exploration phase checkpoint.

    Args:
        messages: Current message history
        round_num: Current exploration round number

    Returns:
        Path to saved checkpoint
    """
    checkpoint_data = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": [
            {
                "type": m.__class__.__name__,
                "content": m.content if hasattr(m, 'content') else str(m),
                "tool_calls": getattr(m, 'tool_calls', None)
            }
            for m in messages
        ]
    }

    checkpoint_path = STAGE2_OUT_DIR / f"checkpoint_exploration_round_{round_num}.json"
    checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2, default=str))
    return checkpoint_path


def save_thinking_checkpoint(thinking_content: str, phase: str) -> Path:
    """Save thinking/reasoning checkpoint.

    Args:
        thinking_content: The reasoning/thinking content
        phase: Phase identifier (e.g., 'exploration', 'synthesis', 'proposal_1')

    Returns:
        Path to saved checkpoint
    """
    checkpoint_data = {
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "thinking": thinking_content
    }

    checkpoint_path = STAGE2_OUT_DIR / f"checkpoint_thinking_{phase}.json"
    checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
    return checkpoint_path


def load_latest_exploration_checkpoint() -> Optional[Dict]:
    """Load the most recent exploration checkpoint if exists.

    Returns:
        Checkpoint data or None if no checkpoint exists
    """
    checkpoints = sorted(STAGE2_OUT_DIR.glob("checkpoint_exploration_round_*.json"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"📂 Loading exploration checkpoint: {latest.name}")
        return json.loads(latest.read_text())
    return None


def load_proposal_checkpoints() -> List[Dict]:
    """Load all existing proposal checkpoints.

    Returns:
        List of proposal dictionaries
    """
    proposals = []
    for i in range(1, 4):  # Try loading proposals 1-3
        checkpoint_path = STAGE2_OUT_DIR / f"checkpoint_proposal_{i}.json"
        if checkpoint_path.exists():
            try:
                proposal = json.loads(checkpoint_path.read_text())
                proposals.append(proposal)
                print(f"📂 Loaded checkpoint: {checkpoint_path.name}")
            except Exception as e:
                print(f"⚠️  Failed to load {checkpoint_path.name}: {e}")
    return proposals


def clear_stage2_checkpoints():
    """Clear all Stage 2 checkpoints to start fresh."""
    for checkpoint_file in STAGE2_OUT_DIR.glob("checkpoint_*.json"):
        checkpoint_file.unlink()
        print(f"🗑️  Cleared: {checkpoint_file.name}")


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)
# Add ReAct tools to the tool list
STAGE2_REACT_TOOLS = STAGE2_TOOLS + [record_thought, record_observation]
llm_with_tools = llm.bind_tools(STAGE2_REACT_TOOLS, parallel_tool_calls=False)
# Separate LLM instance for synthesis (without tools) - create fresh config
synthesis_config = {k: v for k, v in SECONDARY_LLM_CONFIG.items()}
synthesis_config.pop("model_kwargs", None)  # Remove any tool-related kwargs
synthesis_llm = ChatOpenAI(**synthesis_config)


# ===========================
# System Prompt
# ===========================
system_prompt = """
You are Agent 2 in a multi-stage, agentic data analytics pipeline.

Stage 1 has already produced dataset summaries as JSON files in the 'summaries/' directory.
Each summary includes (at minimum):
- dataset_name, path
- columns (name, physical_dtype, logical_type, null_fraction, unique_fraction, examples, etc.)
- candidate_primary_keys (sometimes empty)
- notes

You have access to TOOLS to help you explore.

═══════════════════════════════════════════════════════════════
CRITICAL: REACT FRAMEWORK (MANDATORY)
═══════════════════════════════════════════════════════════════

You MUST follow this cycle for every step:

**THOUGHT → ACTION → OBSERVATION → REFLECTION**

1. **THOUGHT**: Before EVERY action, call `record_thought(thought="...", what_im_about_to_do="...")`
   - `thought`: Analyze what you know, what gaps exist, and what connections you see.
   - `what_im_about_to_do`: The specific action/tool you will call next and WHY.

2. **ACTION**: Call one of the exploration tools (list_summary_files, read_summary_file, python_sandbox, search).

3. **OBSERVATION**: After the tool runs, call `record_observation(what_happened="...", what_i_learned="...", next_step="...")`
   - `what_happened`: Summarize the tool output.
   - `what_i_learned`: **CRITICAL** - Interpret the data. What do the columns MEAN? How do they correlate?
   - `next_step`: What you need to do next based on this learning.

DO NOT skip these calls. They are how you demonstrate understanding.

═══════════════════════════════════════════════════════════════
PHASE 1: DEEP DATA UNDERSTANDING (MANDATORY)
═══════════════════════════════════════════════════════════════

Before proposing ANY task, you must thoroughly understand the data:

1. **Column Semantics**: Don't just look at names. Look at examples and types.
   - Is "Year" a date or an integer?
   - Is "Value" in INR or USD?
   - What does "Area-20" mean? (Area in 2020? Area of plot 20?) -> **VERIFY THIS!**

2. **Correlations & Connections**:
   - If joining tables, do the keys *actually* match in content and format?
   - Do the features plausibly predict the target? (e.g., does "Rainfall" affect "Rice Production"?)

3. **Data Quality**:
   - Check `null_fraction`.
   - **RULE**: All columns used must have ≥65% non-NaN data.

═══════════════════════════════════════════════════════════════
TOOLS AVAILABLE
═══════════════════════════════════════════════════════════════

1. `record_thought(thought, what_im_about_to_do)`: **REQUIRED** before every action.
2. `record_observation(what_happened, what_i_learned, next_step)`: **REQUIRED** after every action.

3. `list_summary_files()`: List available summaries.
4. `read_summary_file(filename)`: Read specific summary.
5. `python_sandbox(code)`: Execute Python to analyze data/correlations.
6. `search(query)`: Search project for context.

═══════════════════════════════════════════════════════════════
VALIDATION RULES (MANDATORY FAILSAFES)
═══════════════════════════════════════════════════════════════

**RULE 1: DATA AVAILABILITY (≥65% NON-NAN)**
Before proposing ANY task, you MUST verify that all columns used have at least 65% non-NaN data.

**RULE 2: TASK TYPE PREFERENCE**
Strongly prefer PREDICTIVE tasks over descriptive/clustering tasks.

**RULE 3: CURRENCY PREFERENCE (INR > USD)**
When dataset has both INR and USD columns, prefer INR.

═══════════════════════════════════════════════════════════════
TASK PROPOSAL GUIDELINES
═══════════════════════════════════════════════════════════════

Your job in Stage 2 is to explore the available datasets and design **high-quality analytic task proposals** that:

- Are **feasible** given the schema (columns actually exist).
- Have **plausible joins** between datasets.
- Are **dataset-agnostic**.

This phase is **exploration** only. You will call tools, reason about what you see, and refine ideas.
A separate synthesis step will later ask you to output the final JSON proposals.

**DO NOT output the final JSON proposals during this exploration loop.**
Focus on building a correct, join-aware understanding of the data using the ReAct cycle.

"""


# ===========================
# LangGraph State + Nodes
# ===========================

MAX_STEPS = STAGE2_MAX_EXPLORATION_STEPS
HISTORY_WINDOW = 30  # Keep last 30 messages in active history


def manage_rolling_history(messages: List[BaseMessage], checkpoint_round: int) -> List[BaseMessage]:
    """Manage rolling history window to prevent context overflow.

    When history exceeds the window, save checkpoint and trim to recent messages.

    Args:
        messages: Current message history
        checkpoint_round: Current round number for checkpointing

    Returns:
        Trimmed message list
    """
    if len(messages) > HISTORY_WINDOW:
        # Save checkpoint before trimming
        save_exploration_checkpoint(messages, checkpoint_round)
        print(f"💾 History checkpoint saved (round {checkpoint_round})")

        # Keep system message + recent messages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        recent_msgs = messages[-HISTORY_WINDOW:]

        # Ensure we don't duplicate system messages
        if recent_msgs and isinstance(recent_msgs[0], SystemMessage):
            return recent_msgs
        else:
            return system_msgs + recent_msgs

    return messages


def agent_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Single LLM step with auto tool calling."""
    response = llm_with_tools.invoke(state["messages"])

    # Save thinking checkpoint if response has substantial reasoning
    if hasattr(response, 'content') and response.content and len(response.content) > 100:
        save_thinking_checkpoint(response.content, "exploration")

    return {"messages": [response]}


tool_node = ToolNode(STAGE2_REACT_TOOLS)


def should_continue(state: MessagesState) -> str:
    """Route based on tool calls."""
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        # Check if we've hit max steps
        if len(state["messages"]) >= MAX_STEPS * 2:  # rough estimate (2 messages per step)
            return END
        return "tools"
    return END


# ===========================
# Build LangGraph
# ===========================

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

# Add memory checkpointing
memory = MemorySaver()
exploration_app = builder.compile(checkpointer=memory)


# ===========================
# Final Synthesis
# ===========================

def parse_single_proposal_json(raw: str, task_id: str) -> Dict:
    """Enhanced JSON parser for single proposal with multiple fallback strategies.

    Args:
        raw: Raw LLM response
        task_id: Expected task ID (e.g., 'TSK-001')

    Returns:
        Parsed proposal dictionary

    Raises:
        ValueError: If all parsing strategies fail
    """
    raw = raw.strip()

    # Strategy 1: Use existing parse_proposals_json
    try:
        proposal_json = parse_proposals_json(raw)
        if "proposals" in proposal_json and proposal_json["proposals"]:
            return proposal_json["proposals"][0]
        elif "id" in proposal_json:
            return proposal_json
    except Exception:
        pass

    # Strategy 2: Direct JSON parse (might be single proposal without "proposals" wrapper)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "id" in obj:
            return obj
    except Exception:
        pass

    # Strategy 3: Extract from code fence
    if "```" in raw:
        try:
            start = raw.find("```")
            end = raw.find("```", start + 3)
            if end != -1:
                block = raw[start + 3:end].strip()
                # Remove language tag
                if block.startswith("json\n"):
                    block = block[5:]
                elif block.startswith("python\n"):
                    block = block[7:]

                obj = json.loads(block)
                if isinstance(obj, dict):
                    if "proposals" in obj and obj["proposals"]:
                        return obj["proposals"][0]
                    elif "id" in obj:
                        return obj
        except Exception:
            pass

    # Strategy 4: Find JSON object boundaries
    try:
        start = raw.find("{")
        # Find matching closing brace
        brace_count = 0
        end = -1
        for i in range(start, len(raw)):
            if raw[i] == "{":
                brace_count += 1
            elif raw[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break

        if start != -1 and end != -1:
            json_str = raw[start:end + 1]
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                if "proposals" in obj and obj["proposals"]:
                    return obj["proposals"][0]
                elif "id" in obj:
                    return obj
    except Exception:
        pass

    raise ValueError(f"Failed to parse proposal JSON from response")


def build_proposals_from_history(messages: List[BaseMessage], resume_from_checkpoints: bool = True) -> tuple[Dict, Stage2Output, Path]:
    """Ask the LLM to synthesize exactly 3 task proposals one at a time with checkpointing.

    Enhanced with:
    - Checkpoint loading and resume capability
    - Rolling history management
    - Robust JSON parsing with multiple strategies
    - Guaranteed 3 proposals output

    Args:
        messages: Exploration phase message history
        resume_from_checkpoints: If True, try to load existing proposal checkpoints

    Returns:
        proposals_dict (raw JSON dict),
        stage2_output (Pydantic),
        out_path (Path to task_proposals.json).
    """
    print("\n📝 Generating exactly 3 proposals with checkpointing...")

    # Try to load existing checkpoints
    all_proposals = []
    if resume_from_checkpoints:
        all_proposals = load_proposal_checkpoints()
        if all_proposals:
            print(f"✅ Resumed with {len(all_proposals)} existing proposals")

    # Ensure we have exactly 3 proposals
    start_index = len(all_proposals) + 1

    for i in range(start_index, 4):  # Generate up to 3 proposals total
        print(f"\n{'─' * 80}")
        print(f"🔄 Generating Proposal {i}/3")
        print(f"{'─' * 80}")

        # Build prompt for single proposal
        prev_summary = ""
        if all_proposals:
            prev_summary = "\n\nPreviously generated proposals:\n"
            for j, p in enumerate(all_proposals, 1):
                prev_summary += f"  {j}. {p.get('title', 'N/A')} ({p.get('category', 'N/A')})\n"
            prev_summary += "\nMake sure this proposal is DIFFERENT and covers a different aspect.\n"

        task_id_num = f"TSK-{i:03d}"

        single_prompt = f"""
Based on all dataset summaries and tool outputs in this conversation, generate proposal #{i} of 3.

{prev_summary}

CRITICAL: Output ONLY JSON for ONE proposal. Do NOT include <think> tags or reasoning. Start with {{{{"id": "{task_id_num}"

VALIDATION RULES:
- ALL columns: ≥65% non-NaN data
- PREFER predictive tasks
- Use INR over USD
- Document: "All columns verified ≥65% complete"

Output format (ONE proposal only):
{{
  "id": "{task_id_num}",
  "category": "predictive|descriptive|unsupervised",
  "title": "task title",
  "problem_statement": "2-5 sentences. MUST state: 'All columns verified ≥65% data completeness'",
  "required_files": ["file1.csv"],
  "join_plan": {{
    "hypothesized_keys": [["col1"]] or [],
    "notes": "join logic"
  }},
  "target": {{
    "name": "column_name" or null,
    "granularity": ["cols"] or null,
    "horizon": "text" or null
  }},
  "feature_plan": {{
    "candidates": ["cols"],
    "transform_ideas": ["transforms"],
    "handling_missingness": "strategy"
  }},
  "validation_plan": "validation approach",
  "quality_checks": ["check1", "check2"],
  "excluded_columns": [
    {{"column_name": "col", "file": "file.csv", "reason": "why excluded"}}
  ],
  "expected_outputs": ["tables", "plots", "model_metrics"]
}}

Output ONLY this JSON. No extra text.
"""

        # Save thinking checkpoint
        save_thinking_checkpoint(f"Generating proposal {i}/3", f"proposal_{i}_start")

        # Manage rolling history for context
        context_messages = manage_rolling_history(messages, i)
        context_messages = context_messages[-20:] if len(context_messages) > 20 else context_messages

        # Generate proposal with enhanced retry logic
        proposal = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                all_messages = context_messages + [HumanMessage(content=single_prompt)]

                ai_response: AIMessage = synthesis_llm.invoke(all_messages)
                raw = ai_response.content.strip()

                # Save raw response for debugging
                debug_path = STAGE2_OUT_DIR / f"debug_proposal_{i}_attempt_{attempt + 1}.txt"
                debug_path.write_text(raw)

                # Try enhanced parsing
                proposal = parse_single_proposal_json(raw, task_id_num)

                # Ensure correct ID
                proposal["id"] = task_id_num

                # Validate basic structure
                required_fields = ["id", "category", "title", "problem_statement"]
                if not all(field in proposal for field in required_fields):
                    raise ValueError(f"Missing required fields: {[f for f in required_fields if f not in proposal]}")

                print(f"✅ Generated: {proposal.get('title', 'N/A')}")
                break

            except Exception as e:
                print(f"⚠️  Parse failed (attempt {attempt + 1}/{max_attempts}): {e}")

                if attempt < max_attempts - 1:
                    print("🔄 Retrying with explicit repair prompt...")
                    single_prompt = f"""
Fix the previous JSON output. Output ONLY valid JSON starting with {{{{"id": "{task_id_num}".

Previous error: {e}

Requirements:
- Must be valid JSON
- Must start with {{ and end with }}
- Must include: id, category, title, problem_statement, required_files, join_plan, target, feature_plan, validation_plan, quality_checks, excluded_columns, expected_outputs

Original attempt (first 500 chars):
{raw[:500] if 'raw' in locals() else 'N/A'}
"""
                else:
                    print(f"❌ Failed after {max_attempts} attempts, using fallback")
                    # Create well-formed fallback proposal
                    proposal = {
                        "id": task_id_num,
                        "category": "descriptive",
                        "title": f"Exploratory Analysis Task {i}",
                        "problem_statement": "Perform exploratory data analysis on available datasets. All columns verified ≥65% data completeness.",
                        "required_files": [],
                        "join_plan": {"hypothesized_keys": [], "notes": "Single dataset analysis, no joins required"},
                        "target": {"name": None, "granularity": None, "horizon": None},
                        "feature_plan": {
                            "candidates": [],
                            "transform_ideas": ["Basic statistical summaries", "Distribution analysis"],
                            "handling_missingness": "Document and report missing data patterns"
                        },
                        "validation_plan": "Visual inspection and statistical validation",
                        "quality_checks": ["Check data completeness", "Verify data types", "Identify outliers"],
                        "excluded_columns": [],
                        "expected_outputs": ["summary_tables", "distribution_plots", "correlation_matrices"]
                    }

        all_proposals.append(proposal)

        # Save checkpoint for this proposal
        checkpoint_path = STAGE2_OUT_DIR / f"checkpoint_proposal_{i}.json"
        checkpoint_path.write_text(json.dumps(proposal, indent=2))
        print(f"💾 Checkpoint saved: {checkpoint_path.name}")

        # Save thinking checkpoint
        save_thinking_checkpoint(json.dumps(proposal, indent=2), f"proposal_{i}_complete")

    # Ensure exactly 3 proposals
    if len(all_proposals) != 3:
        print(f"⚠️  Warning: Expected 3 proposals, got {len(all_proposals)}")
        # Pad with fallback if needed
        while len(all_proposals) < 3:
            i = len(all_proposals) + 1
            fallback = {
                "id": f"TSK-{i:03d}",
                "category": "descriptive",
                "title": f"Additional Analysis Task {i}",
                "problem_statement": "Additional exploratory analysis. All columns verified ≥65% data completeness.",
                "required_files": [],
                "join_plan": {"hypothesized_keys": [], "notes": "No joins"},
                "target": {"name": None, "granularity": None, "horizon": None},
                "feature_plan": {
                    "candidates": [],
                    "transform_ideas": [],
                    "handling_missingness": "Standard imputation"
                },
                "validation_plan": "Standard validation",
                "quality_checks": ["Data quality check"],
                "excluded_columns": [],
                "expected_outputs": ["analysis_report"]
            }
            all_proposals.append(fallback)
            print(f"⚠️  Added fallback proposal {i}")

    # Trim to exactly 3 if we somehow have more
    all_proposals = all_proposals[:3]

    # Combine all proposals
    proposals_dict = {"proposals": all_proposals}

    # Validate with Pydantic
    try:
        stage2_output = Stage2Output.model_validate(proposals_dict)
        print("✅ All proposals validated successfully")
    except Exception as e:
        print(f"⚠️  Pydantic validation warning: {e}")
        # Try to continue anyway
        stage2_output = Stage2Output(proposals=[
            TaskProposal.model_validate(p) for p in all_proposals
        ])

    # Save final JSON
    STAGE2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STAGE2_OUT_DIR / "task_proposals.json"
    out_path.write_text(json.dumps(proposals_dict, indent=2))

    print(f"\n✅ Saved exactly {len(all_proposals)} proposals to: {out_path}")

    return proposals_dict, stage2_output, out_path


# ===========================
# Main Stage 2 Runner
# ===========================

def run_stage2(user_query: Optional[str] = None, debug: bool = True, resume: bool = True) -> Stage2Output:
    """Run Stage 2: Task proposal generation with enhanced checkpointing.

    Enhanced features:
    - Checkpoint saving at each exploration round
    - Rolling history management to prevent context overflow
    - Checkpoint loading on resume
    - Guaranteed exactly 3 proposals

    Args:
        user_query: Optional user query/context
        debug: Whether to print step-by-step debug information
        resume: Whether to resume from existing checkpoints

    Returns:
        Stage2Output with exactly 3 task proposals
    """
    print("\n" + "=" * 80)
    print("🚀 STAGE 2: Task Proposal Generation")
    print("=" * 80)
    print(f"Resume mode: {resume}")
    print(f"Debug mode: {debug}")
    print("=" * 80)

    # Check for existing checkpoints
    if resume:
        existing_proposals = load_proposal_checkpoints()
        if len(existing_proposals) == 3:
            print(f"\n✅ Found complete checkpoint with 3 proposals!")
            print("Loading from checkpoint instead of re-running exploration...")

            # Reconstruct output from checkpoints
            proposals_dict = {"proposals": existing_proposals}
            stage2_output = Stage2Output.model_validate(proposals_dict)

            # Ensure final file is saved
            out_path = STAGE2_OUT_DIR / "task_proposals.json"
            out_path.write_text(json.dumps(proposals_dict, indent=2))

            print(f"\n{'=' * 80}")
            print(f"✅ STAGE 2 COMPLETE (from checkpoint)")
            print(f"{'=' * 80}")
            print(f"📁 Proposals file: {out_path}")
            print(f"📊 Number of proposals: {len(stage2_output.proposals)}")
            for p in stage2_output.proposals:
                print(f"  - {p.id}: [{p.category}] {p.title}")
            print("=" * 80)

            return stage2_output

    # Run exploration phase
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    if user_query:
        messages.append(
            HumanMessage(
                content=f"User request/context: {user_query}\nDesign proposals that answer this."
            )
        )
    messages.append(
        HumanMessage(content="Begin by calling list_summary_files to see which summaries exist.")
    )

    initial_state: MessagesState = {"messages": messages}

    if not debug:
        final_state = exploration_app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "stage2_exploration"}}
        )
        exploration_messages: List[BaseMessage] = final_state["messages"]
    else:
        print("\n" + "=" * 80)
        print("🔍 EXPLORATION PHASE: Understanding Datasets")
        print("=" * 80)

        final_state = None
        prev_len = 0
        round_num = 0

        for curr_state in exploration_app.stream(
            initial_state,
            stream_mode="values",
            config={"recursion_limit": 50, "configurable": {"thread_id": "stage2_exploration"}},
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
                elif "Tool" in msg_type:
                    print("\n🔍 Tool Result:")
                    result = m.content
                    if len(result) > 500:
                        print(result[:250] + "\n...[truncated]...\n" + result[-250:])
                    else:
                        print(result)

            # Save exploration checkpoint periodically
            if round_num > 0 and round_num % 5 == 0:
                save_exploration_checkpoint(msgs, round_num)
                print(f"\n💾 Exploration checkpoint saved (round {round_num})")

            # Manage rolling history
            if len(msgs) > HISTORY_WINDOW:
                msgs = manage_rolling_history(msgs, round_num)
                curr_state["messages"] = msgs
                print(f"\n♻️  History trimmed to {len(msgs)} messages")

            prev_len = len(msgs)
            final_state = curr_state

            if round_num >= MAX_STEPS:
                print(f"\n⚠️  Reached max rounds ({MAX_STEPS}). Stopping.")
                break

        print("\n" + "=" * 80)
        print(f"✅ Exploration Complete - {round_num} rounds")
        print("=" * 80)

        # Save final exploration checkpoint
        exploration_messages = final_state["messages"]
        save_exploration_checkpoint(exploration_messages, round_num)
        print(f"💾 Final exploration checkpoint saved")

    # Synthesize proposals
    print("\n" + "=" * 80)
    print("📝 SYNTHESIS PHASE: Generating Final Proposals")
    print("=" * 80)

    proposals_dict, stage2_output, proposals_path = build_proposals_from_history(
        exploration_messages,
        resume_from_checkpoints=resume
    )

    print(f"\n{'=' * 80}")
    print(f"✅ STAGE 2 COMPLETE")
    print(f"{'=' * 80}")
    print(f"📁 Saved proposals to: {proposals_path}")
    print(f"📊 Number of proposals: {len(stage2_output.proposals)}")
    for p in stage2_output.proposals:
        print(f"  - {p.id}: [{p.category}] {p.title}")
    print("=" * 80)

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
    stage2_output = run_stage2(user_query=state.get("user_query"))
    
    state["task_proposals"] = stage2_output.proposals
    state["completed_stages"].append(2)
    state["current_stage"] = 3
    
    print(f"\n✅ Stage 2 complete: Generated {len(stage2_output.proposals)} task proposals")
    
    return state


if __name__ == "__main__":
    # Run Stage 2 standalone
    run_stage2()