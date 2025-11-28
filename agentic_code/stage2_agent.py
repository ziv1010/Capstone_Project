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
from . import tools  # for shared sandbox state
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
Each summary includes (at minimum):
- dataset_name, path
- columns (name, physical_dtype, logical_type, null_fraction, unique_fraction, examples, etc.)
- candidate_primary_keys (sometimes empty)
- notes

You have access to FOUR TOOLS:

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
       * call list_summary_files()
       * call read_summary_file('<summary-filename>')
       * open and inspect files directly
       * print intermediate results.
   - Returns whatever was printed to stdout, or an error string.

4. search(query: str, within: str = "project", file_glob: str = "**/*", ...)
   - Workspace text search.
   - Use it to:
       * see how raw datasets are used in existing code,
       * find prior analyses or metrics for specific datasets,
       * locate existing join logic or feature engineering code.

═══════════════════════════════════════════════════════════════
CRITICAL VALIDATION RULES (MANDATORY FAILSAFES)
═══════════════════════════════════════════════════════════════

**RULE 1: DATA AVAILABILITY (≥65% NON-NAN)**
Before proposing ANY task, you MUST verify that all columns used have at least 65% non-NaN data.

How to check:
1. Review Stage 1 summaries for null_fraction per column
2. For a column to be usable: (1 - null_fraction) >= 0.65
3. If null_fraction > 0.35, DO NOT use that column
4. If unsure, use python_sandbox() to calculate: `df['column'].notna().sum() / len(df)`

Example validation:
```python
# Check data availability
import pandas as pd
df = load_dataframe('file.csv')

for col in ['export_value', 'production', 'year']:
    non_nan_pct = df[col].notna().sum() / len(df)
    print(f"{col}: {non_nan_pct*100:.1f}% non-NaN")
    if non_nan_pct < 0.65:
        print(f"  ❌ REJECT - Insufficient data!")
    else:
        print(f"  ✓ OK - Can use this column")
```

**RULE 2: TASK TYPE PREFERENCE**
Strongly prefer PREDICTIVE tasks over descriptive/clustering tasks.

Priority order:
1. **PREDICTIVE** (forecasting, regression, classification) ← HIGHEST PRIORITY
2. Clustering (if predictive not viable)
3. Descriptive (last resort)

When multiple task types are possible, ALWAYS propose predictive tasks first.

**RULE 3: CURRENCY PREFERENCE (INR > USD)**
When dataset has both INR and USD columns:
- **Default:** Use INR (Indian Rupees) columns
- **Exception:** Use USD only if user explicitly mentions international analysis or USD

Example:
- Columns: "Value (INR)", "Value (USD)"
- ✓ Prefer: "Value (INR)"
- ✗ Avoid: "Value (USD)" (unless user requests it)

═══════════════════════════════════════════════════════════════
VALIDATION WORKFLOW (MANDATORY)
═══════════════════════════════════════════════════════════════

For EACH proposed task:

STEP 1: Identify all required columns
  - List target column(s)
  - List feature column(s)
  - List join key column(s) if multiple files

STEP 2: Validate data availability
  - Check Stage 1 summaries for null_fraction
  - Calculate: data_availability = 1 - null_fraction
  - Require: data_availability >= 0.65 for ALL columns
  - If any column fails, REJECT the task or find alternative

STEP 3: Verify task type priority
  - Is this predictive? (if yes, proceed)
  - If not, can it be made predictive? (try to convert)
  - If still not predictive, justify why clustering/descriptive is needed

STEP 4: Check currency preference
  - Are there INR and USD columns?
  - If yes, use INR unless user specifies otherwise

STEP 5: Document validation
  - In task description, mention data availability check
  - State: "All required columns have ≥65% non-NaN data"

═══════════════════════════════════════════════════════════════
TASK PROPOSAL GUIDELINES
═══════════════════════════════════════════════════════════════

Your job in Stage 2 is to explore the available datasets and design **high-quality analytic task proposals** that:

- Are **feasible** given the schema (columns actually exist).
- Have **plausible joins** between datasets:
    * Hypothesized keys must be consistent with column names in the summaries.
    * Do NOT pair columns that do not coexist across the relevant datasets.
- Are **dataset-agnostic**:
    * Never assume domain-specific structure that is not implied by summaries or tool results.
    * Base all reasoning on the summaries, python_sandbox outputs, and (optionally) search().

This phase is **exploration** only:
- You will call tools, reason about what you see, and refine ideas.
- A separate synthesis step will later ask you to output the final JSON proposals.

═══════════════════════════════════════════════════════════════
REACT-STYLE EXPLORATION LOOP
═══════════════════════════════════════════════════════════════

In each exploration step, follow this pattern internally:

1. THOUGHT:
   - Briefly think about what you know and what you still need.
   - Examples:
     * "I know dataset A has yearly columns; I need to see if dataset B has a matching year field."
     * "I know column X is numeric and low-null; it might make a good target."

2. ACTION (choose ONE tool to call):
   - list_summary_files()
       * Use at the beginning to discover all available summaries.
   - read_summary_file(filename=...)
       * Use to inspect specific datasets in detail.
   - python_sandbox(code=...)
       * Use for heavier analysis such as:
           + building a mapping: dataset_name → list of columns
           + computing candidate join keys between pairs of datasets
           + checking overlaps of column names or suggested primary keys
   - search(query=..., within="project" | "data" | "code" | "output" | "all")
       * Use to discover how these datasets have been joined or analyzed previously.

3. OBSERVATION:
   - The tool will be executed and its result will be fed back into the conversation.
   - Use that result in your next THOUGHT.

═══════════════════════════════════════════════════════════════
TOOL-CALLING PROTOCOL (IMPORTANT)
═══════════════════════════════════════════════════════════════

In this exploration phase, you do NOT yet output final proposals.

Instead, in each step, you must output exactly ONE Python dict literal describing
a single tool call, for example:

    {"tool_name": "list_summary_files", "tool_args": {}}

or:

    {"tool_name": "read_summary_file", "tool_args": {"filename": "some_file.summary.json"}}

Valid tool_name values in this phase:
- "list_summary_files"
- "read_summary_file"
- "python_sandbox"
- "search"

Inside python_sandbox, you can also access the previous tool output as `result`, `last_result`, or `last_tool_result`.
Do NOT paste entire summary JSON blobs into python_sandbox code; instead call read_summary_file(...) or use result/last_result.

You MAY precede this dict with some natural language, or wrap it in ```python ...``` fences;
the orchestration code will extract the dict. But the dict itself MUST be valid Python syntax.

═══════════════════════════════════════════════════════════════
JOIN-AWARE EXPLORATION (VERY IMPORTANT)
═══════════════════════════════════════════════════════════════

As you explore, you must form a mental model of **which datasets can be joined and how**:

- For each dataset summary you inspect, keep track of:
    * dataset_name
    * list of column names
    * candidate_primary_keys (if any)

- When considering joins between datasets:
    * Use explicit overlaps in column names as your starting point for join keys.
    * You MAY also consider compatible logical types and patterns (e.g., "year", "state_code"),
      but you must still anchor them in real columns from the summaries.

- A **join key set** like ["col1", "col2"] is only valid if:
    * BOTH datasets you intend to join have **all** of these columns.
    * The columns are not obviously different concepts (e.g., one is numeric year, other is text category),
      unless the summaries suggest they represent the same concept.

- It is ILLEGAL (for later stages) to propose a join key list where:
    * Some columns exist only in dataset A and others only in dataset B.
      (e.g., ["col_only_in_A", "col_only_in_B"] is invalid.)
    * Columns do not appear at all in a dataset's summary.

If you are unsure whether a join is feasible:
- Prefer to mark the join as ambiguous in your notes.
- Avoid inventing join keys that are not supported by the summaries.
- It is better to propose a single-dataset task than to rely on a fabricated multi-dataset join.

═══════════════════════════════════════════════════════════════
GOAL FOR THE SYNTHESIS PHASE
═══════════════════════════════════════════════════════════════

All of this exploration will feed into a later synthesis step where you must propose:

- 3–8 TaskProposals
- Each with:
    * A well-motivated analytic question
    * A clear category (predictive / descriptive / unsupervised)
    * A feasible set of required_files
    * A join_plan whose hypothesized_keys obey the join rules above
    * A realistic target and feature_plan
    * Validation and quality-check ideas

In this exploration phase, do NOT output proposals directly.
Focus on building a correct, join-aware understanding of the data.

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
                "Do not include explanation or markdown. Do NOT paste raw summary JSON; use read_summary_file(...) or result/last_result instead."
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
            # Make last tool output accessible to the sandbox for convenience
            tools.LAST_TOOL_RESULT = result
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

REMINDER: MANDATORY VALIDATION RULES
- ✓ ALL columns must have ≥65% non-NaN data (1 - null_fraction ≥ 0.65)
- ✓ PREFER predictive tasks (forecasting, regression, classification)
- ✓ When INR and USD both exist, USE INR (unless user specifies otherwise)
- ✓ Document data validation in problem_statement: "All columns verified ≥65% complete"
- ✓ DOCUMENT EXCLUDED COLUMNS: Any column you considered but rejected must be listed in excluded_columns

You MUST output a SINGLE STRICT JSON object with the following structure:

{
  "proposals": [
    {
      "id": "TSK-001",
      "category": "predictive" | "descriptive" | "unsupervised",
      "title": "short human-readable title",
      "problem_statement": "2–5 sentences explaining the analytic question and why it matters. MUST mention: 'All columns verified ≥65% data completeness'",
      "required_files": ["filename1.csv", "filename2.csv"],
      "join_plan": {
        "hypothesized_keys": [
          ["col1"],
          ["col1", "col2"]
        ],
        "notes": "brief commentary about join logic and any doubts"
      },
      "target": {
        "name": "column name or null",
        "granularity": ["columns that define a prediction/aggregation unit"] or null,
        "horizon": "forecast horizon like '1-year ahead' or null"
      },
      "feature_plan": {
        "candidates": ["pattern-*", "explicit_column_name", "..."],
        "transform_ideas": ["lagged features", "growth rates", "aggregations", "..."],
        "handling_missingness": "brief strategy for NA values"
      },
      "validation_plan": "how to evaluate or sanity-check this task",
      "quality_checks": [
        "simple checks to avoid leakage or broken joins",
        "..."
      ],
      "excluded_columns": [
        {
          "column_name": "Price_USD",
          "file": "export_data.csv",
          "reason": "Only 45% non-NaN data, below 65% threshold. Using Price_INR instead."
        },
        {
          "column_name": "Legacy_ID",
          "file": "production.csv",
          "reason": "90% missing data, unusable for analysis"
        }
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

═══════════════════════════════════════════════════════════════
STRICT JSON REQUIREMENTS
═══════════════════════════════════════════════════════════════

- Use double quotes for all keys and string values.
- Use true / false / null for booleans and missing values.
- No comments, no trailing commas, no Python None/True/False.
- The top-level object MUST have exactly one key: "proposals".
- It is OK (but not required) to wrap your JSON in ```json ... ``` fences;
  if you do, the content inside MUST still be valid JSON.

Do NOT wrap your answer in extra natural language.

═══════════════════════════════════════════════════════════════
JOIN CONSISTENCY RULES FOR join_plan.hypothesized_keys
═══════════════════════════════════════════════════════════════

For EACH proposal:

1. Use ONLY information from:
   - the dataset summaries you read with read_summary_file(),
   - any python_sandbox analyses you ran (e.g., column lists, overlaps),
   - and (optionally) search() results showing existing join logic.

2. For each proposal, consider the set of required_files.
   - For any pair of files you intend to join, you must have at least one
     plausible set of join keys or you must mark the join as ambiguous in notes.

3. Each entry in hypothesized_keys is a list of column names, like:
   - ["col1"] OR ["col1", "col2"].

   It MUST satisfy:
   - Every column name in that list exists in ALL datasets you intend to join using that key set.
     (You can infer existence only from summaries / your own tool outputs.)
   - You MUST NOT construct a single key list that mixes columns that live in different tables
     (e.g. ["col_only_in_file_A", "col_only_in_file_B"] is invalid).

4. If there is NO common set of columns that exists across the relevant datasets:
   - Set hypothesized_keys to [].
   - In join_plan.notes, explicitly say that the join is ambiguous or not safely defined
     based on the available summaries.
   - It is better to have no hypothesized_keys than to propose impossible joins.

5. You are allowed to propose:
   - Single-dataset tasks (required_files has length 1, hypothesized_keys may be []).
   - Multi-dataset tasks where joins are tentative, as long as:
       * you do NOT fabricate keys that contradict the summaries, and
       * you clearly explain the uncertainty in join_plan.notes.

═══════════════════════════════════════════════════════════════
TARGET, FEATURES, AND VALIDATION
═══════════════════════════════════════════════════════════════

- category:
   * "predictive": tasks with a clear target column and prediction objective.
   * "descriptive": EDA-style tasks with no explicit target.
   * "unsupervised": clustering / dimensionality reduction / segmentation tasks.

- target:
   * For predictive tasks, target.name MUST be an existing numeric or categorical column.
   * granularity should list columns that define a single row / prediction unit.
   * horizon is a free-text description if there is a time component, otherwise null.

- feature_plan:
   * candidates can mix explicit column names and wildcard patterns ("prefix-*", "*_suffix").
   * transform_ideas should reference generic operations (lags, ratios, aggregates).
   * handling_missingness should be short but concrete.

- validation_plan and quality_checks:
   * Describe how you would evaluate the task (train/test split, time-based split, etc.).
   * Include checks that explicitly mention:
       - verifying join row counts and duplicate keys,
       - checking nulls on key columns,
       - avoiding data leakage where applicable.

═══════════════════════════════════════════════════════════════
NUMBER AND VARIETY OF PROPOSALS
═══════════════════════════════════════════════════════════════

- Produce between 3 and 8 proposals total.
- If the user provided a specific request/context earlier, ensure at least one proposal directly
  addresses it.
- Prefer proposals that:
   * are feasible given the schemas,
   * exercise different types of analysis (predictive, descriptive, unsupervised),
   * and use joins only when the summaries clearly support them.

Output ONLY the JSON object described above, with no extra explanation.
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

def run_stage2(user_query: Optional[str] = None) -> Stage2Output:
    """Run Stage 2: Task proposal generation.
    
    Returns:
        Stage2Output with all task proposals
    """
    print("\n" + "=" * 80)
    print("STAGE 2: Task Proposal Generation")
    print("=" * 80)
    
    # Run exploration
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

    initial_state: AgentState = {
        "messages": messages,
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
    stage2_output = run_stage2(user_query=state.get("user_query"))
    
    state["task_proposals"] = stage2_output.proposals
    state["completed_stages"].append(2)
    state["current_stage"] = 3
    
    print(f"\n✅ Stage 2 complete: Generated {len(stage2_output.proposals)} task proposals")
    
    return state


if __name__ == "__main__":
    # Run Stage 2 standalone
    run_stage2()
