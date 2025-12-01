"""
Stage 1: Dataset Summarization Agent

Profiles CSV files and generates structured summaries using an LLM with LangGraph.
Uses a two-step process: analyze -> describe for better column descriptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.utils.json import parse_partial_json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .config import DATA_DIR, SUMMARIES_DIR, PRIMARY_LLM_CONFIG, SECONDARY_LLM_CONFIG
from .models import DatasetSummary
from .utils import profile_csv, extract_json_block


# ===========================
# LangGraph State
# ===========================

class Stage1State(TypedDict):
    """State for Stage 1 LangGraph workflow."""
    profile: Dict[str, Any]
    analysis: str  # Analysis of columns and dataset context
    summary: DatasetSummary | None


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)

parser = PydanticOutputParser(pydantic_object=DatasetSummary)

# Step 1: Analysis prompt (thinking step)
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a meticulous data analyst examining a dataset profile.

Your task is to ANALYZE and THINK about the dataset before describing it:

1. **Understand the dataset context**: What domain is this data from? What is the overall purpose?
2. **Analyze each column**:
   - What does this column represent in the real world?
   - How does it relate to other columns?
   - What patterns do you see in the examples?
   - Are there any data quality issues or quirks?
3. **Identify relationships**: Which columns are keys? Which are metrics? Which are dimensions?
4. **Consider the logical types**: Does the inferred type make sense given the examples?

Write a thorough analysis that will help generate accurate column descriptions.
Focus on understanding the MEANING and CONTEXT of each column with respect to the dataset's purpose.

Output your analysis as clear, structured text."""),
    ("human", "Analyze this dataset profile:\n\n{profile_json}")
])

# Step 2: Description prompt (using analysis)
description_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a meticulous data profiling assistant.

You receive:
1. A machine-generated profile of a tabular dataset
2. A detailed analysis of the dataset context and column meanings

Your job:
1. Refine the logical type for each column (choose from: numeric, integer, float,
   categorical, text, datetime, boolean, unknown).
2. Write a short, precise description for each column based on:
   - The column name
   - Example values
   - The analysis context you received
   - How it relates to other columns in the dataset
3. Decide if the column is nullable.
4. Mark columns that are plausible keys (e.g., id, code, combination of state+year).
5. Propose candidate_primary_keys: each entry is a list of column names that
   could form a primary key (unique identifier for rows).
6. Add a short 'notes' field if there is anything non-obvious or suspicious. Always include a short notes string summarizing any quirks.

You MUST output a JSON object that matches the DatasetSummary schema.
Output ONLY JSON. Do NOT add any explanation, heading, or surrounding text.

{format_instructions}"""),
    ("human", """Dataset profile:
{profile_json}

Analysis context:
{analysis}

Now generate the complete DatasetSummary JSON:""")
]).partial(format_instructions=parser.get_format_instructions())


# ===========================
# LangGraph Nodes
# ===========================

def analyze_node(state: Stage1State) -> Stage1State:
    """Analyze the dataset profile to understand context and meaning.

    This node performs deep analysis before generating descriptions.
    """
    profile_json = json.dumps(state["profile"], indent=2)
    messages = analysis_prompt.format_messages(profile_json=profile_json)

    resp = llm.invoke(messages)
    analysis = resp.content if hasattr(resp, "content") else str(resp)

    return {"analysis": analysis}


def describe_node(state: Stage1State) -> Stage1State:
    """Generate structured DatasetSummary using the analysis.

    This node creates the final output with enhanced descriptions.
    """
    profile_json = json.dumps(state["profile"], indent=2)
    messages = description_prompt.format_messages(
        profile_json=profile_json,
        analysis=state["analysis"]
    )

    resp = llm.invoke(messages)
    raw_text = resp.content if hasattr(resp, "content") else str(resp)

    # Extract JSON substring
    json_str = extract_json_block(raw_text)

    # Parse and validate
    data = parse_partial_json(json_str)
    summary = DatasetSummary.model_validate(data)

    return {"summary": summary}


# ===========================
# Build LangGraph
# ===========================

def build_stage1_graph() -> StateGraph:
    """Build the LangGraph workflow for Stage 1."""
    workflow = StateGraph(Stage1State)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("describe", describe_node)

    # Define edges (deterministic flow)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "describe")
    workflow.add_edge("describe", END)

    return workflow.compile()


# ===========================
# Core Functions
# ===========================

# Build the graph once at module level
stage1_graph = build_stage1_graph()


def summarize_profile(profile: Dict[str, Any]) -> DatasetSummary:
    """Generate LLM summary for a dataset profile using LangGraph.

    Args:
        profile: Machine-generated profile dictionary

    Returns:
        Validated DatasetSummary
    """
    # Initialize state
    initial_state: Stage1State = {
        "profile": profile,
        "analysis": "",
        "summary": None
    }

    # Run the graph
    final_state = stage1_graph.invoke(initial_state)

    return final_state["summary"]


def verify_summary(profile: Dict[str, Any], summary: DatasetSummary) -> Dict[str, Any]:
    """Verify that summary matches the profile.
    
    Args:
        profile: Original profile
        summary: Generated summary
        
    Returns:
        Dictionary with verification results
    """
    prof_cols = {c["name"].strip().lower() for c in profile["columns"]}
    sum_cols  = {c.name.strip().lower() for c in summary.columns}

    missing_in_summary = sorted(prof_cols - sum_cols)
    extra_in_summary   = sorted(sum_cols - prof_cols)

    ok = (not missing_in_summary) and (not extra_in_summary)
    return {
        "ok": ok,
        "missing_in_summary": missing_in_summary,
        "extra_in_summary": extra_in_summary,
    }


# ===========================
# Main Stage 1 Runner
# ===========================

def run_stage1(
    data_dir: Path = DATA_DIR,
    out_dir: Path = SUMMARIES_DIR,
    pattern: str = "*.csv",
    sample_rows: int = 5000,
) -> List[DatasetSummary]:
    """Run Stage 1: Dataset summarization.
    
    Args:
        data_dir: Directory containing CSV files
        out_dir: Directory to save summaries
        pattern: Glob pattern for files to process
        sample_rows: Number of rows to sample for profiling
        
    Returns:
        List of DatasetSummary objects
    """
    paths = sorted(data_dir.glob(pattern))
    print(f"Found {len(paths)} CSVs in {data_dir}")
    
    summaries = []
    
    for path in paths:
        print("\n" + "=" * 80)
        print(f"Dataset: {path.name}")

        profile = profile_csv(path, sample_rows=sample_rows)
        summary = summarize_profile(profile)
        check   = verify_summary(profile, summary)

        if not check["ok"]:
            print("WARNING: Column mismatch detected!")
            print("Missing in summary:", check["missing_in_summary"])
            print("Extra in summary  :", check["extra_in_summary"])
        else:
            print("Schema check: OK")

        # Save summary JSON to disk
        out_path = out_dir / f"{path.stem}.summary.json"
        with out_path.open("w") as f:
            f.write(summary.model_dump_json(indent=2))

        print(f"Wrote summary -> {out_path}")
        summaries.append(summary)
    
    return summaries


# ===========================
# State Node for Master Graph
# ===========================

def stage1_node(state: dict) -> dict:
    """Stage 1 node for the master pipeline graph.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state with dataset_summaries populated
    """
    print("\n" + "=" * 80)
    print("STAGE 1: Dataset Summarization")
    print("=" * 80)
    
    summaries = run_stage1()
    
    state["dataset_summaries"] = summaries
    state["completed_stages"].append(1)
    state["current_stage"] = 2
    
    print(f"\n✅ Stage 1 complete: Generated {len(summaries)} summaries")
    
    return state


if __name__ == "__main__":
    # Run Stage 1 standalone
    run_stage1()
