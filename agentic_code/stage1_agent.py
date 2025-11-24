"""
Stage 1: Dataset Summarization Agent

Profiles CSV files and generates structured summaries using an LLM.
This stage does not use LangGraph - it's a straightforward batch process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.utils.json import parse_partial_json
from langchain_openai import ChatOpenAI

from .config import DATA_DIR, SUMMARIES_DIR, PRIMARY_LLM_CONFIG
from .models import DatasetSummary
from .utils import profile_csv, extract_json_block


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**PRIMARY_LLM_CONFIG)

parser = PydanticOutputParser(pydantic_object=DatasetSummary)

system_prompt = """
You are a meticulous data profiling assistant.

You receive a *machine-generated profile* of a tabular dataset:
- Each column has a physical dtype, a guessed logical type, null_fraction, unique_fraction, and example values.

Your job:
1. Refine the logical type for each column (choose from: numeric, integer, float,
   categorical, text, datetime, boolean, unknown).
2. Write a short, precise description for each column based on its name and examples.
3. Decide if the column is nullable.
4. Mark columns that are plausible keys (e.g., id, code, combination of state+year).
5. Propose candidate_primary_keys: each entry is a list of column names that
   could form a primary key (unique identifier for rows).
6. Add a short 'notes'field if there is anything non-obvious or suspicious. Always include a short notes string summarizing any quirks

You MUST output a JSON object that matches the DatasetSummary schema.
Output ONLY JSON. Do NOT add any explanation, heading, or surrounding text.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt + "\n\n{format_instructions}"),
        (
            "human",
            "Here is the machine-generated profile for one dataset:\n\n{profile_json}"
        ),
    ]
).partial(format_instructions=parser.get_format_instructions())


# ===========================
# Core Functions
# ===========================

def summarize_profile(profile: Dict[str, Any]) -> DatasetSummary:
    """Generate LLM summary for a dataset profile.
    
    Args:
        profile: Machine-generated profile dictionary
        
    Returns:
        Validated DatasetSummary
    """
    profile_json = json.dumps(profile, indent=2)

    # Build messages from prompt
    messages = prompt.format_messages(profile_json=profile_json)

    # Call LLM
    resp = llm.invoke(messages)
    raw_text = resp.content if hasattr(resp, "content") else str(resp)

    # Extract JSON substring (strip any "Here is..." preface)
    json_str = extract_json_block(raw_text)

    # Be tolerant to slight truncation using parse_partial_json
    data = parse_partial_json(json_str)

    # Validate against Pydantic schema
    summary = DatasetSummary.model_validate(data)
    return summary


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
