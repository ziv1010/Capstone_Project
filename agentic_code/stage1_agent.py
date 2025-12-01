"""
Stage 1: Dataset Summarization Agent

Uses direct function calls for profiling with LLM enhancement for descriptions.
Deterministic execution prevents loops while LLM provides intelligent insights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI

from .config import DATA_DIR, SUMMARIES_DIR, STAGE1_SAMPLE_ROWS, SECONDARY_LLM_CONFIG
from .models import DatasetSummary, ColumnSummary
from .utils import profile_csv


# ===========================
# LLM Setup
# ===========================

llm = ChatOpenAI(**SECONDARY_LLM_CONFIG)


# ===========================
# Core Profiling Logic (Deterministic)
# ===========================

def create_basic_description(col_name: str, logical_type: str) -> str:
    """Generate a basic description for a column."""
    if logical_type == 'categorical':
        return f"Categorical variable: {col_name}"
    elif logical_type in ['integer', 'float', 'numeric']:
        return f"Numeric variable: {col_name}"
    elif 'date' in col_name.lower() or 'time' in col_name.lower() or 'year' in col_name.lower():
        return f"Temporal variable: {col_name}"
    else:
        return f"Variable: {col_name}"


def enhance_column_descriptions(summary: DatasetSummary) -> DatasetSummary:
    """Use LLM to generate intelligent descriptions for columns."""
    print(f"  🤖 Enhancing column descriptions with LLM...")
    
    try:
        # Prepare compact column info for LLM
        columns_info = []
        for col in summary.columns:
            columns_info.append({
                "name": col.name,
                "type": col.logical_type,
                "examples": col.examples[:3]
            })
        
        prompt = f"""Analyze this dataset and provide concise descriptions for each column.

Dataset: {summary.dataset_name}

Columns:
{json.dumps(columns_info, indent=2)}

Provide a brief description for each column explaining what it represents.

IMPORTANT: Respond ONLY with valid JSON in EXACTLY this format (no other text):
{{
  "column_descriptions": {{
    "Crop": "Name of the agricultural crop",
    "Area": "Cultivated area in hectares",
    ...
  }},
  "dataset_inference": "Brief summary of what this dataset contains"
}}"""

        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Remove thinking tags if present
        if "<think>" in response_text:
            # Extract content after </think> tag
            think_end = response_text.find("</think>")
            if think_end != -1:
                response_text = response_text[think_end + 8:].strip()
        
        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        # If still not JSON, try to find JSON object in the text
        if not response_text.startswith("{"):
            start_idx = response_text.find("{")
            if start_idx != -1:
                # Find matching closing brace
                brace_count = 0
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == "{":
                        brace_count += 1
                    elif response_text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            response_text = response_text[start_idx:i+1]
                            break
        
        result = json.loads(response_text)
        
        # Update column descriptions
        desc_map = result.get("column_descriptions", {})
        updated_count = 0
        for col in summary.columns:
            if col.name in desc_map and desc_map[col.name]:
                col.description = desc_map[col.name]
                updated_count += 1
        
        # Update inferences
        if result.get("dataset_inference"):
            summary.inferences = result["dataset_inference"]
        
        print(f"  ✅ Enhanced {updated_count}/{len(summary.columns)} column descriptions")
        
    except json.JSONDecodeError as e:
        print(f"  ⚠️  LLM returned invalid JSON: {e}")
        print(f"     Response preview: {response_text[:200] if 'response_text' in locals() else 'N/A'}...")
        print(f"     Keeping basic descriptions")
    except Exception as e:
        print(f"  ⚠️  LLM enhancement failed: {e}")
        print(f"     Keeping basic descriptions")
    
    return summary


def profile_and_create_summary(csv_path: Path, sample_rows: int = 5000, use_llm: bool = True) -> DatasetSummary:
    """Profile a CSV file and create a DatasetSummary with optional LLM enhancement."""
    print(f"\n📊 Profiling {csv_path.name}...")
    
    # Get raw profile data using deterministic function
    profile = profile_csv(csv_path, sample_rows=sample_rows)
    
    # Convert to ColumnSummary objects
    columns = []
    for col in profile['columns']:
        logical_type = col.get('logical_type_guess', 'unknown')
        col_name = col['name']
        examples = [str(x) for x in col.get('examples', [])[:5]]
        
        col_summary = ColumnSummary(
            name=col_name,
            physical_dtype=col.get('physical_dtype', 'unknown'),
            logical_type=logical_type,
            description=create_basic_description(col_name, logical_type),
            nullable=col.get('null_fraction', 0.0) > 0,
            null_fraction=col.get('null_fraction', 0.0),
            unique_fraction=col.get('unique_fraction', 0.0),
            examples=examples,
            is_potential_key=(col.get('unique_fraction', 0.0) > 0.95 and col.get('null_fraction', 0.0) == 0.0)
        )
        columns.append(col_summary)
    
    # Identify candidate keys
    candidate_keys = []
    for col in columns:
        if col.is_potential_key:
            candidate_keys.append([col.name])
    
    # Create summary
    summary = DatasetSummary(
        dataset_name=csv_path.name,
        path=str(csv_path),
        approx_n_rows=profile.get('n_rows_sampled'),
        columns=columns,
        candidate_primary_keys=candidate_keys,
        notes=f"Profiled {len(columns)} columns from {profile.get('n_rows_sampled')} rows.",
        inferences=None
    )
    
    # Enhance with LLM if requested
    if use_llm:
        summary = enhance_column_descriptions(summary)
    
    return summary


# ===========================
# Main Stage 1 Runner
# ===========================

def run_stage1(
    data_dir: Path = DATA_DIR,
    out_dir: Path = SUMMARIES_DIR,
    pattern: str = "*.csv",
    sample_rows: int = STAGE1_SAMPLE_ROWS,
    use_llm_descriptions: bool = True,
    debug: bool = True,
) -> List[DatasetSummary]:
    """Run Stage 1: Dataset summarization.
    
    Args:
        data_dir: Directory containing CSV files
        out_dir: Directory to save summaries
        pattern: Glob pattern for files to process
        sample_rows: Number of rows to sample for profiling
        debug: Whether to print detailed execution logs
        
    Returns:
        List of DatasetSummary objects
    """
    print("\n" + "=" * 80)
    print("🚀 STAGE 1: Dataset Summarization")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Sample rows: {sample_rows}")
    print("=" * 80)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get list of CSV files
    csv_files = sorted(data_dir.glob(pattern))
    csv_names = [f.name for f in csv_files]
    print(f"\n📁 Found {len(csv_files)} CSV files: {csv_names}")

    if not csv_files:
        print("⚠️  No CSV files found!")
        return []

    # Process each file deterministically
    summaries = []
    for idx, csv_path in enumerate(csv_files, 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(csv_files)}] Processing: {csv_path.name}")
        print(f"{'=' * 80}")
        
        try:
            # Profile and create summary (with LLM enhancement)
            summary = profile_and_create_summary(csv_path, sample_rows=sample_rows, use_llm=use_llm_descriptions)
            
            # Save to disk
            base_name = csv_path.stem
            output_path = out_dir / f"{base_name}.summary.json"
            output_path.write_text(summary.model_dump_json(indent=2))
            
            print(f"✅ Summary saved: {output_path.name}")
            print(f"   - Columns: {len(summary.columns)}")
            print(f"   - Rows sampled: {summary.approx_n_rows}")
            if summary.candidate_primary_keys:
                print(f"   - Candidate keys: {summary.candidate_primary_keys}")
            
            summaries.append(summary)
            
        except Exception as e:
            print(f"❌ ERROR processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"✅ STAGE 1 COMPLETE")
    print(f"{'=' * 80}")
    print(f"📁 Summaries directory: {out_dir}")
    print(f"📊 Processed: {len(summaries)}/{len(csv_files)} datasets")
    
    if len(summaries) < len(csv_files):
        missing = set(csv_names) - {s.dataset_name for s in summaries}
        print(f"⚠️  Failed to process: {missing}")
    
    for s in summaries:
        print(f"  ✓ {s.dataset_name}: {len(s.columns)} columns")
    print("=" * 80)

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
