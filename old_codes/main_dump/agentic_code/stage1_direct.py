"""
Stage 1: Direct Dataset Summarization (No Agent)

Simple, reliable version that profiles all CSVs and generates summaries directly.
"""

from pathlib import Path
import json
from typing import List

from .config import DATA_DIR, SUMMARIES_DIR, STAGE1_SAMPLE_ROWS
from .models import DatasetSummary, ColumnSummary
from .utils import profile_csv


def infer_logical_type(physical_dtype: str) -> str:
    """Infer logical type from physical dtype."""
    dtype_lower = physical_dtype.lower()
    if 'int' in dtype_lower:
        return 'integer'
    elif 'float' in dtype_lower or 'double' in dtype_lower:
        return 'float'
    elif 'bool' in dtype_lower:
        return 'boolean'
    elif 'datetime' in dtype_lower or 'date' in dtype_lower:
        return 'datetime'
    elif 'object' in dtype_lower or 'string' in dtype_lower:
        # Could be categorical or text - default to categorical
        return 'categorical'
    else:
        return 'unknown'


def create_column_summary(col_data: dict) -> ColumnSummary:
    """Convert raw column data to ColumnSummary."""
    # Ensure examples are strings
    examples = col_data.get('examples', [])
    examples_str = [str(x) for x in examples[:5]]  # Limit to 5 examples

    # Infer logical type
    logical_type = infer_logical_type(col_data.get('physical_dtype', 'unknown'))

    # Create description
    col_name = col_data.get('name', 'unknown')
    null_frac = col_data.get('null_fraction', 0.0)
    unique_frac = col_data.get('unique_fraction', 0.0)

    if unique_frac > 0.95:
        description = f"Unique identifier or highly unique values for {col_name}"
    elif unique_frac < 0.05:
        description = f"Low-cardinality categorical variable: {col_name}"
    elif null_frac > 0.5:
        description = f"Sparse column with many missing values: {col_name}"
    else:
        description = f"{logical_type.capitalize()} values for {col_name}"

    return ColumnSummary(
        name=col_name,
        physical_dtype=col_data.get('physical_dtype', 'unknown'),
        logical_type=logical_type,
        description=description,
        nullable=null_frac > 0,
        null_fraction=null_frac,
        unique_fraction=unique_frac,
        examples=examples_str,
        is_potential_key=unique_frac > 0.95 and null_frac == 0.0,
    )


def identify_candidate_keys(columns: List[ColumnSummary]) -> List[List[str]]:
    """Identify potential primary keys."""
    candidates = []

    # Single column keys
    for col in columns:
        if col.is_potential_key:
            candidates.append([col.name])

    # If no single column keys, look for combinations
    if not candidates:
        # Find columns with low null rates
        low_null_cols = [c for c in columns if c.null_fraction < 0.1]
        if len(low_null_cols) >= 2:
            # Suggest first 2 non-null columns as composite key
            candidates.append([c.name for c in low_null_cols[:2]])

    return candidates


def profile_single_dataset(csv_path: Path, sample_rows: int = STAGE1_SAMPLE_ROWS) -> DatasetSummary:
    """Profile a single CSV file and return a DatasetSummary."""
    print(f"  📊 Profiling: {csv_path.name}")

    # Get raw profile data
    profile_data = profile_csv(csv_path, sample_rows=sample_rows)

    # Convert columns
    columns = [create_column_summary(col) for col in profile_data['columns']]

    # Identify keys
    candidate_keys = identify_candidate_keys(columns)

    # Create summary
    summary = DatasetSummary(
        dataset_name=csv_path.name,
        path=str(csv_path),
        approx_n_rows=profile_data['n_rows_sampled'],
        columns=columns,
        candidate_primary_keys=candidate_keys,
        notes=f"Profiled {len(columns)} columns from {profile_data['n_rows_sampled']} rows.",
    )

    return summary


def run_stage1_direct(
    data_dir: Path = DATA_DIR,
    out_dir: Path = SUMMARIES_DIR,
    pattern: str = "*.csv",
    sample_rows: int = STAGE1_SAMPLE_ROWS,
) -> List[DatasetSummary]:
    """Run Stage 1 directly without agents.

    Args:
        data_dir: Directory containing CSV files
        out_dir: Directory to save summaries
        pattern: Glob pattern for files
        sample_rows: Number of rows to sample

    Returns:
        List of DatasetSummary objects
    """
    print("\n" + "=" * 80)
    print("🚀 STAGE 1: Dataset Summarization (Direct)")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Sample rows: {sample_rows}")
    print("=" * 80)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(data_dir.glob(pattern))
    print(f"\n📁 Found {len(csv_files)} CSV files")

    summaries = []

    for csv_path in csv_files:
        try:
            # Profile the dataset
            summary = profile_single_dataset(csv_path, sample_rows)
            summaries.append(summary)

            # Save to file
            base_name = csv_path.stem  # Filename without extension
            output_path = out_dir / f"{base_name}.summary.json"
            output_path.write_text(summary.model_dump_json(indent=2))

            print(f"  ✅ Saved: {output_path.name}")

        except Exception as e:
            print(f"  ❌ Failed to profile {csv_path.name}: {e}")
            continue

    print(f"\n{'=' * 80}")
    print(f"✅ STAGE 1 COMPLETE")
    print(f"{'=' * 80}")
    print(f"📁 Summaries directory: {out_dir}")
    print(f"📊 Number of summaries: {len(summaries)}")
    for s in summaries:
        print(f"  - {s.dataset_name}: {len(s.columns)} columns")
    print("=" * 80)

    return summaries


if __name__ == "__main__":
    run_stage1_direct()
