"""
Configuration module for the unified agentic AI pipeline.

Centralizes all paths, LLM settings, and constants used across all 5 stages.
"""

from pathlib import Path
from typing import Optional

# ===========================
# Project Paths
# ===========================
# Directory Paths
# ===========================

PROJECT_ROOT = Path("/scratch/ziv_baretto/llmserve").resolve()
FINAL_CODE_ROOT = PROJECT_ROOT / "final_code"
DATA_DIR = PROJECT_ROOT / "data"

# Output directories - all under final_code/output
OUTPUT_ROOT = FINAL_CODE_ROOT / "output"
SUMMARIES_DIR = OUTPUT_ROOT / "summaries"
STAGE2_OUT_DIR = OUTPUT_ROOT / "stage2_out"
STAGE3_OUT_DIR = OUTPUT_ROOT / "stage3_out"
STAGE4_OUT_DIR = OUTPUT_ROOT / "stage4_out"
STAGE5_OUT_DIR = OUTPUT_ROOT / "stage5_out"

# Working directories for code execution
STAGE4_WORKSPACE = STAGE4_OUT_DIR / "code_workspace"
STAGE5_WORKSPACE = STAGE5_OUT_DIR / "viz_workspace"

# Create all output directories
for dir_path in [OUTPUT_ROOT, SUMMARIES_DIR, STAGE2_OUT_DIR, STAGE3_OUT_DIR, STAGE4_OUT_DIR, STAGE5_OUT_DIR, STAGE4_WORKSPACE, STAGE5_WORKSPACE]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ===========================
# LLM Configuration
# ===========================

# LLM Configuration
# Currently using only Qwen model on port 8001 for all stages
PRIMARY_LLM_CONFIG = {
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "base_url": "http://127.0.0.1:8001/v1",
    "api_key": "EMPTY",
    "temperature": 0.0,
    "max_tokens": 8192,
}

SECONDARY_LLM_CONFIG = {
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "base_url": "http://127.0.0.1:8001/v1",
    "api_key": "EMPTY",
    "temperature": 0.0,
    "max_tokens": 8192,
}

# ===========================
# Stage-Specific Settings
# ===========================

# Stage 1: Dataset Summarization
STAGE1_SAMPLE_ROWS = 5000  # Number of rows to sample for profiling

# Stage 2: Task Proposal
STAGE2_MAX_EXPLORATION_STEPS = 10  # Max tool calls during exploration

# Stage 3: Planning
STAGE3_MAX_ROUNDS = 15  # Max rounds for planning agent

# Stage 4: Execution
STAGE4_MAX_ROUNDS = 25  # Max rounds for execution agent

# Stage 5: Visualization
STAGE5_MAX_ROUNDS = 20  # Max rounds for visualization agent

# ===========================
# Helper Functions
# ===========================

def get_llm_config(use_secondary: bool = False) -> dict:
    """Get LLM configuration.
    
    Args:
        use_secondary: Whether to use the secondary (more capable) LLM
        
    Returns:
        Dictionary with LLM configuration
    """
    return SECONDARY_LLM_CONFIG if use_secondary else PRIMARY_LLM_CONFIG


def print_config():
    """Print current configuration for debugging."""
    print("=" * 80)
    print("AGENTIC PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"\nOutput Directories:")
    print(f"  - Summaries: {SUMMARIES_DIR}")
    print(f"  - Stage 2: {STAGE2_OUT_DIR}")
    print(f"  - Stage 3: {STAGE3_OUT_DIR}")
    print(f"  - Stage 4: {STAGE4_OUT_DIR}")
    print(f"  - Stage 5: {STAGE5_OUT_DIR}")
    print(f"\nLLM Configuration:")
    print(f"  - Primary Model: {PRIMARY_LLM_CONFIG['model']}")
    print(f"  - Secondary Model: {SECONDARY_LLM_CONFIG['model']}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
