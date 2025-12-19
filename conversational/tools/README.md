<div align="center">

# 🔧 Tool Implementations
### *Capabilities for AI Agents*

</div>

This directory defines the **Actions** that agents can take. Each function is exposed as a tool to the LLM.

---

## 📁 Tool Suite Overview

| File | Capabilities Provided |
|---|---|
| `stage1_tools.py` | `load_dataset`, `analyze_dataset`, `save_summary` |
| `stage2_tools.py` | `generate_task_proposals`, `validate_proposal` |
| `stage3_tools.py` | `create_execution_plan`, `define_features` |
| `stage3_5b_tools.py` | `run_benchmark_code`, `compare_methods`, `save_results` |
| `stage4_tools.py` | `execute_model`, `generate_predictions` |
| `stage5_tools.py` | `create_visualization`, `save_plot` |
| `stage7_guardrails_tools.py` | `validate_model`, `check_fairness` |
| `eda_tools.py` | `run_eda_code`, `create_plot`, `query_data` |

---

## 🛠️ Tool Structure

We use LangChain's `@tool` decorator to expose Python functions to the LLM.

```python
from langchain_core.tools import tool

@tool
def save_visualization(
    plot_path: str,
    title: str,
    description: str
) -> str:
    """
    Save a matplotlib visualization to disk.
    
    Args:
        plot_path: Path to save the plot
        title: Title of the visualization
        description: Description for the report
    
    Returns:
        Confirmation message with saved path
    """
    # ... implementation ...
    return f"Saved visualization to {plot_path}"
```

---

## 🔑 Critical Tools

### 🧪 Benchmarking (Stage 3.5B)
```python
run_benchmark_code(code: str, plan_id: str, method_id: str) -> str
```
> Executes dynamically generated Python code to benchmark a specific ML method (e.g., XGBoost) via Cross-Validation. Saves the model if successful.

### ⚡ Execution (Stage 4)
```python
execute_winning_method(plan_id: str, method_id: str) -> str
```
> Loads the best-performing model checkpoint from Stage 3.5B and runs it on the full dataset to generate final predictions.

### 🎨 Visualization (Stage 5)
```python
create_visualization(viz_type: str, data_path: str, output_path: str) -> str
```
> Generates publication-ready plots (Residuals, Feature Importance, Actual vs Predicted) using Matplotlib/Seaborn.

---
