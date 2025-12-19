<div align="center">

# 🧠 Agent Implementations
### *ReAct Agents Powered by LangGraph*

</div>

This directory contains the core intelligence of the system. Each agent is designed to handle a specific stage of the data science workflow using the **ReAct (Reason + Act)** pattern.

---

## 📁 Files Overview

| File | Role | Description |
|---|---|---|
| `master_orchestrator.py` | 👑 **Orchestrator** | Coordinates all agents, manages pipeline state, and ensures data passing. |
| `conversation_agent.py` | 🗣️ **Interface** | Handles user interactions, intent classification, and routing. |
| `eda_agent.py` | 🔍 **Explorer** | Exploratory data analysis with autonomous code generation. |
| `stage1_agent.py` | 📊 **Analyst** | Dataset analysis and summary generation. |
| `stage2_agent.py` | 💡 **Strategist** | Task proposal generation based on user intent. |
| `stage3_agent.py` | 🗺️ **Planner** | Execution plan creation with feature engineering. |
| `stage3_5b_agent.py` | 🧪 **Scientist** | Benchmarking multiple ML methods (Cross-Validation). |
| `stage4_agent.py` | ⚡ **Executor** | Final model execution and prediction generation. |
| `stage5_agent.py` | 🎨 **Artist** | Publication-quality visualization generation. |
| `stage6_agent.py` | ✍️ **Author** | Comprehensive markdown report generation. |
| `config.py` | ⚙️ **Config** | Centralized configuration for LLMs and paths. |

---

## 🔄 Agent Architecture

All agents follow a standardized **ReAct** pattern:

> **Thought** → **Action** (Tool Call) → **Observation** (Result) → **Answer**

```python
# Simplified Agent Structure
agent = create_react_agent(
    model=llm,                     # Qwen/GPT-4/Llama-3
    tools=stage_tools,             # Stage-specific tools
    system_prompt=system_prompt    # Specialized instructions
)

# Execution Loop
while not done:
    thought = agent.reason(state)
    action = agent.select_action(thought)
    result = tool.execute(action)
    state = update_state(result)
```

---

## 🎯 Master Orchestrator

The `ConversationalOrchestrator` is the central nervous system.

1.  **Session Management**: Tracks total conversation state.
2.  **Intent Routing**: Decides if user wants to *Plan*, *Execute*, or *Talk*.
3.  **State Persistence**: Saves checkpoints (`.json`) after every stage.

---

## ⚙️ Configuration

Key settings in `config.py` you might want to tune:

```python
# LLM Configuration
PRIMARY_LLM_CONFIG = {
    "base_url": LLM_BASE_URL,
    "model": "Qwen/Qwen3-32B-Instruct",  # Enhanced reasoning
    "temperature": 0.0,                   # Deterministic output
    "max_tokens": 4096,
}

# Iteration Limits
STAGE_MAX_ROUNDS = {
    "stage1": 1,       # Fast pass
    "stage2": 15,      # Allow thinking
    "stage3": 100,     # Complex planning
    # ...
}
```

---

## 📝 Extending the System

Want to add a new agent?

1.  Create `stage_new_agent.py`
2.  Define a **System Prompt** describing its role.
3.  Create corresponding tools in `tools/stage_new_tools.py`.
4.  Register it in `master_orchestrator.py`.
