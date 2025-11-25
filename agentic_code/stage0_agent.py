"""
Stage 0: Conversational Agent (Query Interpretation & Routing)

Serves as the entry point for the conversational interface.
Interprets natural language queries and routes them to appropriate pipeline stages.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .config import get_llm_config, STAGE3_MAX_ROUNDS
from .tools import STAGE0_TOOLS
from .models import PipelineState, ConversationState

# ===========================
# LLM & Tools
# ===========================

from langchain_openai import ChatOpenAI

llm_config = get_llm_config()
llm = ChatOpenAI(**llm_config)
tools = STAGE0_TOOLS
llm_with_tools = llm.bind_tools(tools)


# ===========================
# System Prompt
# ===========================

SYSTEM_PROMPT = """You are a Conversational Data Analysis Agent. Your goal is to help users analyze their data through natural language.

You have access to a powerful 5-stage data pipeline:
- Stage 1: Dataset Summarization (Profiling)
- Stage 2: Task Proposal (Suggesting what can be predicted)
- Stage 3: Planning (Creating execution plans)
- Stage 4: Execution (Running code)
- Stage 5: Visualization (Creating charts)

CORE RESPONSIBILITIES:

1. INTERPRET: Understand if the user wants to explore data, make a prediction, or visualize results.
2. ROUTE: Use `trigger_pipeline_stages` to run the necessary parts of the pipeline.
3. EXPLAIN: Always explain what you are doing and summarize results in plain English.

INTERACTION FLOWS (CRITICAL):

A. PREDICTION REQUESTS ("Predict sales for 2024")
   1. FIRST, check if you have task proposals. If not, run Stage 1-2: `trigger_pipeline_stages(1, 2)`
   2. STOP and present the available proposals to the user. ASK them to choose one.
      "I found these prediction tasks: [List tasks]. Which one matches your goal?"
   3. ONCE USER SELECTS A TASK: Run Stage 3-4 with that task ID: `trigger_pipeline_stages(3, 4, task_id='TSK-XXX')`

B. DATA EXPLORATION ("What data do we have?")
   1. Use `query_data_capabilities()` to see what's available.
   2. If no data is summarized, run Stage 1: `trigger_pipeline_stages(1, 1)`
   3. PROACTIVELY SUGGEST next steps: "I can check for missing values, or generate prediction proposals. What would you like?"

C. CUSTOM ANALYSIS ("Correlation between X and Y")
   1. Use `execute_dynamic_analysis` to write and run custom Python code.
   2. Explain the findings clearly.

STATE MANAGEMENT:
- Use `get_conversation_context` to remember what has happened.
- Use `save_conversation_state` to persist important context.

Refuse to answer questions unrelated to data analysis.
"""

# ===========================
# Graph Nodes
# ===========================

def agent_node(state: Dict):
    """Core agent node that processes messages and decides actions."""
    messages = state.get("messages", [])
    
    # Ensure system message is present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    # Invoke LLM
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def should_continue(state: Dict) -> Literal["tools", END]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    return END


# ===========================
# Build Graph
# ===========================

# Define state for the graph (standard LangGraph message state)
class AgentState(Dict):
    messages: List[Any]

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

# Set entry point
builder.set_entry_point("agent")

# Add edges
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
builder.add_edge("tools", "agent")

# Compile
memory = MemorySaver()
stage0_app = builder.compile(checkpointer=memory)


# ===========================
# Runner
# ===========================

def run_conversational_turn(user_query: str, thread_id: str = "default") -> str:
    """Run a single turn of the conversation.
    
    Args:
        user_query: The user's natural language question
        thread_id: Session ID for memory
        
    Returns:
        The agent's final text response
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Add user message
    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }
    
    # Run graph
    final_state = stage0_app.invoke(initial_state, config=config)
    
    # Extract final response
    last_message = final_state["messages"][-1]
    return last_message.content


if __name__ == "__main__":
    # Simple CLI test
    print("🤖 Conversational Agent (Stage 0)")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            response = run_conversational_turn(query)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
