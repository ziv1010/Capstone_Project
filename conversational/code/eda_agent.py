"""
EDA Agent: Exploratory Data Analysis

This agent intelligently handles user queries about datasets by:
1. Understanding the query and planning the analysis
2. Writing and executing custom Python code
3. Creating visualizations as needed
4. Detecting and offering to summarize new datasets

The agent does NOT use hardcoded analysis - it writes its own code based on the query.
"""

import json
from typing import Dict, Any, Optional, Annotated, List
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import (
    EDA_LLM_CONFIG, EDA_OUT_DIR, EDA_WORKSPACE, DATA_DIR, SUMMARIES_DIR,
    STAGE_MAX_ROUNDS, DataPassingManager, logger
)
from code.models import EDAResponse, EDACodeResult, EDAVisualization, EDAQueryType
from tools.eda_tools import EDA_TOOLS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class EDAState(BaseModel):
    """State for EDA agent."""
    messages: Annotated[list, add_messages] = []
    query: str = ""
    query_type: str = "custom_analysis"
    datasets_analyzed: list = []
    visualizations_created: list = []
    code_executed: list = []
    insights: list = []
    new_datasets_detected: list = []
    iteration: int = 0
    complete: bool = False
    needs_user_confirmation: bool = False
    confirmation_message: str = ""


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

EDA_SYSTEM_PROMPT = """You are an intelligent EDA (Exploratory Data Analysis) Agent.

## Your Role
You help users explore and understand their data by:
1. Answering questions about dataset contents, quality, and structure
2. Computing statistics, correlations, and finding patterns
3. Creating visualizations to illustrate findings
4. Detecting new datasets and asking the user if they want them summarized

## CRITICAL: You Write Your Own Code
You do NOT have hardcoded analysis functions. Instead, you:
1. Understand what the user wants to know
2. Write Python code using execute_analysis_code to perform the analysis
3. Interpret the results and explain them to the user

## ReAct Framework
For each query, follow THOUGHT → ACTION → OBSERVATION:

1. **THOUGHT**: What does the user want to know? What analysis approach should I use?
2. **ACTION**: Use appropriate tools (especially execute_analysis_code for custom analysis)
3. **OBSERVATION**: Interpret results and decide if more analysis is needed

## Available Tools

### Dataset Discovery
- `list_all_datasets`: See all available data and which are new
- `get_dataset_info`: Get details about a specific dataset
- `check_for_new_datasets`: Find datasets that haven't been summarized

### Analysis (Use execute_analysis_code for custom work!)
- `execute_analysis_code`: Run your own Python code for analysis
  - Access: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
  - Data path: DATA_DIR / 'filename.csv'
  - Save plots to: EDA_OUT_DIR / 'filename.png'
- `compute_statistics`: Get descriptive statistics
- `compute_correlation`: Compute correlation matrix
- `find_patterns`: Analyze patterns in a column
- `compare_datasets`: Compare two datasets

### Visualization
- `create_visualization`: Create standard plots (bar, line, scatter, etc.)
- Or use execute_analysis_code for custom visualizations!

### Reporting
- `save_eda_report`: Save your analysis as a report
- `summarize_new_dataset`: Summarize a new dataset (after user confirms!)

## Guidelines

1. **Start by understanding the data**: Use list_all_datasets or get_dataset_info first
2. **Write custom code for complex queries**: Use execute_analysis_code
3. **Create visualizations when helpful**: They make findings clearer
4. **Ask before summarizing new datasets**: Never auto-summarize without permission
5. **Provide insights, not just numbers**: Explain what the results mean

## Example Analysis with execute_analysis_code

For "What is the trend in crop production over the years?":

```python
# Use execute_analysis_code with code like:
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(DATA_DIR / 'crop_data.csv')

# Aggregate by year
yearly_production = df.groupby('Year')['Production'].sum()

# Plot trend
plt.figure(figsize=(12, 6))
plt.plot(yearly_production.index, yearly_production.values, marker='o')
plt.title('Crop Production Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.grid(True, alpha=0.3)
plt.savefig(EDA_OUT_DIR / 'production_trend.png', dpi=150)

print(f"Production range: {yearly_production.min():.0f} to {yearly_production.max():.0f}")
print(f"Growth rate: {(yearly_production.iloc[-1] / yearly_production.iloc[0] - 1) * 100:.1f}%")
```

## New Dataset Handling

When you detect new datasets:
1. Report them to the user
2. ASK: "Would you like me to summarize these new datasets?"
3. WAIT for user confirmation before using summarize_new_dataset

## Remember
- Be helpful and explain findings in plain language
- Write code that handles edge cases (missing values, wrong types)
- Create visualizations that tell a story
- Always verify data exists before analyzing it
"""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def create_eda_agent():
    """Create the EDA agent graph."""
    
    llm = ChatOpenAI(**EDA_LLM_CONFIG)
    llm_with_tools = llm.bind_tools(EDA_TOOLS, parallel_tool_calls=False)
    
    def agent_node(state: EDAState) -> Dict[str, Any]:
        """Main agent reasoning node."""
        messages = state.messages
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=EDA_SYSTEM_PROMPT)] + list(messages)
        
        # Check iteration limit
        max_rounds = STAGE_MAX_ROUNDS.get("eda", 40)
        if state.iteration >= max_rounds:
            return {
                "messages": [AIMessage(content="I've reached my analysis limit. Here are my findings so far. Please ask a more specific question if you need more analysis.")],
                "complete": True
            }
        
        # Get LLM response
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "iteration": state.iteration + 1
        }
    
    def should_continue(state: EDAState) -> str:
        """Determine if we should continue or end."""
        if state.complete:
            return "end"
        
        # Check if waiting for user confirmation
        if state.needs_user_confirmation:
            return "end"
        
        # Check last message for tool calls
        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
        
        return "end"
    
    # Build graph
    builder = StateGraph(EDAState)
    
    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(EDA_TOOLS))
    
    # Set entry point
    builder.set_entry_point("agent")
    
    # Add edges
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    builder.add_edge("tools", "agent")
    
    # Compile with checkpointer
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def run_eda(query: str, session_id: str = None) -> EDAResponse:
    """
    Run EDA analysis for a user query.
    
    Args:
        query: User's natural language query about data
        session_id: Optional session ID for continuity
        
    Returns:
        EDAResponse with answer, code results, visualizations, and insights
    """
    logger.info(f"Starting EDA analysis for query: {query[:100]}...")
    
    # Create agent
    graph = create_eda_agent()
    
    # Create initial message
    initial_message = HumanMessage(content=f"""
Analyze this user query about the data:

"{query}"

Follow the ReAct framework:
1. THOUGHT: What does the user want to know? What's the best approach?
2. ACTION: Use tools to get the answer. Write custom code if needed.
3. OBSERVATION: Interpret results and provide a clear answer.

Remember:
- Use list_all_datasets or get_dataset_info to understand what data is available
- Use execute_analysis_code to write custom analysis code
- Create visualizations when they help explain findings
- If you find new datasets, ASK the user before summarizing them

Start by understanding what data is available, then perform the analysis.
""")
    
    # Run agent
    if session_id is None:
        session_id = f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = {"configurable": {"thread_id": session_id}}
    initial_state = EDAState(messages=[initial_message], query=query)
    
    try:
        final_state = graph.invoke(initial_state, config)
        
        # Extract the response from messages
        answer = ""
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break
        
        # Collect visualizations from tool results
        visualizations = []
        code_results = []
        
        for msg in final_state.get("messages", []):
            # Check for tool messages with visualization paths
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                if 'eda_' in msg.content and '.png' in msg.content:
                    # Extract paths from tool output
                    import re
                    paths = re.findall(r'(/[^\s]+\.png)', msg.content)
                    for path in paths:
                        if Path(path).exists():
                            visualizations.append(EDAVisualization(
                                filepath=path,
                                plot_type="auto",
                                title="EDA Visualization",
                                description="Generated visualization"
                            ))
        
        # Create response
        response = EDAResponse(
            query=query,
            query_type=EDAQueryType.CUSTOM_ANALYSIS,
            answer=answer,
            code_results=code_results,
            visualizations=visualizations,
            datasets_used=final_state.get("datasets_analyzed", []),
            insights=final_state.get("insights", []),
            new_datasets_detected=final_state.get("new_datasets_detected", [])
        )
        
        logger.info(f"EDA analysis complete. Generated {len(visualizations)} visualizations.")
        return response
        
    except Exception as e:
        logger.error(f"EDA analysis failed: {e}")
        return EDAResponse(
            query=query,
            answer=f"I encountered an error during analysis: {e}. Please try a more specific query.",
            query_type=EDAQueryType.CUSTOM_ANALYSIS
        )


# ============================================================================
# EDA NODE FOR PIPELINE INTEGRATION
# ============================================================================

def eda_node(query: str) -> Dict[str, Any]:
    """
    EDA node that can be called from the conversation agent or orchestrator.
    
    Args:
        query: User's EDA query
        
    Returns:
        Dict with response, visualizations, and any actions needed
    """
    response = run_eda(query)
    
    return {
        "response": response.answer,
        "visualizations": [v.filepath for v in response.visualizations],
        "insights": response.insights,
        "new_datasets": response.new_datasets_detected,
        "needs_confirmation": len(response.new_datasets_detected) > 0
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What datasets are available and what do they contain?"
    
    print(f"\n{'='*60}")
    print(f"EDA Query: {query}")
    print('='*60 + "\n")
    
    response = run_eda(query)
    
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(response.answer)
    
    if response.visualizations:
        print("\nVisualizations created:")
        for viz in response.visualizations:
            print(f"  - {viz.filepath}")
    
    if response.new_datasets_detected:
        print(f"\nNew datasets found: {response.new_datasets_detected}")
