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

EDA_SYSTEM_PROMPT = """You are an intelligent EDA (Exploratory Data Analysis) Agent that TAKES ACTION.

## CRITICAL RULE: ACT, DON'T ASK
When a user asks a question, you must IMMEDIATELY use tools to answer it.
- NEVER ask "would you like me to...?" - just DO IT
- NEVER ask for clarification if you can infer the intent
- ALWAYS execute code or use tools to get actual answers
- If something fails, try a different approach

## Your Capabilities
You can answer ANY data question by writing and executing Python code.
You have full access to pandas, numpy, matplotlib, seaborn, and scipy.

## How to Answer Questions

### For "What columns/rows/info about dataset X?"
→ Use `get_dataset_info` OR write code:
```python
df = pd.read_csv(DATA_DIR / 'dataset.csv')
print("Columns:", list(df.columns))
print("Shape:", df.shape)
print(df.dtypes)
```

### For "Show me statistics/mean/distribution"
→ Use `compute_statistics` OR write code:
```python
df = pd.read_csv(DATA_DIR / 'dataset.csv')
print(df.describe())
```

### For "Create a plot/chart/visualization"
→ Use `create_visualization` OR write code:
```python
df = pd.read_csv(DATA_DIR / 'dataset.csv')
plt.figure(figsize=(10, 6))
plt.hist(df['column'], bins=30)
plt.title('Distribution')
plt.savefig(EDA_OUT_DIR / 'myplot.png')
```

### For "Find correlations/patterns"  
→ Use `compute_correlation` or `find_patterns` OR write code

### For "Summarize/analyze the dataset"
→ Combine: get info + compute stats + create 2-3 visualizations

## Available Tools

### Quick Tools (use for simple queries)
- `get_dataset_info`: Get columns, types, sample data for a dataset
- `list_all_datasets`: See all available CSV files
- `compute_statistics`: Descriptive stats for numeric columns
- `compute_correlation`: Correlation matrix
- `find_patterns`: Analyze a specific column
- `create_visualization`: Create standard plots

### Power Tool (use for ANY custom work)
- `execute_analysis_code`: Run your own Python code
  - `pd`, `np`, `plt`, `sns` are pre-imported
  - Read data: `df = pd.read_csv(DATA_DIR / 'file.csv')`
  - Save plots: `plt.savefig(EDA_OUT_DIR / 'name.png')`
  - Print results to show them

## Examples of CORRECT Behavior

User: "What columns are in heart.csv?"
You: Use get_dataset_info("heart.csv") → Return the column list directly

User: "How many rows in the data?"
You: Use execute_analysis_code with:
```python
df = pd.read_csv(DATA_DIR / 'heart.csv')
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
```

User: "Show me the distribution of age"
You: Use create_visualization OR execute_analysis_code to create histogram

User: "Find correlations"
You: Use compute_correlation("heart.csv") → Return the correlation matrix

## Examples of WRONG Behavior (DON'T DO THIS)
❌ "Would you like me to show you the columns?"
❌ "Which dataset would you like me to analyze?"
❌ "I can help you with that. What specific information do you need?"
❌ Responding without using any tools

## ReAct Framework
1. THOUGHT: What specific answer does the user want?
2. ACTION: Use tools IMMEDIATELY to get that answer
3. OBSERVATION: Return the results clearly

## File Naming
When saving plots: `{dataset}_{type}_{column}_{timestamp}.png`

## Remember
- The user wants ANSWERS, not questions
- If uncertain about which dataset, try the most likely one
- If uncertain about which column, show available columns
- Always provide actual data/numbers, not just descriptions
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
