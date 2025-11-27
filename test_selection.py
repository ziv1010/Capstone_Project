#!/usr/bin/env python3
"""Test the improved conversational agent with a selection scenario"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.stage0_agent import stage0_app
from langchain_core.messages import HumanMessage

# Test with a fresh thread ID 
config = {"configurable": {"thread_id": "test-selection-456"}}

print("=" * 80)
print("Testing conversational agent with selection handling")
print("=" * 80)

# First query - ask for proposals
print("\n[User]: can you give me proposals for rice forecasting")
state1 = stage0_app.invoke(
    {"messages": [HumanMessage(content="can you give me proposals for rice forecasting")]},
    config=config
)
response1 = state1["messages"][-1].content
print(f"\n[Agent]: {response1[:300]}...")

# Second query - user selects option 3 (new task)
print("\n\n[User]: 3")
state2 = stage0_app.invoke(
    {"messages": [HumanMessage(content="3")]},
    config=config
)

# Check if any tools were called
print("\n\n--- Analyzing response ---")
for msg in state2["messages"][-3:]:
    msg_type = type(msg).__name__
    print(f"\nMessage type: {msg_type}")
    
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"  Tool calls: {[tc.get('name') for tc in msg.tool_calls]}")
    
    if hasattr(msg, 'content') and msg.content:
        content = msg.content[:200] if len(msg.content) > 200 else msg.content
        print(f"  Content: {content}")

response2 = state2["messages"][-1].content
print(f"\n\n[Final Agent Response]: {response2[:400]}...")
