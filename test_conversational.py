#!/usr/bin/env python3
"""
Quick test script to debug the conversational agent
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.stage0_agent import stage0_app
from langchain_core.messages import HumanMessage

# Test with a fresh thread ID to avoid memory issues
config = {"configurable": {"thread_id": "test-debug-123"}}

# Simple query
initial_state = {
    "messages": [HumanMessage(content="suggest some potential analyses")]
}

print("=" * 80)
print("Testing conversational agent with debug output")
print("=" * 80)

# Run with streaming to see what's happening
state = {}
for i, s in enumerate(stage0_app.stream(initial_state, config=config, stream_mode="values")):
    print(f"\n--- Step {i} ---")
    messages = s.get("messages", [])
    if messages:
        last_msg = messages[-1]
        print(f"Type: {type(last_msg).__name__}")
        
        # Print content if available
        if hasattr(last_msg, 'content'):
            content = last_msg.content
            if content:
                print(f"Content: {content[:200]}...")
        
        # Print tool calls if any
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            print(f"Tool calls: {len(last_msg.tool_calls)}")
            for tc in last_msg.tool_calls:
                print(f"  - {tc.get('name', 'unknown')}")
    
    state = s

print("\n" + "=" * 80)
print("Final messages:")
print("=" * 80)
for i, msg in enumerate(state.get("messages", [])):
    print(f"\n{i}. {type(msg).__name__}")
    if hasattr(msg, 'content'):
        print(f"   Content: {msg.content[:150]}...")
