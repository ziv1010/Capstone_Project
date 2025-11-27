#!/usr/bin/env python3
"""
Test the full conversational flow after bug fixes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.run_conversational import run_conversational_turn

# Use a fresh thread ID
thread_id = "test-final-flow"

print("=" * 80)
print("Testing Full Conversational Flow After Bug Fixes")
print("=" * 80)

# Test 1: Ask for new proposals
print("\n\n[TEST 1] User: 'generate new tasks for rice forecasting next 5 years'")
response1 = run_conversational_turn(
    "generate new tasks for rice forecasting next 5 years",
    thread_id=thread_id
)
print(f"\n[Agent Response]:\n{response1}\n")

# Check if response mentions the new proposals or shows activity
if "TSK-" in response1 or "task" in response1.lower() or "proposal" in response1.lower():
    print("✅ TEST 1 PASSED: Agent responded with task-related content")
else:
    print("⚠️  TEST 1: Agent response doesn't mention tasks")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
