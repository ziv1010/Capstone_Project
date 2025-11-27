#!/usr/bin/env python3
"""
Test if the conversational agent lists tasks correctly when asked
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.run_conversational import run_conversational_turn

# Use a fresh thread ID
thread_id = "test-menu-listing"

print("=" * 80)
print("Testing Task Listing in Menu")
print("=" * 80)

# Query that should trigger the menu
print("\n[User]: what tasks are available")
response = run_conversational_turn(
    "what tasks are available",
    thread_id=thread_id
)
print(f"\n[Agent Response]:\n{response}\n")

# Check if response lists specific tasks
if "TSK-001" in response and "TSK-002" in response:
    print("✅ TEST PASSED: Agent listed specific tasks")
else:
    print("⚠️  TEST FAILED: Agent did not list specific tasks")
    print("Expected TSK-001, TSK-002 in response")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
