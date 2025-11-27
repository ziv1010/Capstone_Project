#!/usr/bin/env python3
"""
Test if the conversational agent lists tasks when user selects '2'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.run_conversational import run_conversational_turn

# Use a fresh thread ID
thread_id = "test-menu-selection-2"

print("=" * 80)
print("Testing '2' Selection Flow")
print("=" * 80)

# First query to get the menu
print("\n[User]: hello")
run_conversational_turn("hello", thread_id=thread_id)

# Second query - select option 2
print("\n[User]: 2")
response = run_conversational_turn("2", thread_id=thread_id)
print(f"\n[Agent Response]:\n{response}\n")

# Check if response lists specific tasks
if "TSK-001" in response:
    print("✅ TEST PASSED: Agent listed specific tasks after selecting '2'")
else:
    print("⚠️  TEST FAILED: Agent did not list specific tasks")
    print("Expected TSK-001 in response")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
