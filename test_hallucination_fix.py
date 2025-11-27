#!/usr/bin/env python3
"""
Test if the hallucination fix works (Conversational Only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.run_conversational import run_conversational_turn

# Use a fresh thread ID
thread_id = "test-hallucination-fix-v2"

print("=" * 80)
print("Testing Hallucination Fix (Conversational)")
print("=" * 80)

print("[TEST] User: generate new tasks for rice forecasting")

# We'll use a mock query that triggers the "new tasks" flow
response = run_conversational_turn(
    "generate new tasks for rice forecasting",
    thread_id=thread_id
)
print(f"\n[Agent Response]:\n{response}\n")

# Check if response matches the actual proposals
# We expect TSK-001, TSK-002, TSK-003 and "Rice" in the response
if "TSK-001" in response and "Rice" in response:
    print("✅ TEST PASSED: Agent used actual proposal details")
else:
    print("⚠️  TEST FAILED: Agent might be hallucinating or ignoring details")
    print("Expected TSK-001 and 'Rice' in response")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
