#!/usr/bin/env python3
"""Test the conversational agent with a simple query"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_code.run_conversational import main
import sys

# Override sys.argv to simulate command-line args
sys.argv = [
    "run_conversational.py",
    "--query",
    "suggest some potential analyses",
]

# Run the main function
exitcode = main()
print(f"\nExit code: {exitcode}")
