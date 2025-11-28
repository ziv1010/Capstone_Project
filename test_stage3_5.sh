#!/bin/bash
# Test Stage 3.5 Tester Agent with existing Stage 3 output

echo "=================================================="
echo "Testing Stage 3.5 Tester Agent"
echo "=================================================="
echo ""

# Activate environment
echo "1. Activating micromamba environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate llm

echo ""
echo "2. Running Stage 3.5 with PLAN-TSK-001..."
echo ""

# Run Stage 3.5 agent as a module (fixes import issues)
cd /scratch/ziv_baretto/llmserve/final_code
python -m agentic_code.stage3_5_agent PLAN-TSK-001

echo ""
echo "=================================================="
echo "Stage 3.5 Test Complete!"
echo "=================================================="
echo ""
echo "Check results in:"
echo "  output/stage3_5_tester/tester_PLAN-TSK-001_*.json"
echo ""
