#!/bin/bash
# Test Stage 3B Data Preparation Agent

echo "=================================================="
echo "Testing Stage 3B Data Preparation Agent"
echo "=================================================="
echo ""

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate llm

echo "Test 1: Verify Stage 3B imports"
python -c "
from agentic_code.stage3b_agent import stage3b_node
from agentic_code.tools import STAGE3B_TOOLS
print('✓ Stage 3B agent imports successfully')
print(f'✓ Stage 3B has {len(STAGE3B_TOOLS)} tools')
"

echo ""
echo "Test 2: Verify master pipeline integration"
python -c "
from agentic_code.master_agent import build_master_graph
graph = build_master_graph()
nodes = list(graph.get_graph().nodes.keys())
assert 'stage3b' in nodes, 'stage3b not in pipeline!'
print('✓ Stage 3B integrated into master pipeline')
print(f'✓ Pipeline nodes: {nodes}')
"

echo ""
echo "Test 3: Run Stage 3B standalone on PLAN-TSK-001"
cd /scratch/ziv_baretto/llmserve/final_code
python -m agentic_code.stage3b_agent PLAN-TSK-001

echo ""
echo "=================================================="
echo "Stage 3B Tests Complete!"
echo "=================================================="
echo ""
echo "Check results in:"
echo "  output/stage3b_data_prep/prep_PLAN-TSK-001_*.json"
echo "  output/stage3b_data_prep/prepared_PLAN-TSK-001.parquet"
echo ""
