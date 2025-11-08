#!/bin/bash
# Batch experiment script - runs simulations with same engine for both agents

echo "======================================"
echo "Batch L2L Experiments (Same Engine)"
echo "Start time: $(date +'%Y/%m/%d %H:%M:%S')"
echo "======================================"

cd "$(dirname "$0")/.."

# Configuration
N_EXP=100
N_ROUND=25
LOG_PATH="logs"

# Available engines
ENGINES=("gpt-4o-mini" "claude-3-7-sonnet-20250219" "gemini-2.0-flash")

# Counter for tracking experiments
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Run experiments with same engine for both agents
for ENGINE in "${ENGINES[@]}"; do
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    echo ""
    echo "======================================"
    echo "Run $TOTAL_RUNS: $ENGINE vs $ENGINE"
    echo "======================================"

    python scripts/run_simulation.py \
        --n_exp=$N_EXP \
        --n_round=$N_ROUND \
        --log_path=$LOG_PATH \
        --agent_1_engine=$ENGINE \
        --agent_2_engine=$ENGINE \
        --verbose=1 \
        --ver=batch

    if [ $? -eq 0 ]; then
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        echo "✓ $ENGINE vs $ENGINE - SUCCESS"
    else
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "✗ $ENGINE vs $ENGINE - FAILED"
    fi
done

echo ""
echo "======================================"
echo "Batch Experiments Complete"
echo "End time: $(date +'%Y/%m/%d %H:%M:%S')"
echo "======================================"
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $SUCCESSFUL_RUNS"
echo "Failed: $FAILED_RUNS"
echo "======================================"

if [ $FAILED_RUNS -eq 0 ]; then
    exit 0
else
    exit 1
fi
