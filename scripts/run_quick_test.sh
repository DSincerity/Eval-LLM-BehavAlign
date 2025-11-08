#!/bin/bash
# Quick test script - runs 1 experiment for testing purposes
# Usage: ./scripts/run_quick_test.sh [engine1] [engine2]
# Example: ./scripts/run_quick_test.sh claude-3-7-sonnet-20250219 gemini-2.0-flash

# Default engines
ENGINE1="${1:-gpt-4o-mini}"
ENGINE2="${2:-gpt-4o-mini}"

echo "======================================"
echo "Quick Test (1 experiment, 10 rounds)"
echo "Engine 1: $ENGINE1"
echo "Engine 2: $ENGINE2"
echo "Start time: $(date +'%Y/%m/%d %H:%M:%S')"
echo "======================================"

cd "$(dirname "$0")/.."

python scripts/run_simulation.py \
    --n_exp=1 \
    --n_round=10 \
    --log_path logs/test \
    --agent_1_engine "$ENGINE1" \
    --agent_2_engine "$ENGINE2" \
    --verbose=1 \
    --ver=quick_test

EXIT_CODE=$?

echo "======================================"
echo "End time: $(date +'%Y/%m/%d %H:%M:%S')"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test successful with $ENGINE1 vs $ENGINE2!"
else
    echo "✗ Test failed with $ENGINE1 vs $ENGINE2!"
fi
echo "======================================"

exit $EXIT_CODE
