#!/bin/bash
# Test all available LLM engines individually
# This script tests each engine to verify API keys and connectivity

echo "=========================================="
echo "Testing All LLM Engines"
echo "Start time: $(date +'%Y/%m/%d %H:%M:%S')"
echo "=========================================="

cd "$(dirname "$0")/.."

# Available engines
ENGINES=(
    "gpt-4o-mini"
    "claude-3-7-sonnet-20250219"
    "gemini-2.0-flash"
)

# Tracking
TOTAL=0
SUCCESS=0
FAILED=0

# Test each engine
for ENGINE in "${ENGINES[@]}"; do
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "=========================================="
    echo "Test $TOTAL: $ENGINE vs $ENGINE"
    echo "=========================================="

    python scripts/run_simulation.py \
        --n_exp=1 \
        --n_round=5 \
        --log_path logs/test \
        --agent_1_engine "$ENGINE" \
        --agent_2_engine "$ENGINE" \
        --verbose=0 \
        --ver=engine_test

    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "✓ $ENGINE: SUCCESS"
    else
        FAILED=$((FAILED + 1))
        echo "✗ $ENGINE: FAILED"
    fi
done

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total tests: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "End time: $(date +'%Y/%m/%d %H:%M:%S')"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "✓ All engines working!"
    exit 0
else
    echo "✗ Some engines failed. Check logs in logs/test/"
    exit 1
fi
