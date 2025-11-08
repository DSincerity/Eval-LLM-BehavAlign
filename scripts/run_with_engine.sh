#!/bin/bash
# Flexible script to run simulation with custom engine configuration

# Default values
N_EXP=100
N_ROUND=25
LOG_PATH="logs"
AGENT1_ENGINE="gpt-4o-mini"
AGENT2_ENGINE="gpt-4o-mini"
VERBOSE=1
VERSION="default"

# Display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --engine1 ENGINE        Agent 1 engine (default: gpt-4o-mini)"
    echo "  --engine2 ENGINE        Agent 2 engine (default: gpt-4o-mini)"
    echo "  --n_exp N               Number of experiments (default: 100)"
    echo "  --n_round N             Number of rounds per experiment (default: 25)"
    echo "  --log_path PATH         Log directory path (default: logs)"
    echo "  --verbose LEVEL         Verbosity level 0-1 (default: 1)"
    echo "  --version VER           Version identifier (default: default)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Available engines:"
    echo "  - gpt-4o-mini"
    echo "  - claude-3-7-sonnet-20250219"
    echo "  - gemini-2.0-flash"
    echo ""
    echo "Example:"
    echo "  $0 --engine1 gpt-4o-mini --engine2 claude-3-7-sonnet-20250219 --n_exp 50"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --engine1)
            AGENT1_ENGINE="$2"
            shift 2
            ;;
        --engine2)
            AGENT2_ENGINE="$2"
            shift 2
            ;;
        --n_exp)
            N_EXP="$2"
            shift 2
            ;;
        --n_round)
            N_ROUND="$2"
            shift 2
            ;;
        --log_path)
            LOG_PATH="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "======================================"
echo "L2L Negotiation Simulation"
echo "======================================"
echo "Agent 1: $AGENT1_ENGINE"
echo "Agent 2: $AGENT2_ENGINE"
echo "Experiments: $N_EXP"
echo "Rounds: $N_ROUND"
echo "Log path: $LOG_PATH"
echo "Version: $VERSION"
echo "Start time: $(date +'%Y/%m/%d %H:%M:%S')"
echo "======================================"

cd "$(dirname "$0")/.."

python scripts/run_simulation.py \
    --n_exp=$N_EXP \
    --n_round=$N_ROUND \
    --log_path=$LOG_PATH \
    --agent_1_engine=$AGENT1_ENGINE \
    --agent_2_engine=$AGENT2_ENGINE \
    --verbose=$VERBOSE \
    --ver=$VERSION

EXIT_CODE=$?

echo "======================================"
echo "End time: $(date +'%Y/%m/%d %H:%M:%S')"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: ✓ Success"
else
    echo "Status: ✗ Failed (exit code: $EXIT_CODE)"
fi
echo "======================================"

exit $EXIT_CODE
