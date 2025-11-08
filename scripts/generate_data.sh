#!/bin/bash
# Simulation & Annotation Pipeline for L2L Negotiation
#
# This script runs the complete pipeline:
#   1. Simulation: Generate LLM negotiation conversations
#   2. Emotion Annotation: Label utterances with emotions (EmoBERTa)
#   3. IRP Strategy Annotation: Label utterances with IRP strategies (GPT-4)
#   4. Entrainment Calculation: Calculate linguistic entrainment values (nCLiD)
#
# Output: Fully annotated dataset ready for behavioral alignment evaluation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL=""
N_EXP=250
N_ROUND=10
SKIP_SIMULATION=false
SKIP_EMOTION=false
SKIP_IRP=false
SKIP_ENTRAINMENT=false
KEEP_INTERMEDIATE=false

# Function to print colored messages
print_step() {
    echo -e "${BLUE}===================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 --model MODEL [OPTIONS]

Run simulation and multi-level annotation pipeline for L2L negotiation.

Required Arguments:
  --model MODEL             LLM model to use (e.g., gpt-4o-mini, claude-3-7-sonnet-20250219)

Optional Arguments:
  --n-exp N                 Number of negotiation dialogues to simulate (default: 250)
  --n-round N               Maximum rounds per negotiation (default: 10)
  --skip-simulation         Skip simulation step (use existing conversations)
  --skip-emotion            Skip emotion annotation step
  --skip-irp                Skip IRP strategy annotation step
  --skip-entrainment        Skip linguistic entrainment calculation step
  --keep-intermediate       Keep intermediate annotation files (default: auto-clean)
  -h, --help                Show this help message

Examples:
  # Run full pipeline: Simulation + Emotion + IRP annotation (250 dialogues)
  $0 --model gpt-4o-mini

  # Run with 100 conversations and 5 rounds max
  $0 --model gpt-4o-mini --n-exp 100 --n-round 5

  # Only run IRP annotation on existing emotion-annotated data
  $0 --model gpt-4o-mini --skip-simulation --skip-emotion

  # Keep intermediate IRP annotation files for inspection
  $0 --model gpt-4o-mini --keep-intermediate

Pipeline Steps:
  1. Simulation (LLM)                 → data/simulations/{model}.json
  2. Emotion Annotation (EmoBERTa)    → data/emotions/{model}_emo.json
  3. IRP Annotation (GPT-4)           → data/complete/{model}_complete.json
  4. Entrainment Calculation (nCLiD)  → data/linguistic_entrainment/LE_values_{model}.csv

Final Output:
  data/complete/{model}_complete.json + data/linguistic_entrainment/LE_values_{model}.csv
  ↳ Fully annotated dataset ready for behavioral alignment evaluation

Intermediate Files:
  .cache/IRP_Annotation/{model}_annotations/ (auto-cleaned unless --keep-intermediate)

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --n-exp)
            N_EXP="$2"
            shift 2
            ;;
        --n-round)
            N_ROUND="$2"
            shift 2
            ;;
        --skip-simulation)
            SKIP_SIMULATION=true
            shift
            ;;
        --skip-emotion)
            SKIP_EMOTION=true
            shift
            ;;
        --skip-irp)
            SKIP_IRP=true
            shift
            ;;
        --skip-entrainment)
            SKIP_ENTRAINMENT=true
            shift
            ;;
        --keep-intermediate)
            KEEP_INTERMEDIATE=true
            shift
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

# Validate required arguments
if [ -z "$MODEL" ]; then
    print_error "Model name is required"
    usage
fi

# Print configuration
echo ""
print_step "L2L Simulation & Annotation Pipeline"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dialogues to simulate: $N_EXP"
echo "  Max rounds per dialogue: $N_ROUND"
echo "  Keep intermediate files: $([ "$KEEP_INTERMEDIATE" = true ] && echo "YES" || echo "NO")"
echo ""
echo "Pipeline steps:"
echo "  1. LLM Simulation:          $([ "$SKIP_SIMULATION" = true ] && echo "SKIP" || echo "RUN")"
echo "  2. Emotion Annotation:      $([ "$SKIP_EMOTION" = true ] && echo "SKIP" || echo "RUN")"
echo "  3. IRP Strategy Annotation: $([ "$SKIP_IRP" = true ] && echo "SKIP" || echo "RUN")"
echo "  4. Entrainment Calculation: $([ "$SKIP_ENTRAINMENT" = true ] && echo "SKIP" || echo "RUN")"
echo ""

# Step 1: Run Simulation
if [ "$SKIP_SIMULATION" = false ]; then
    print_step "Step 1/4: Running Simulation"
    python scripts/run_simulation.py \
        --agent_1_engine "$MODEL" \
        --agent_2_engine "$MODEL" \
        --n_exp "$N_EXP" \
        --n_round "$N_ROUND"

    if [ $? -eq 0 ]; then
        print_success "Simulation completed: data/simulations/${MODEL}.json"
    else
        print_error "Simulation failed"
        exit 1
    fi
else
    print_warning "Skipping simulation step"
    if [ ! -f "data/simulations/${MODEL}.json" ]; then
        print_error "Simulation file not found: data/simulations/${MODEL}.json"
        exit 1
    fi
    print_success "Found existing simulation: data/simulations/${MODEL}.json"
fi

# Step 2: Emotion Annotation
if [ "$SKIP_EMOTION" = false ]; then
    print_step "Step 2/4: Running Emotion Annotation"
    python scripts/annotate_emotions.py \
        --model "$MODEL"

    if [ $? -eq 0 ]; then
        print_success "Emotion annotation completed: data/emotions/${MODEL}_emo.json"
    else
        print_error "Emotion annotation failed"
        exit 1
    fi
else
    print_warning "Skipping emotion annotation step"
    if [ ! -f "data/emotions/${MODEL}_emo.json" ]; then
        print_error "Emotion file not found: data/emotions/${MODEL}_emo.json"
        exit 1
    fi
    print_success "Found existing emotion annotation: data/emotions/${MODEL}_emo.json"
fi

# Step 3: IRP Annotation & Merge (combined into single step)
if [ "$SKIP_IRP" = false ]; then
    print_step "Step 3/4: Running IRP Annotation"

    # Use cache directory for intermediate files
    CACHE_DIR=".cache/IRP_Annotation/${MODEL}_annotations"

    # Run annotation with cache directory
    python scripts/annotate_irp.py \
        --model-name "$MODEL" \
        --output-dir "$CACHE_DIR"

    if [ $? -ne 0 ]; then
        print_error "IRP annotation failed"
        exit 1
    fi

    # Merge annotations directly to complete file
    python scripts/merge_irp.py \
        --model-name "$MODEL" \
        --annotation-dir "$CACHE_DIR"

    if [ $? -eq 0 ]; then
        print_success "IRP annotation completed: data/complete/${MODEL}_complete.json"

        # Clean up intermediate files unless --keep-intermediate is specified
        if [ "$KEEP_INTERMEDIATE" = false ]; then
            print_warning "Cleaning up intermediate annotation files..."
            rm -rf "$CACHE_DIR"
            print_success "Intermediate files removed"
        else
            print_warning "Keeping intermediate files in: $CACHE_DIR"
        fi
    else
        print_error "IRP merge failed"
        exit 1
    fi
else
    print_warning "Skipping IRP annotation step"
    if [ ! -f "data/complete/${MODEL}_complete.json" ]; then
        print_error "Complete file not found: data/complete/${MODEL}_complete.json"
        exit 1
    fi
    print_success "Found existing complete file: data/complete/${MODEL}_complete.json"
fi

# Step 4: Linguistic Entrainment Calculation
if [ "$SKIP_ENTRAINMENT" = false ]; then
    print_step "Step 4/4: Calculating Linguistic Entrainment"

    python scripts/calculate_entrainment.py \
        --model "$MODEL" \
        --input "data/complete/${MODEL}_complete.json"

    if [ $? -eq 0 ]; then
        print_success "Entrainment calculation completed: data/linguistic_entrainment/LE_values_${MODEL}.csv"
    else
        print_error "Entrainment calculation failed"
        exit 1
    fi
else
    print_warning "Skipping entrainment calculation step"
    if [ ! -f "data/linguistic_entrainment/LE_values_${MODEL}.csv" ]; then
        print_warning "Entrainment file not found: data/linguistic_entrainment/LE_values_${MODEL}.csv"
        print_warning "LEG metric will not be available in evaluation"
    else
        print_success "Found existing entrainment file: data/linguistic_entrainment/LE_values_${MODEL}.csv"
    fi
fi

# Print final summary
echo ""
print_step "Simulation & Annotation Complete!"
echo ""
echo "Generated Files:"
echo "  1. data/simulations/${MODEL}.json                           - LLM negotiation conversations"
echo "  2. data/emotions/${MODEL}_emo.json                          - + Emotion labels (7 classes)"
echo "  3. data/complete/${MODEL}_complete.json                     - + IRP strategy labels (9 types)"
echo "  4. data/linguistic_entrainment/LE_values_${MODEL}.csv       - + Linguistic entrainment (nCLiD)"
echo ""
print_success "Fully annotated dataset ready for behavioral alignment evaluation!"
echo ""
echo "Next steps:"
echo "  1. (Optional) Generate LIWC CSV for linguistic gap metrics"
echo "     See README.md section 'LIWC Analysis' for instructions"
echo "  2. Run behavioral alignment evaluation:"
echo "     ./scripts/run_evaluation.sh --model ${MODEL}"
echo ""
