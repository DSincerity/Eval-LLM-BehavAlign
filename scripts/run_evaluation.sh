#!/bin/bash
# Evaluation Script for L2L Negotiation Models
# Runs comprehensive evaluation on generated datasets

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODELS=()
AUTO_DETECT=false
LIWC_DIR="data/LIWC"
KODIS_LIWC=""
METRICS="all"
OUTPUT_DIR="evaluation_results"
USE_CACHE=true

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

print_info() {
    echo -e "${NC}$1${NC}"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run evaluation on L2L negotiation simulation results.

Model Selection (choose one):
  --model MODEL             Single model to evaluate (e.g., gpt-4o-mini)
  --models MODEL1 MODEL2... Multiple models to evaluate
  --auto-detect             Auto-detect all models in data/complete/

LIWC Configuration:
  --liwc-dir DIR            Directory containing LIWC CSV files (default: data/LIWC)
  --kodis-liwc FILE         Path to KODIS LIWC CSV file (default: data/LIWC/LIWC_22_Aggregated_KODIS.csv)

For each model, expects LIWC file at:
  {liwc-dir}/LIWC_22_Aggregated_{model}.csv

Evaluation Options:
  --metrics METRICS         Metrics to run: anger, strategic, linguistic, entrainment, all (default: all)
  --output-dir DIR          Directory to save results (default: evaluation_results)
  --no-cache                Force recomputation, ignore cached results

Other Options:
  -h, --help                Show this help message

Examples:
  # Auto-detect and evaluate all models
  $0 --auto-detect

  # Evaluate specific model
  $0 --model gpt-4o-mini

  # Evaluate multiple models
  $0 --models gpt-4o-mini claude-3-7-sonnet-20250219

  # Evaluate with custom LIWC directory
  $0 --auto-detect --liwc-dir /path/to/liwc/files

  # Run only anger and strategic metrics
  $0 --auto-detect --metrics anger strategic

  # Force recomputation without cache
  $0 --model gpt-4o-mini --no-cache

Required Data Files:
  1. data/complete/{model}_complete.json         - Complete annotated dataset
  2. data/LIWC/LIWC_22_Aggregated_{model}.csv   - Model LIWC analysis (for linguistic metrics)
  3. data/LIWC/LIWC_22_Aggregated_KODIS.csv     - Human baseline LIWC (for linguistic metrics)

Note: If LIWC files are missing, linguistic gap metrics (LG-Dispute, LG-IRP) will be skipped.
      See README.md for instructions on generating LIWC CSV files.

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODELS=("$2")
            shift 2
            ;;
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --auto-detect)
            AUTO_DETECT=true
            shift
            ;;
        --liwc-dir)
            LIWC_DIR="$2"
            shift 2
            ;;
        --kodis-liwc)
            KODIS_LIWC="$2"
            shift 2
            ;;
        --metrics)
            shift
            METRICS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                METRICS="$METRICS $1"
                shift
            done
            METRICS=$(echo $METRICS | xargs)  # Trim whitespace
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-cache)
            USE_CACHE=false
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

# Auto-detect models if requested
if [ "$AUTO_DETECT" = true ]; then
    print_step "Auto-detecting Models"
    MODELS=()
    if [ -d "data/complete" ]; then
        for file in data/complete/*_complete.json; do
            if [ -f "$file" ]; then
                model=$(basename "$file" | sed 's/_complete.json$//')
                MODELS+=("$model")
            fi
        done
    fi

    if [ ${#MODELS[@]} -eq 0 ]; then
        print_error "No models found in data/complete/"
        print_info "Please run data generation first: ./scripts/generate_data.sh --model <MODEL>"
        exit 1
    fi

    print_success "Detected ${#MODELS[@]} model(s): ${MODELS[*]}"
    echo ""
fi

# Validate that we have models to evaluate
if [ ${#MODELS[@]} -eq 0 ]; then
    print_error "No models specified"
    print_info "Use --model, --models, or --auto-detect to specify models"
    usage
fi

# Set default KODIS LIWC path if not specified
if [ -z "$KODIS_LIWC" ]; then
    KODIS_LIWC="${LIWC_DIR}/LIWC_22_Aggregated_KODIS.csv"
fi

# Print configuration
echo ""
print_step "L2L Negotiation Evaluation"
echo ""
echo "Configuration:"
echo "  Models: ${MODELS[*]}"
echo "  Metrics: $METRICS"
echo "  LIWC Directory: $LIWC_DIR"
echo "  KODIS LIWC: $KODIS_LIWC"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Cache: $([ "$USE_CACHE" = true ] && echo "Enabled" || echo "Disabled")"
echo ""

# Check data files for each model
print_step "Checking Data Files"
MISSING_DATA=false
MISSING_LIWC=false

for model in "${MODELS[@]}"; do
    # Check complete data file
    COMPLETE_FILE="data/complete/${model}_complete.json"
    if [ -f "$COMPLETE_FILE" ]; then
        print_success "Found: $COMPLETE_FILE"
    else
        print_error "Missing: $COMPLETE_FILE"
        MISSING_DATA=true
    fi

    # Check LIWC file (only warn, don't fail) - use glob pattern to match *_sample.csv etc
    LIWC_PATTERN="${LIWC_DIR}/LIWC_22_Aggregated_${model}*.csv"
    LIWC_FILE=$(ls $LIWC_PATTERN 2>/dev/null | head -1)
    if [ -n "$LIWC_FILE" ]; then
        print_success "Found: $LIWC_FILE"
    else
        print_warning "Missing: ${LIWC_DIR}/LIWC_22_Aggregated_${model}*.csv"
        MISSING_LIWC=true
    fi
done

# Check KODIS LIWC (only warn, don't fail) - use glob pattern
KODIS_LIWC_PATTERN="${LIWC_DIR}/LIWC_22_Aggregated_KODIS*.csv"
KODIS_LIWC_FILE=$(ls $KODIS_LIWC_PATTERN 2>/dev/null | head -1)
if [ -n "$KODIS_LIWC_FILE" ]; then
    print_success "Found: $KODIS_LIWC_FILE"
else
    print_warning "Missing: ${LIWC_DIR}/LIWC_22_Aggregated_KODIS*.csv"
    MISSING_LIWC=true
fi

echo ""

# Exit if critical data is missing
if [ "$MISSING_DATA" = true ]; then
    print_error "Critical data files are missing!"
    print_info "Please run data generation first: ./scripts/generate_data.sh --model <MODEL>"
    exit 1
fi

# Warn about missing LIWC files
if [ "$MISSING_LIWC" = true ]; then
    print_warning "Some LIWC files are missing"
    print_info "Linguistic Gap metrics (LG-Dispute, LG-IRP) will be skipped"
    print_info "See README.md section 'LIWC Analysis' for instructions on generating LIWC files"
    echo ""
fi

# Build python command
CMD="python scripts/run_evaluation.py"

# Add models
for model in "${MODELS[@]}"; do
    CMD="$CMD --models $model"
done

# Add LIWC directory
CMD="$CMD --liwc_dir $LIWC_DIR"

# Add metrics if not all
if [ "$METRICS" != "all" ]; then
    CMD="$CMD --metrics $METRICS"
else
    CMD="$CMD --metrics all"
fi

# Add output directory
CMD="$CMD --output_dir $OUTPUT_DIR"

# Add cache option
if [ "$USE_CACHE" = false ]; then
    CMD="$CMD --no_cache"
fi

# Run evaluation
print_step "Running Evaluation"
echo ""
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    print_step "Evaluation Complete!"
    echo ""
    print_success "Results saved to: $OUTPUT_DIR/"
    echo ""
    print_info "View the report:"
    LATEST_REPORT=$(ls -t $OUTPUT_DIR/evaluation_report_*.md 2>/dev/null | head -1)
    if [ -n "$LATEST_REPORT" ]; then
        echo "  cat $LATEST_REPORT"
    fi
    echo ""
else
    print_error "Evaluation failed"
    exit 1
fi
