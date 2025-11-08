#!/bin/bash
# Batch emotion annotation script
# Annotates emotions for all model conversations and KODIS dataset

set -e  # Exit on error

echo "======================================================================"
echo "Batch Emotion Annotation"
echo "======================================================================"

# Configuration
DEVICE="${DEVICE:-cpu}"  # Default to CPU, override with env var
MODEL_TYPE="${MODEL_TYPE:-emoberta-base}"  # Default to base model
DATA_DIR="${DATA_DIR:-data}"

echo "Device: $DEVICE"
echo "Model: $MODEL_TYPE"
echo "Data directory: $DATA_DIR"
echo ""

# Function to annotate a file
annotate_file() {
    local input_file=$1
    local data_type=$2

    if [ -f "$input_file" ]; then
        echo "----------------------------------------------------------------------"
        echo "Annotating: $(basename $input_file)"
        echo "----------------------------------------------------------------------"

        python scripts/annotate_emotions.py \
            --input "$input_file" \
            --data-type "$data_type" \
            --device "$DEVICE" \
            --model-type "$MODEL_TYPE"

        echo "✓ Completed: $(basename $input_file)"
        echo ""
    else
        echo "⚠ File not found: $input_file"
        echo ""
    fi
}

# Models to process
MODELS=(
    "gpt-4.1"
    "gpt-4.1-mini"
    "claude-3-7-sonnet-20250219"
    "gemini-2.0-flash"
)

# Annotate model conversations
echo "======================================================================"
echo "Annotating Model Conversations"
echo "======================================================================"
echo ""

for model in "${MODELS[@]}"; do
    # Try different file patterns
    for pattern in "${DATA_DIR}/${model}-merged_*.json" "${DATA_DIR}/${model}_*.json"; do
        for file in $pattern; do
            if [ -f "$file" ] && [[ ! "$file" =~ _emo\.json$ ]]; then
                annotate_file "$file" "model"
                break 2  # Break both loops once file is found
            fi
        done
    done
done

# Annotate KODIS dataset
echo "======================================================================"
echo "Annotating KODIS Dataset"
echo "======================================================================"
echo ""

# Try different KODIS file patterns
KODIS_PATTERNS=(
    "${DATA_DIR}/KODIS-merged_*.json"
    "${DATA_DIR}/KODIS_combined_*.json"
    "${DATA_DIR}/KODIS.json"
)

for pattern in "${KODIS_PATTERNS[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ] && [[ ! "$file" =~ _emo\.json$ ]]; then
            annotate_file "$file" "kodis"
            break 2
        fi
    done
done

echo "======================================================================"
echo "Batch Annotation Complete!"
echo "======================================================================"
echo ""
echo "Annotated files are saved with '_emo.json' suffix in the same directory."
echo ""
