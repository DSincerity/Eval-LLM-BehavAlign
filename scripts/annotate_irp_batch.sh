#!/bin/bash

# Batch IRP annotation script for all model conversations and KODIS dataset
#
# Usage:
#   # Run with default settings
#   ./scripts/annotate_irp_batch.sh
#
#   # Specify custom data directory
#   DATA_DIR=/path/to/data ./scripts/annotate_irp_batch.sh
#
#   # Use different OpenAI model
#   MODEL=gpt-4o-mini ./scripts/annotate_irp_batch.sh

set -e  # Exit on error

# Configuration
DATA_DIR="${DATA_DIR:-data}"
MODEL="${MODEL:-gpt-4o}"
ANNOTATION_BASE_DIR="${DATA_DIR}/IRP_Annotation"

echo "========================================="
echo "Batch IRP Annotation"
echo "========================================="
echo "Data directory: ${DATA_DIR}"
echo "Model: ${MODEL}"
echo "Annotation base directory: ${ANNOTATION_BASE_DIR}"
echo "========================================="

# Create annotation base directory
mkdir -p "${ANNOTATION_BASE_DIR}"

# Model names
MODELS=("gpt-4.1" "gpt-4.1-mini" "claude-3-7-sonnet-20250219" "gemini-2.0-flash")

# Annotate model conversations
for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing: ${model}"
    echo "========================================="

    # Find emotion-annotated file
    EMO_FILE=$(find "${DATA_DIR}" -name "${model}-merged_*_emo.json" | head -n 1)

    if [ -z "${EMO_FILE}" ]; then
        echo "⚠️  Emotion-annotated file not found for ${model}, skipping"
        continue
    fi

    echo "Input file: ${EMO_FILE}"

    # Set output directory
    OUTPUT_DIR="${ANNOTATION_BASE_DIR}/${model}_annotations"

    echo "Output directory: ${OUTPUT_DIR}"

    # Run IRP annotation
    echo "Step 1: Running IRP annotation..."
    python scripts/annotate_irp.py \
        --input "${EMO_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --data-type model \
        --model "${MODEL}"

    if [ $? -eq 0 ]; then
        echo "✓ IRP annotation completed for ${model}"

        # Determine output filename
        OUTPUT_FILE="${EMO_FILE/_emo.json/_emo_irp.json}"

        echo "Step 2: Merging annotations..."
        python scripts/merge_irp.py \
            --input "${EMO_FILE}" \
            --annotation-dir "${OUTPUT_DIR}" \
            --output "${OUTPUT_FILE}" \
            --data-type model

        if [ $? -eq 0 ]; then
            echo "✓ Merge completed for ${model}"
            echo "✓ Final file: ${OUTPUT_FILE}"
        else
            echo "✗ Merge failed for ${model}"
        fi
    else
        echo "✗ IRP annotation failed for ${model}"
    fi
done

# Annotate KODIS dataset
echo ""
echo "========================================="
echo "Processing: KODIS Dataset"
echo "========================================="

KODIS_EMO_FILE=$(find "${DATA_DIR}" -name "KODIS_combined_dialogues_emo*.json" -not -name "*_irp.json" | head -n 1)

if [ -n "${KODIS_EMO_FILE}" ]; then
    echo "Input file: ${KODIS_EMO_FILE}"

    OUTPUT_DIR="${ANNOTATION_BASE_DIR}/KODIS_annotations"
    echo "Output directory: ${OUTPUT_DIR}"

    echo "Step 1: Running IRP annotation..."
    python scripts/annotate_irp.py \
        --input "${KODIS_EMO_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --data-type kodis \
        --model "${MODEL}" \
        --majority-voting 5

    if [ $? -eq 0 ]; then
        echo "✓ IRP annotation completed for KODIS"

        OUTPUT_FILE="${KODIS_EMO_FILE/.json/_irp.json}"

        echo "Step 2: Merging annotations..."
        python scripts/merge_irp.py \
            --input "${KODIS_EMO_FILE}" \
            --annotation-dir "${OUTPUT_DIR}" \
            --output "${OUTPUT_FILE}" \
            --data-type kodis \
            --combine-same-speaker

        if [ $? -eq 0 ]; then
            echo "✓ Merge completed for KODIS"
            echo "✓ Final file: ${OUTPUT_FILE}"
        else
            echo "✗ Merge failed for KODIS"
        fi
    else
        echo "✗ IRP annotation failed for KODIS"
    fi
else
    echo "⚠️  KODIS emotion-annotated file not found, skipping"
fi

echo ""
echo "========================================="
echo "Batch IRP Annotation Complete"
echo "========================================="
echo "All annotations saved to: ${ANNOTATION_BASE_DIR}/"
