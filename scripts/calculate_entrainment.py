"""Script to calculate Linguistic Entrainment (LE) values for conversation data.

This script calculates nCLiD (normalized Conversational Linguistic Distance) values
for both model-generated conversations and KODIS dataset.
"""
import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics.linguistic_entrainment import (
    calculate_le_values,
    preprocess_kodis_conversation_from_json,
    preprocess_model_conversation
)


def load_word2vec_model():
    """Load word2vec model.

    Returns:
        word2vec model or None if loading fails
    """
    print("Loading word2vec model (this may take a moment)...")
    try:
        import gensim.downloader as api
        word2vec = api.load('word2vec-google-news-300')
        print("  ✓ Word2vec model loaded successfully")
        return word2vec
    except Exception as e:
        print(f"  ✗ Error loading word2vec model: {e}")
        print("\nPlease install gensim:")
        print("  pip install gensim")
        return None


def annotate_model_entrainment(
    input_path: str,
    output_path: str,
    model_name: str,
    k: int = 3
):
    """Calculate LE values for model-generated conversations.

    Args:
        input_path: Path to model conversation JSON file
        output_path: Path to save LE values CSV
        model_name: Name of the model
        k: Number of following utterances to consider
    """
    print(f"\nProcessing {model_name} conversations...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    # Load word2vec model
    word2vec = load_word2vec_model()
    if word2vec is None:
        return False

    # Load conversations
    print(f"\nLoading data from: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Preprocess conversations
    print("Preprocessing conversations...")
    conversations = []

    # Handle complete.json format: single dict with 'conversation' key containing list of conversation lists
    if isinstance(data, dict) and 'conversation' in data:
        # data/complete/*.json format
        conversation_lists = data['conversation']
        for conv_id, turns in enumerate(conversation_lists):
            conv_df = preprocess_model_conversation(turns, conv_id, model_name)
            if len(conv_df) > 0:
                conversations.append(conv_df)
    elif isinstance(data, dict):
        # Handle other dict formats
        for conv_id, conversation in data.items():
            if isinstance(conversation, dict) and 'conversation' in conversation:
                turns = conversation['conversation']
            elif isinstance(conversation, list):
                turns = conversation
            else:
                continue
            conv_df = preprocess_model_conversation(turns, conv_id, model_name)
            if len(conv_df) > 0:
                conversations.append(conv_df)
    elif isinstance(data, list):
        # Handle list format
        for i, conversation in enumerate(data):
            conv_id = i
            if isinstance(conversation, dict) and 'conversation' in conversation:
                turns = conversation['conversation']
            elif isinstance(conversation, list):
                turns = conversation
            else:
                continue
            conv_df = preprocess_model_conversation(turns, conv_id, model_name)
            if len(conv_df) > 0:
                conversations.append(conv_df)

    print(f"  Preprocessed {len(conversations)} conversations")

    if len(conversations) == 0:
        print("  ✗ No valid conversations found!")
        return False

    # Calculate LE values
    print(f"\nCalculating LE values (k={k})...")
    le_values = calculate_le_values(
        conversations,
        word2vec,
        id_column=f'{model_name}-id',
        k=k,
        desc=f"  Computing {model_name} LE"
    )

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as CSV with single column
    result_df = pd.DataFrame({f"LE Values - {model_name}": le_values})
    result_df.to_csv(output_path, index=False)

    print(f"\n  ✓ Saved {len(le_values)} LE values to: {output_path}")
    print(f"  Mean LE: {le_values.mean():.4f}")
    print(f"  Std LE: {le_values.std():.4f}")

    return True


def annotate_kodis_entrainment(
    input_path: str,
    output_path: str,
    k: int = 3
):
    """Calculate LE values for KODIS conversations.

    Args:
        input_path: Path to KODIS conversation JSON file
        output_path: Path to save LE values CSV
        k: Number of following utterances to consider
    """
    print("\nProcessing KODIS conversations...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    # Load word2vec model
    word2vec = load_word2vec_model()
    if word2vec is None:
        return False

    # Load conversations
    print(f"\nLoading data from: {input_path}")
    with open(input_path, 'r') as f:
        kodis_data = json.load(f)

    # Preprocess conversations
    print("Preprocessing conversations...")
    conversations = []

    for kodis_filename, conversation in kodis_data.items():
        conv_df = preprocess_kodis_conversation_from_json(conversation, kodis_filename)
        if len(conv_df) > 0:
            conversations.append(conv_df)

    print(f"  Preprocessed {len(conversations)} conversations")

    if len(conversations) == 0:
        print("  ✗ No valid conversations found!")
        return False

    # Calculate LE values
    print(f"\nCalculating LE values (k={k})...")
    le_values = calculate_le_values(
        conversations,
        word2vec,
        id_column='kodis-id',
        k=k,
        desc="  Computing KODIS LE"
    )

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as CSV with single column
    result_df = pd.DataFrame({f"LE Values - KODIS": le_values})
    result_df.to_csv(output_path, index=False)

    print(f"\n  ✓ Saved {len(le_values)} LE values to: {output_path}")
    print(f"  Mean LE: {le_values.mean():.4f}")
    print(f"  Std LE: {le_values.std():.4f}")

    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Calculate Linguistic Entrainment values using nCLiD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate LE for model conversation
  python scripts/calculate_entrainment.py --model gpt-4.1-mini --input data/complete/gpt-4.1-mini_complete.json

  # Calculate LE for KODIS dataset
  python scripts/calculate_entrainment.py --input data/KODIS/KODIS_merged_20_samples_emo_irp.json --data-type kodis

  # Specify output path
  python scripts/calculate_entrainment.py --model gpt-4.1-mini --input data/complete/gpt-4.1-mini_complete.json --output data/LE/LE_values_gpt-4.1-mini.csv

  # Use custom k value
  python scripts/calculate_entrainment.py --model gpt-4.1-mini --input data/complete/gpt-4.1-mini_complete.json --k 5
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file path"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: data/linguistic_entrainment/LE_values_{model}.csv)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., gpt-4.1-mini). Required for model data type."
    )

    parser.add_argument(
        "--data-type",
        type=str,
        choices=["model", "kodis"],
        default="model",
        help="Type of input data: 'model' or 'kodis' (default: model)"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of following utterances to consider (default: 3)"
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("Linguistic Entrainment Annotation")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Data type: {args.data_type}")
    print(f"k value: {args.k}")
    print("=" * 60)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.data_type == "kodis":
            output_path = "data/linguistic_entrainment/LE_values_KODIS.csv"
        else:
            if not args.model:
                print("Error: --model is required for model data type")
                return 1
            output_path = f"data/linguistic_entrainment/LE_values_{args.model}.csv"

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Process based on data type
    if args.data_type == "kodis":
        success = annotate_kodis_entrainment(
            args.input,
            output_path,
            k=args.k
        )
    else:
        if not args.model:
            print("Error: --model is required for model data type")
            return 1
        success = annotate_model_entrainment(
            args.input,
            output_path,
            args.model,
            k=args.k
        )

    if success:
        print("\n" + "=" * 60)
        print("Linguistic Entrainment Annotation Complete!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("Annotation failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
