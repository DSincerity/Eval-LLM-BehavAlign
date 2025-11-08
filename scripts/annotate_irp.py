"""IRP annotation script for model conversations and KODIS dataset.

This script annotates conversations with IRP (Interest, Rights, Power) strategies
using GPT-4 with majority voting for quality assurance.

Usage:
    # Annotate model conversations
    python scripts/annotate_irp.py \\
        --input data/gpt-4.1-merged_250_emo.json \\
        --data-type model \\
        --output-dir data/IRP_Annotation/gpt-4.1_annotations

    # Annotate KODIS dataset
    python scripts/annotate_irp.py \\
        --input data/KODIS_combined_dialogues_emo.json \\
        --data-type kodis \\
        --output-dir data/IRP_Annotation/KODIS_annotations
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.irp_annotation import annotate_model_conversations, annotate_kodis_conversations


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Annotate conversations with IRP strategies'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input JSON file (emotion-annotated)'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name (e.g., gpt-4.1-mini) - auto-detects input from data/emotions/'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save individual IRP annotation files (default: data/IRP_Annotation/<model>_annotations)'
    )

    parser.add_argument(
        '--data-type',
        type=str,
        default='model',
        choices=['model', 'kodis'],
        help='Type of data to annotate (model or kodis)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use for annotation (default: gpt-4o)'
    )

    parser.add_argument(
        '--majority-voting',
        type=int,
        default=None,
        help='Number of annotations for majority voting (default: 3 for model, 5 for KODIS)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Auto-detect input file if --model-name is provided
    if args.model_name and not args.input:
        args.input = f"data/emotions/{args.model_name}_emo.json"
        if not os.path.exists(args.input):
            print(f"Error: Emotion-annotated file not found: {args.input}")
            print(f"Please run emotion annotation first: python scripts/annotate_emotions.py --model {args.model_name}")
            return 1

    # Require either --input or --model-name
    if not args.input:
        print("Error: Either --input or --model-name must be specified")
        return 1

    # Auto-set output directory if not provided
    if not args.output_dir:
        if args.model_name:
            args.output_dir = f"data/IRP_Annotation/{args.model_name}_annotations"
        else:
            # Extract model name from input path
            import re
            match = re.search(r'/([\w.-]+)_emo\.json', args.input)
            if match:
                model_name = match.group(1)
                args.output_dir = f"data/IRP_Annotation/{model_name}_annotations"
            else:
                args.output_dir = f"data/IRP_Annotation/annotations"

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Set default majority voting based on data type
    if args.majority_voting is None:
        args.majority_voting = 5 if args.data_type == 'kodis' else 3

    print("=" * 60)
    print("IRP Annotation")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.data_type}")
    print(f"Model: {args.model}")
    print(f"Majority voting: {args.majority_voting}")
    print("=" * 60)

    try:
        if args.data_type == 'model':
            output_dir = annotate_model_conversations(
                input_path=args.input,
                output_dir=args.output_dir,
                model=args.model,
                majority_voting_max=args.majority_voting,
                verbose=args.verbose
            )
        else:  # kodis
            output_dir = annotate_kodis_conversations(
                input_path=args.input,
                output_dir=args.output_dir,
                model=args.model,
                majority_voting_max=args.majority_voting,
                verbose=args.verbose
            )

        print("\n" + "=" * 60)
        print("IRP Annotation Complete!")
        print("=" * 60)
        print(f"Individual annotations saved to: {output_dir}")
        print("\nNext step: Run merge script to combine annotations with original data")

        return 0

    except Exception as e:
        print(f"\nError during IRP annotation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
