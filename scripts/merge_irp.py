"""Merge IRP annotation files with original data.

This script merges individual IRP annotation files back into the main data file,
creating two formats:
- irp_1: Strategy list per original utterance
- irp_2: Individual annotated sentences with split information

Usage:
    # Merge model annotations
    python scripts/merge_irp.py \\
        --input data/gpt-4.1-merged_250_emo.json \\
        --annotation-dir data/IRP_Annotation/gpt-4.1_annotations \\
        --output data/gpt-4.1-merged_250_emo_irp.json \\
        --data-type model

    # Merge KODIS annotations
    python scripts/merge_irp.py \\
        --input data/KODIS_combined_dialogues_emo.json \\
        --annotation-dir data/IRP_Annotation/KODIS_annotations \\
        --output data/KODIS_combined_dialogues_emo_irp.json \\
        --data-type kodis
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.irp_annotation import merge_model_annotations, merge_kodis_annotations


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge IRP annotations with original data'
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
        help='Model name (e.g., gpt-4.1-mini) - auto-detects input and annotation files'
    )

    parser.add_argument(
        '--annotation-dir',
        type=str,
        default=None,
        help='Directory containing individual IRP annotation files (default: data/IRP_Annotation/<model>_annotations)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save merged output file (default: data/complete/<model>_complete.json)'
    )

    parser.add_argument(
        '--data-type',
        type=str,
        default='model',
        choices=['model', 'kodis'],
        help='Type of data (model or kodis)'
    )

    parser.add_argument(
        '--combine-same-speaker',
        action='store_true',
        help='Combine consecutive utterances from same speaker'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Auto-detect paths if --model-name is provided
    if args.model_name:
        if not args.input:
            args.input = f"data/emotions/{args.model_name}_emo.json"
        if not args.annotation_dir:
            args.annotation_dir = f"data/IRP_Annotation/{args.model_name}_annotations"
        if not args.output:
            os.makedirs("data/complete", exist_ok=True)
            args.output = f"data/complete/{args.model_name}_complete.json"

    # Require necessary paths
    if not args.input:
        print("Error: Either --input or --model-name must be specified")
        return 1
    if not args.annotation_dir:
        print("Error: Either --annotation-dir or --model-name must be specified")
        return 1
    if not args.output:
        print("Error: Either --output or --model-name must be specified")
        return 1

    # Validate input files
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    if not os.path.exists(args.annotation_dir):
        print(f"Error: Annotation directory not found: {args.annotation_dir}")
        print(f"Please run IRP annotation first: python scripts/annotate_irp.py --model-name {args.model_name if args.model_name else '...'}")
        return 1

    print("=" * 60)
    print("Merging IRP Annotations")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Annotation directory: {args.annotation_dir}")
    print(f"Output file: {args.output}")
    print(f"Data type: {args.data_type}")
    print(f"Combine same speaker: {args.combine_same_speaker}")
    print("=" * 60)

    try:
        if args.data_type == 'model':
            merged_data = merge_model_annotations(
                input_data_path=args.input,
                annotation_dir=args.annotation_dir,
                output_path=args.output,
                combine_same_speaker=args.combine_same_speaker
            )
        else:  # kodis
            merged_data = merge_kodis_annotations(
                input_data_path=args.input,
                annotation_dir=args.annotation_dir,
                output_path=args.output,
                combine_same_speaker=args.combine_same_speaker
            )

        print("\n" + "=" * 60)
        print("Merge Complete!")
        print("=" * 60)
        print(f"Merged data saved to: {args.output}")

        if args.data_type == 'model':
            print("\nOutput format:")
            print("  - irp_1: Strategy list per original utterance")
            print("  - irp_2: Individual annotated sentences with split information")
        else:
            print("\nOutput format:")
            print("  - Combined dialogues with 'strategy' field per utterance")

        return 0

    except Exception as e:
        print(f"\nError during merge: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
