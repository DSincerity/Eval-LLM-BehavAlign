"""Script to annotate emotions in conversation data using EmoBERTa.

This script provides a convenient interface to annotate emotion labels
for both model-generated conversations and KODIS dataset.
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.emotion_annotation.emotion_classification import (
    load_tokenizer_model,
    annotate_model_conversations,
    annotate_kodis_conversations
)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Annotate emotions in conversation data using EmoBERTa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate model conversation
  python scripts/annotate_emotions.py --input data/gpt-4.1-merged_250.json --data-type model

  # Annotate KODIS dataset
  python scripts/annotate_emotions.py --input data/KODIS-merged_220_IRP.json --data-type kodis

  # Use GPU and verbose output
  python scripts/annotate_emotions.py --input data/claude-merged_250.json --device cuda --verbose

  # Specify output path
  python scripts/annotate_emotions.py --input data/gemini-merged_250.json --output data/gemini_annotated.json
        """
    )

    # Model configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="emoberta-base",
        choices=["emoberta-base", "emoberta-large"],
        help="EmoBERTa model variant to use (default: emoberta-base)"
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., gpt-4.1-mini) - auto-detects input file from data/simulations/"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: data/emotions/<model>_emo.json)"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="model",
        choices=["model", "kodis"],
        help="Type of input data (default: model)"
    )

    # Options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Auto-detect input file if --model is provided
    if args.model and not args.input:
        args.input = f"data/simulations/{args.model}.json"
        if not os.path.exists(args.input):
            print(f"Error: Simulation file not found: {args.input}")
            print(f"Please run simulation first: python scripts/run_simulation.py --agent_1_engine {args.model}")
            sys.exit(1)

    # Require either --input or --model
    if not args.input:
        print("Error: Either --input or --model must be specified")
        parser.print_help()
        sys.exit(1)

    # Auto-set output path if not provided
    if not args.output:
        if args.model:
            os.makedirs("data/emotions", exist_ok=True)
            args.output = f"data/emotions/{args.model}_emo.json"
        else:
            # Default: add _emo suffix to input filename
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_emo{input_path.suffix}")

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("Emotion Annotation using EmoBERTa")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Data type: {args.data_type}")
    print(f"Model: {args.model_type}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    load_tokenizer_model(args.model_type, args.device)

    # Annotate conversations
    print("\nAnnotating emotions...")
    if args.data_type == "model":
        annotate_model_conversations(
            input_path=args.input,
            output_path=args.output,
            verbose=args.verbose
        )
    elif args.data_type == "kodis":
        annotate_kodis_conversations(
            input_path=args.input,
            output_path=args.output,
            verbose=args.verbose
        )

    print("\n" + "=" * 60)
    print("Emotion annotation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
