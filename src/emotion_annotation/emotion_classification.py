"""Emotion annotation using EmoBERTa model.

This module provides emotion classification for conversation turns using the EmoBERTa model.
It supports both model conversations and KODIS dataset annotation.
"""
import argparse
import logging
import os
import json
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------- GLOBAL VARIABLES ---------------------- #
EMOTIONS = [
    "neutral",
    "joy",
    "surprise",
    "anger",
    "sadness",
    "disgust",
    "fear",
]
ID2EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

# Global model state
tokenizer = None
model = None
device = None


def load_json(file_path: str) -> dict:
    """Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json(data: dict, file_path: str) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    logger.info(f"Saved results to {file_path}")


def load_tokenizer_model(model_type: str, device_: str) -> None:
    """Load tokenizer and model.

    Args:
        model_type: Should be either "emoberta-base" or "emoberta-large"
        device_: "cpu" or "cuda"

    Raises:
        ValueError: If model_type is not valid
    """
    # Normalize model type
    if "large" in model_type.lower():
        model_type = "emoberta-large"
    elif "base" in model_type.lower():
        model_type = "emoberta-base"
    else:
        raise ValueError(
            f"{model_type} is not a valid model type! Should be 'base' or 'large'."
        )

    # Check if model is local or use HuggingFace
    if not os.path.isdir(model_type):
        model_type = f"tae898/{model_type}"

    global device, tokenizer, model
    device = device_

    logger.info(f"Loading tokenizer from {model_type}...")
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    logger.info(f"Loading model from {model_type}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_type)
    model.eval()
    model.to(device)

    logger.info(f"Model loaded successfully on {device}")


def inference(text: str) -> Dict[str, float]:
    """Perform emotion classification inference.

    Args:
        text: Input text to classify

    Returns:
        Dictionary mapping emotion labels to probabilities

    Raises:
        AssertionError: If model/tokenizer not loaded or invalid input
    """
    assert tokenizer is not None, "Tokenizer is not loaded. Please load the tokenizer first."
    assert model is not None, "Model is not loaded. Please load the model first."
    assert device is not None, "Device is not set. Please set the device first."
    assert text is not None, "Text is None. Please provide a valid text input."
    assert isinstance(text, str), "Text should be a string."
    assert len(text) > 0, "Text is empty. Please provide a non-empty text input."

    # Tokenize
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs["logits"], dim=1).squeeze().cpu().numpy()

    # Convert to emotion dictionary
    emotion_scores = {ID2EMOTION[idx]: float(prob) for idx, prob in enumerate(probs)}

    return emotion_scores


def annotate_model_conversations(
    input_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """Annotate emotion labels for model-generated conversations.

    Args:
        input_path: Path to input JSON file containing conversations
        output_path: Path to save annotated results (default: auto-generated)
        verbose: Whether to print detailed progress

    Returns:
        Annotated conversation data
    """
    logger.info(f"Loading model conversations from {input_path}...")
    data = load_json(input_path)

    if 'conversation' not in data:
        raise ValueError("Input JSON must have 'conversation' field")

    all_episodes = len(data['conversation'])
    logger.info(f"Annotating {all_episodes} conversations...")

    for sid in tqdm(range(all_episodes), desc="Annotating conversations"):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Episode {sid}")
            print('=' * 60)

        conversation = data['conversation'][sid]

        for turn_idx, turn in enumerate(conversation):
            text = turn.get('content', '')

            if not text or len(text.strip()) == 0:
                logger.warning(f"Empty text at Episode {sid}, Turn {turn_idx}")
                continue

            # Perform emotion classification
            emotion_scores = inference(text)
            turn['emotion'] = emotion_scores

            if verbose:
                print(f"Turn {turn_idx}: {text[:50]}...")
                print(f"Emotions: {emotion_scores}")

    # Save results
    if output_path is None:
        output_path = input_path.replace('.json', '_emo.json')

    save_json(data, output_path)
    logger.info(f"Annotation complete: {output_path}")

    return data


def annotate_kodis_conversations(
    input_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """Annotate emotion labels for KODIS dataset.

    Args:
        input_path: Path to KODIS JSON file
        output_path: Path to save annotated results (default: auto-generated)
        verbose: Whether to print detailed progress

    Returns:
        Annotated KODIS data
    """
    logger.info(f"Loading KODIS data from {input_path}...")
    data = load_json(input_path)

    if not data:
        raise ValueError("KODIS data is empty or not found")

    total_sessions = len(data)
    logger.info(f"Annotating {total_sessions} KODIS sessions...")

    for session_id, session in tqdm(data.items(), desc="Annotating KODIS"):
        if not isinstance(session, list):
            logger.warning(f"Session {session_id} is not a list, skipping...")
            continue

        for turn_idx, turn in enumerate(session):
            text = turn.get('sentence', '')

            if not text or len(text.strip()) == 0:
                logger.warning(f"Empty text at Session {session_id}, Turn {turn_idx}")
                continue

            # Perform emotion classification
            emotion_scores = inference(text)
            turn['emotion'] = emotion_scores

            if verbose and turn_idx < 3:  # Print first 3 turns of each session
                print(f"[{session_id}] Turn {turn_idx}: {text[:50]}...")
                print(f"  Emotions: {emotion_scores}")

    # Save results
    if output_path is None:
        output_path = input_path.replace('.json', '_emo.json')

    save_json(data, output_path)
    logger.info(f"KODIS annotation complete: {output_path}")

    return data


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Annotate emotions in conversation data using EmoBERTa"
    )

    # Model configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (cpu or cuda)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="emoberta-base",
        choices=["emoberta-base", "emoberta-large"],
        help="EmoBERTa model variant to use"
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: input_emo.json)"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="model",
        choices=["model", "kodis"],
        help="Type of input data: 'model' for model conversations, 'kodis' for KODIS dataset"
    )

    # Options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Load model
    load_tokenizer_model(args.model_type, args.device)

    # Annotate conversations
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

    logger.info("Emotion annotation complete!")


if __name__ == "__main__":
    main()
