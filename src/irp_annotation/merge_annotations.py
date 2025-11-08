"""Merge IRP annotation files into final format.

This module merges individual IRP annotation files into the main data file,
creating two formats:
- irp_1: Strategy list per original utterance
- irp_2: Individual annotated sentences with split information
"""

import json
import os
import re
from glob import glob
from typing import Dict, List
from tqdm import tqdm


def extract_number(filename: str) -> int:
    """Extract number from filename for sorting.

    Args:
        filename: Filename to extract number from

    Returns:
        Extracted number or infinity if not found
    """
    match = re.search(r'_(\d+)\.json$', filename)
    return int(match.group(1)) if match else float('inf')


def combine_consecutive_utterances(dialogues: List[Dict]) -> List[Dict]:
    """Combine consecutive utterances from the same speaker.

    Args:
        dialogues: List of dialogue utterances

    Returns:
        Combined dialogue list
    """
    if not dialogues:
        return []

    # Initialize first utterance
    if isinstance(dialogues[0].get('strategy'), str):
        dialogues[0]['strategy'] = [dialogues[0]['strategy']]

    combined = [dialogues[0]]

    for dialogue in dialogues[1:]:
        last = combined[-1]

        if dialogue['speaker'] == last['speaker']:
            # Same speaker - combine sentences
            last['sentence'] += ". " + dialogue['sentence']

            if isinstance(last['strategy'], str):
                last['strategy'] = [last['strategy']]

            last['strategy'] = last['strategy'] + [dialogue['strategy']]
        else:
            # Different speaker - add new entry
            if isinstance(dialogue.get('strategy'), str):
                dialogue['strategy'] = [dialogue['strategy']]

            combined.append(dialogue)

    return combined


def merge_model_annotations(
    input_data_path: str,
    annotation_dir: str,
    output_path: str,
    combine_same_speaker: bool = False
) -> Dict:
    """Merge model IRP annotations into main data file.

    Args:
        input_data_path: Path to input emotion-annotated data ({model}-merged_250_emo.json)
        annotation_dir: Directory containing individual IRP annotation files
        output_path: Path to save merged output ({model}-merged_250_emo_irp.json)
        combine_same_speaker: Whether to combine consecutive utterances from same speaker

    Returns:
        Merged data dictionary
    """
    # Load input data
    with open(input_data_path, 'r') as f:
        data = json.load(f)

    # Find all annotation files
    annotation_files = sorted(
        glob(os.path.join(annotation_dir, "irp_conv_*.json")),
        key=extract_number
    )

    if len(annotation_files) != len(data.get('conversation', [])):
        print(f"Warning: Number of annotation files ({len(annotation_files)}) "
              f"does not match number of conversations ({len(data.get('conversation', []))})")

    irp_1 = []  # Strategy list per original utterance
    irp_2 = []  # Individual annotated sentences

    print("Merging IRP annotations...")

    for i, conversation in enumerate(tqdm(data.get('conversation', []))):
        annotation_file = os.path.join(annotation_dir, f"irp_conv_{i}.json")

        if not os.path.exists(annotation_file):
            print(f"  Warning: Annotation file not found for conversation {i}")
            irp_1.append([])
            irp_2.append([])
            continue

        # Load IRP annotations
        with open(annotation_file, 'r') as f:
            irp_annotation_convo = json.load(f)

        # Optionally combine consecutive utterances from same speaker
        if combine_same_speaker:
            irp_annotation_convo = combine_consecutive_utterances(irp_annotation_convo)

        irp_annotation_idx = 0
        irp_1_convo_element = []
        irp_2_convo_element = []

        for original_idx, turn in enumerate(conversation):
            speaker = "Speaker2" if turn["role"] == "Agent2" else "Speaker1"
            utterance = turn["content"]

            # IRP_1: List of strategies per original utterance
            irp_1_element = {
                "role": turn["role"],
                "content": utterance,
                "strategies": []
            }

            split_idx = 0

            # Match annotations to original utterances
            while (irp_annotation_idx < len(irp_annotation_convo) and
                   irp_annotation_convo[irp_annotation_idx]["speaker"] == speaker):

                irp_1_element["strategies"].append(
                    irp_annotation_convo[irp_annotation_idx]["strategy"]
                )

                # IRP_2: Individual annotated sentences
                irp_2_element = {
                    "role": turn["role"],
                    "content": irp_annotation_convo[irp_annotation_idx]["sentence"],
                    "strategy": irp_annotation_convo[irp_annotation_idx]["strategy"],
                    "utterance_original_idx": original_idx,
                    "utterance_split_idx": split_idx
                }

                irp_2_convo_element.append(irp_2_element)

                irp_annotation_idx += 1
                split_idx += 1

            irp_1_convo_element.append(irp_1_element)

        irp_1.append(irp_1_convo_element)
        irp_2.append(irp_2_convo_element)

    # Add IRP annotations to data
    data["irp_1"] = irp_1
    data["irp_2"] = irp_2

    # Save merged data
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Merged annotations saved to: {output_path}")
    return data


def merge_kodis_annotations(
    input_data_path: str,
    annotation_dir: str,
    output_path: str,
    combine_same_speaker: bool = True
) -> Dict:
    """Merge KODIS IRP annotations into combined format.

    Args:
        input_data_path: Path to input KODIS data (KODIS_combined_dialogues_emo.json)
        annotation_dir: Directory containing individual IRP annotation files
        output_path: Path to save merged output (KODIS_combined_dialogues_emo_irp.json)
        combine_same_speaker: Whether to combine consecutive utterances from same speaker

    Returns:
        Merged data dictionary
    """
    # Load input data
    with open(input_data_path, 'r') as f:
        kodis_data = json.load(f)

    # Find all annotation files
    annotation_files = sorted(
        glob(os.path.join(annotation_dir, "irp_kodis_*.json")),
        key=extract_number
    )

    if len(annotation_files) != len(kodis_data):
        print(f"Warning: Number of annotation files ({len(annotation_files)}) "
              f"does not match number of KODIS conversations ({len(kodis_data)})")

    merged_conversations = {}

    print("Merging KODIS IRP annotations...")

    for session_id, conversation in tqdm(kodis_data.items()):
        annotation_file = os.path.join(annotation_dir, session_id)

        if not os.path.exists(annotation_file):
            print(f"  Warning: Annotation file not found for {session_id}")
            merged_conversations[session_id] = conversation
            continue

        # Load IRP annotations
        with open(annotation_file, 'r') as f:
            irp_annotations = json.load(f)

        # Optionally combine consecutive utterances from same speaker
        if combine_same_speaker:
            irp_annotations = combine_consecutive_utterances(irp_annotations)

        # Merge with original conversation
        merged_conversation = []
        for turn, irp in zip(conversation, irp_annotations):
            merged_turn = turn.copy()
            merged_turn['strategy'] = irp['strategy']
            merged_conversation.append(merged_turn)

        merged_conversations[session_id] = merged_conversation

    # Save merged data
    with open(output_path, 'w') as f:
        json.dump(merged_conversations, f, indent=2)

    print(f"✓ Merged KODIS annotations saved to: {output_path}")
    return merged_conversations
