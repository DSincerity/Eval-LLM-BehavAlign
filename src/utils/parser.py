"""Parsing utilities for negotiation dialogues."""
import re
import json
from typing import Optional, Tuple, List


def is_terminated(utterance: str) -> Tuple[bool, Optional[str]]:
    """Check if utterance indicates negotiation termination.

    Args:
        utterance: Raw LLM output

    Returns:
        Tuple of (is_terminated, termination_case)
        where termination_case is 'ACCEPT-DEAL', 'WALK-AWAY', or None
    """
    utterance = utterance.strip()
    is_accept = re.search(r"\bACCEPT-DEAL", utterance) is not None
    is_walkaway = re.search(r"\bWALK-AWAY", utterance) is not None

    case = None
    if is_accept:
        case = 'ACCEPT-DEAL'
    elif is_walkaway:
        case = 'WALK-AWAY'

    return (is_accept or is_walkaway), case


def parse_submission(utterance: str) -> Optional[dict]:
    """Extract JSON submission from utterance.

    Args:
        utterance: LLM output in the format 'SUBMISSION: {...}'

    Returns:
        Parsed dictionary if valid JSON is found, None otherwise
    """
    match = re.search(r"SUBMISSION:\s*(\{.*\})", utterance.strip())

    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
    return None


def is_resolved(utterance: str) -> bool:
    """Check if utterance ends with 'RESOLVE' tag.

    Args:
        utterance: Raw LLM output

    Returns:
        True if utterance ends with 'RESOLVE' or 'RESOLVED'
    """
    utterance = utterance.strip()
    return re.search(r"\bRESOLVE[D]*", utterance) is not None


def return_non_verbal(utterance: str) -> List[str]:
    """Extract nonverbal actions from utterance.

    Assumes nonverbal actions are in format "(NON VERBAL ACTION)"

    Args:
        utterance: Raw LLM output

    Returns:
        List of nonverbal actions without enclosing parentheses
    """
    non_verbals = []
    for elem in re.findall(r"\([A-Z\s]+\)", utterance):
        non_verbals.append(elem[1:-1])
    return non_verbals


def return_text_only(utterance: str) -> str:
    """Extract text without nonverbal actions and resolution tags.

    Args:
        utterance: Raw LLM output

    Returns:
        Cleaned text string
    """
    utterance = re.sub(r"\([A-Z\s]+\)", "", utterance)
    utterance = re.sub(r"\bRESOLVE[D]*", "", utterance)
    return utterance.strip()


def extract_strategy(sentence: str) -> Optional[str]:
    """Extract strategy value from sentence.

    Args:
        sentence: Sentence containing strategy annotation

    Returns:
        Extracted strategy value or None
    """
    pattern = r'\[[\'\"]Strategy[\'\"]: ([^\]]+)\]'
    match = re.search(pattern, sentence)

    if match:
        value = match.group(1)
        # Remove special characters
        clean_value = re.sub(r'[^\w\s]', '', value)
        return clean_value

    return None
