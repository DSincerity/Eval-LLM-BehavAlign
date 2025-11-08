"""Personality trait selection and prompt building utilities."""
import os
import random
from typing import List, Tuple, Dict
from itertools import chain
from textwrap import dedent

from ..utils.file_utils import load_txt_file


# Personality priors based on empirical data
PERSONALITY_POLARITY_DEGREE_PRIORS = {
    "EXT": [0.35, 0.35, 0.52, 0.52, 0.13, 0.13],
    "AGR": [0.09, 0.09, 0.38, 0.38, 0.54, 0.54],
    "CON": [0.05, 0.05, 0.39, 0.39, 0.55, 0.55],
    "NEU": [0.31, 0.31, 0.44, 0.44, 0.24, 0.24],
    "OPE": [0.04, 0.04, 0.49, 0.49, 0.47, 0.47]
}

INTENSITY_MAPPING = {
    "high": "very ",
    "medium": "",
    "low": "a bit "
}

POLARITY_DEGREE_PAIRS = [
    "Low-low", "Low-medium", "Low-high",
    "High-low", "High-medium", "High-high"
]


def select_adjectives(
    personality_dict: Dict[str, Dict[str, List[str]]],
    n: int = 3,
    intensity: str = None
) -> Tuple[Dict[str, Dict[str, List[str]]], str]:
    """Select personality adjectives based on trait distributions.

    Args:
        personality_dict: Dictionary mapping traits to polarity-adjective mappings
        n: Number of adjectives to select for each trait
        intensity: Optional intensity level ('high', 'medium', 'low')

    Returns:
        Tuple of (selected_adjectives, personality_sentence)
    """
    if intensity and intensity not in INTENSITY_MAPPING:
        raise ValueError("Intensity must be 'high', 'medium', or 'low'")

    selected_adjectives = {}

    for trait, adjectives in personality_dict.items():
        prior = PERSONALITY_POLARITY_DEGREE_PRIORS[trait]

        # Sample polarity-degree pair
        polarity_degree = random.choices(POLARITY_DEGREE_PAIRS, weights=prior)[0]
        polarity, degree = polarity_degree.split("-")

        # Get prefix based on degree
        prefix = INTENSITY_MAPPING[degree]

        # Select adjectives
        selected_adjectives[trait] = {
            polarity: [
                prefix + adj
                for adj in random.sample(adjectives[polarity], min(n, len(adjectives[polarity])))
            ]
        }

    # Build personality sentence
    combined = [
        adj
        for trait, polarity_dict in selected_adjectives.items()
        for adj in polarity_dict.values()
    ]
    sentence = "You are " + ", ".join(list(chain(*combined)))

    return selected_adjectives, sentence


def split_100_with_constraints(option: str = "low", target_position: int = 4) -> List[int]:
    """Split 100 into 4 values with constraints on one position.

    This function generates issue importance weights where one issue
    (typically apology) is constrained based on personality (AGR trait).

    Args:
        option: 'low' (5-20 range) or 'high' (40-70 range)
        target_position: 1-indexed position for constrained value (1-4)

    Returns:
        List of 4 integers that sum to 100 (multiples of 5)

    Raises:
        ValueError: If option or target_position is invalid
    """
    total_units = 20  # Since all values are multiples of 5

    # Map option to valid unit bounds
    if option == "low":
        valid_bounds = list(range(1, 5))   # 5-20
    elif option == "high":
        valid_bounds = list(range(8, 15))  # 40-70
    else:
        raise ValueError("Invalid option. Use 'low' or 'high'.")

    if not (1 <= target_position <= 4):
        raise ValueError("target_position must be between 1 and 4.")

    target_index = target_position - 1

    while True:
        # Pick constrained value
        constrained_value = random.choice(valid_bounds)

        # Remaining units
        remaining = total_units - constrained_value
        if remaining < 3:
            continue

        # Split the remaining units into 3 parts
        split1 = random.randint(1, remaining - 2)
        split2 = random.randint(1, remaining - split1 - 1)
        part1 = split1
        part2 = split2
        part3 = remaining - part1 - part2

        other_parts = [part1, part2, part3]

        # Construct the full list with constrained value at specified position
        result_units = []
        for i in range(4):
            if i == target_index:
                result_units.append(constrained_value)
            else:
                result_units.append(other_parts.pop())

        # Convert units to multiples of 5
        final_result = [x * 5 for x in result_units]

        return final_result


def prompt_build_v2(
    prompt_type: str,
    personality_sentence: str,
    issue_importance: List[int],
    prompt_dir: str,
    verbose: bool = False
) -> str:
    """Build a prompt for the agent based on personality and issue importance.

    Args:
        prompt_type: The type of prompt to build ('seller' or 'buyer')
        personality_sentence: The personality description of the agent
        issue_importance: A list of 4 issue importance weights
        prompt_dir: The directory containing prompt templates
        verbose: Whether to print the generated prompt

    Returns:
        The constructed prompt for the agent

    Raises:
        AssertionError: If prompt_type is invalid
    """
    assert prompt_type in ["seller", "buyer"], "Invalid prompt type"

    # Load the base prompt template
    base_prompt = load_txt_file(os.path.join(prompt_dir, f"{prompt_type}_prompt.txt"))

    # Prepare values for f-string evaluation
    personality_place_holder = personality_sentence
    issue_weights_refund = str(issue_importance[0])
    issue_weight_self_rev = str(issue_importance[1])
    issue_weight_partner_rev = str(issue_importance[2])
    issue_weight_apology = str(issue_importance[3])

    # Replace placeholders in the base prompt
    prompt = eval(f'f"""{base_prompt}"""')
    prompt = dedent(prompt)

    if verbose:
        print("=" * 50)
        print("Generated Prompt")
        print("=" * 50)
        print(prompt)
        print("=" * 50)

    return prompt
