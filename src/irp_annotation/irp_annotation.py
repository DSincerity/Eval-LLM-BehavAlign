"""IRP (Interest, Rights, Power) annotation module for negotiation dialogues.

This module provides functionality to annotate negotiation conversations with
IRP strategies using GPT-4 with majority voting for quality assurance.

Based on: Brett, J. M., Shapiro, D. L., & Lytle, A. L. (1998).
Breaking the bonds of reciprocity in negotiations.
Academy of Management Journal, 41(4), 410-424.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import openai
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed, retry_if_exception_type


# IRP Strategy Definitions
IRP_STRATEGY_PROMPT = """
[Cooperative Strategies]
- INTEREST: Reference to the wants, needs, or concerns of one or both parties. This may include questions about why the negotiator wants or feels the way they do. This does not include anything about wanting a deal (apology, refund, removing negative review) without a reason.
Example) "I understand that you've been really busy lately"
Non-example) "How should I do this?" or "I do not understand"
Example) "I'm sorry you feel [x]" or "I apologize for [the thing that you think this is]"
Non-example) "Why is that?"

- POSITIVE EXPECTATIONS: Communicating similarities and common goals between the speakers. This does not include positive feeling-related expressions, there must be an explicit expression of similarity or a common goal between the speakers.
Example) "I know you're an excellent employee and I want to make sure you get a promotion."
Non-example) "Great!", "Glad to hear", "I'm very happy to hear that", or "I really appreciate it"
Example) "This is [good] for everyone!"
Non-example) "I promise", "Yes, will do", or "I will make sure to do [something positive towards an agreement]"
Non-example) "I look forward to seeing you again" or "I hope you have a great day"

- PROPOSAL: Proposing concrete recommendations that may help resolve the conflict. This includes anything related to wanting a deal (apology, refund, removing negative review). Must be looked at in context of that speaker's history and if the speaker is the first to propose this. Must also be directly related to the conflict
Example) "Why don't we record your progress weekly instead of monthly, so we can stay on track?"

- CONCESSION: Changing an initial view or position (in response to a proposal) to resolve a conflict. This is dependent on context and must be in response to a deal proposal (apology, refund, removing negative review) and a change in initial position of the current speaker. Must be related to the current conflict.
Example) "That makes sense—I'll try recording my weekly progress instead of doing it monthly."
Non-example) Any voluntary apology
Example) (Previous utterance) "No I will not give a refund" (now current utterance) "Ok, I can give a partial refund"
Non-example) (Previous utterance by other speaker after conflict has been resolved) "Please mail the package" (now current utterance) "Sure, I will do so"
Example: ""I will do that if you do this"

[Neutral Strategies]
- FACTS: Providing information on the situation or history of the dispute, including requests for information, clarification, or summaries.
Example) "Unfortunately, I haven't been able to keep track of your progress over the last several weeks."
- PROCEDURAL: Introductory messages, including discussion about discussion topics, procedures, etc.
Example) "Hi! How are you? Do you have time today to talk about a promotion?"

[Competitive Strategies]
- POWER: Using threats and coercion to try to force the conversation into a resolution. Utterances in this category must have a threat/coercion or accusatory aspect.
Example) "I'm going to tell everyone you've been missing deadlines."
Non-example) "No", "Absolutely no", "Definitely not, I will not do that",  or "This is final"
Example) "I will not take down my negative review unless you do this"
- RIGHTS: Appealing to fixed norms, fairness, or standards to guide a resolution.
Example) "Sorry, I can't do anything—company policy doesn't allow that."
Example) "That's not fair for us"

[Other Strategies]
- RESIDUAL: Any statements that do not fit into the above categories. These could be statements that are short expressions of disagreements or agreements, thanks, or simple apologies. Can include positive expressions that are not expressing expectations or common goals.
Example) "No, I will not do that."
Example) "Sorry."
Example) "I hope you understand."
Example) "You're understanding is appreciated."
Example) "Great!",
Example) "Glad to hear",
Example) "Glad we agree"
Example) "I appreciate it"
"""

# Split instruction
SPLIT_INSTRUCTION = """
######## Instructions ########
- You need to annotate the following conversation at the utterance level, identifying which strategy from the IRP framework aligns with each sentence.
1. First, split each utterance into subject-verb sequences, ensuring that the split sequences should not lose the original meaning of the utterance. For example, "I will give you a refund, if you return the item" should be one sequence, not two.
2. Then, perform IRP annotation for each subject-verb sequence.
Example: "Speaker1: I propose that we make a compromise. Otherwise, I will fire you. Thank you." would be annotated as "Speaker1: I propose that we make a compromise [PROPOSAL]. Otherwise, I will fire you [POWER]. Thank you [RESIDUAL]."
- For utterances that contains "{special_keywords}", annotate with NULL strategy.
- The final format should store the annotated utterances in JSON format with a list of dictionaries as value. Also, strategy label must be one of [INTEREST, POSITIVE EXPECTATIONS, PROPOSAL, CONCESSION, FACTS, PROCEDURAL, POWER, RIGHTS, RESIDUAL].
ex. {{"annotations": [{{"speaker": 'Speaker1', "sentence": "I propose that we make a compromise", "strategy": "PROPOSAL"}}, {{"speaker": 'Speaker1', "sentence": "Otherwise, I will fire you", "strategy": "POWER"}}, {{"speaker": 'Speaker1', "sentence": "Thank you", "strategy": "RESIDUAL"}}, ...] }}
"""

# Annotation instruction
ANNOTATION_INSTRUCTION = """
######## Instructions ########
- You need to annotate the following conversation at the utterance level, identifying which strategy from the IRP framework aligns with each sentence. Make sure to take into account the context and previous utterances in the conversation.
- Sometimes the utterances are incomplete sentences, so if a sentence is incomplete, label both parts of the sentence with the same label:
Example: "When I recieve a refund" —> PROPOSAL, "I will give you the apology" —> PROPOSAL
- Perform IRP annotation for each line in the conversation below.
Example: "Speaker1: I propose that we make a compromise. Otherwise, I will fire you. Thank you." would be annotated as "Speaker1: I propose that we make a compromise [PROPOSAL]. Otherwise, I will fire you [POWER]. Thank you [RESIDUAL]."
- For utterances that contains "{special_keywords}", annotate with NULL strategy.
- The final format should store the annotated utterances in JSON format with a list of dictionaries as value. Also, strategy label must be one of [INTEREST, POSITIVE EXPECTATIONS, PROPOSAL, CONCESSION, FACTS, PROCEDURAL, POWER, RIGHTS, RESIDUAL].
ex. {{"annotations": [{{"speaker": 'Speaker1', "sentence": "I propose that we make a compromise", "strategy": "PROPOSAL"}}, {{"speaker": 'Speaker1', "sentence": "Otherwise, I will fire you", "strategy": "POWER"}}, {{"speaker": 'Speaker1', "sentence": "Thank you", "strategy": "RESIDUAL"}}, ...] }}
"""


@retry(
    stop=stop_after_attempt(4),
    wait=wait_chain(*[wait_fixed(3) for i in range(2)] + [wait_fixed(5) for i in range(1)]),
    retry=retry_if_exception_type((json.JSONDecodeError, openai.OpenAIError))
)
def completion_with_backoff(**kwargs):
    """OpenAI API wrapper with retry logic."""
    try:
        response = openai.ChatCompletion.create(**kwargs)
        return response
    except openai.OpenAIError as e:
        print(f"APIError: {e}")
        raise e


def split_special_keywords(input_string: str, data_type: str = 'model') -> Tuple[str, Optional[str], str]:
    """Split input string by special keywords (SUBMISSION, REJECT, ACCEPT-DEAL, etc.).

    Args:
        input_string: Input string to split
        data_type: Type of data - 'model' or 'kodis'

    Returns:
        Tuple of (before, keyword, after)
    """
    if data_type == 'model':
        keywords = ["REJECT", "SUBMISSION: {", "ACCEPT-DEAL", "WALK AWAY", "WALK-AWAY"]
    else:  # kodis
        keywords = ["Submitted agreement", "Reject Deal", "Accept Deal", "I Walk Away"]

    for keyword in keywords:
        if keyword in input_string:
            if keyword == "SUBMISSION: {":
                match = re.search(r'SUBMISSION: \{(.*?)\}', input_string)
                if match:
                    matched_text = match.group(0)
                    parts = re.split(r"SUBMISSION: \{.*?\}", input_string, 1)
                    return parts[0].strip(), matched_text, parts[1].strip() if len(parts) > 1 else ""
            else:
                parts = re.split(re.escape(keyword), input_string, 1)
                return parts[0].strip(), keyword, parts[1].strip() if len(parts) > 1 else ""

    return input_string, None, ""


def preprocess_conversation(conversation: List[Dict], data_type: str = 'model') -> str:
    """Preprocess conversation for IRP annotation.

    Args:
        conversation: List of conversation turns
        data_type: Type of data - 'model' or 'kodis'

    Returns:
        Formatted conversation string
    """
    formatted_lines = []

    for turn in conversation:
        if data_type == 'model':
            role = turn.get('role', 'Agent1')
            content = turn.get('content', '')
            speaker = "Speaker2" if role == "Agent2" else "Speaker1"
        else:  # kodis
            speaker = turn.get('speaker', 'Speaker1')
            content = turn.get('sentence', '')

        # Split by special keywords
        before, keyword, after = split_special_keywords(content, data_type)

        if before and len(before) > 0:
            formatted_lines.append(f"{speaker}: {before}")
        if keyword and len(keyword) > 0:
            formatted_lines.append(f"{speaker}: {keyword}")
        if after and len(after) > 0:
            formatted_lines.append(f"{speaker}: {after}")

    return "\n".join(formatted_lines)


def annotate_conversation_with_irp(
    conversation: List[Dict],
    data_type: str = 'model',
    model: str = 'gpt-4o',
    majority_voting_max: int = 3,
    verbose: bool = False
) -> List[Dict]:
    """Annotate a single conversation with IRP strategies using majority voting.

    Args:
        conversation: List of conversation turns
        data_type: Type of data - 'model' or 'kodis'
        model: OpenAI model to use
        majority_voting_max: Number of annotations for majority voting
        verbose: Print detailed progress

    Returns:
        List of annotated utterances with IRP strategies
    """
    # Preprocess conversation
    full_chat = preprocess_conversation(conversation, data_type)

    if len(full_chat) == 0:
        return []

    if verbose:
        print(f"\n=== Conversation ===\n{full_chat}\n")

    # Set special keywords based on data type
    special_keywords = "SUBMISSION, REJECT, ACCEPT-DEAL, WALK-AWAY" if data_type == 'model' else "Submitted agreement, Reject Deal, Accept Deal, I Walk Away"

    # Step 1: Get initial split and annotation (3 times)
    split_inst = SPLIT_INSTRUCTION.replace("{special_keywords}", special_keywords)
    prompt1 = IRP_STRATEGY_PROMPT + split_inst + "\n\n## Conversation ##\n" + full_chat

    if verbose:
        print("Step 1: Getting initial sentence splits...")

    response = completion_with_backoff(
        model=model,
        messages=[{"role": "user", "content": prompt1}],
        response_format={"type": "json_object"},
        n=3
    )

    # Parse results
    first_res = json.loads(response['choices'][0]['message']['content']).get('annotations', [])
    sec_res = json.loads(response['choices'][1]['message']['content']).get('annotations', [])
    third_res = json.loads(response['choices'][2]['message']['content']).get('annotations', [])

    major_cnt_sentence = [len(first_res), len(sec_res), len(third_res)]

    # Find most common split length
    most_common_value = Counter(major_cnt_sentence).most_common(1)[0][0]
    cnt_indices = [i for i, value in enumerate(major_cnt_sentence) if value == most_common_value]

    target_res_list = []
    _res_extra = None

    # Handle cases where split lengths differ
    if len(cnt_indices) == 1:
        if verbose:
            print("Split lengths differ, retrying...")
        while True:
            response_extra = completion_with_backoff(
                model=model,
                messages=[{"role": "user", "content": prompt1}],
                response_format={"type": "json_object"}
            )
            _res_extra = json.loads(response_extra['choices'][0]['message']['content']).get('annotations', [])
            cnt = len(_res_extra)

            if cnt == len(first_res) or cnt == len(sec_res) or cnt == len(third_res):
                if verbose:
                    print("Retry successful")
                break
    else:
        for _idx in cnt_indices:
            _add_cnt = [first_res, sec_res, third_res][_idx]
            target_res_list.append(_add_cnt)
            _res_extra = _add_cnt

    # Step 2: Get majority voting for IRP strategies
    target_string = "\n".join([f"{k['speaker']}: {k['sentence']}, strategy: ''" for k in _res_extra])
    annotation_inst = ANNOTATION_INSTRUCTION.replace("{special_keywords}", special_keywords)
    prompt2 = IRP_STRATEGY_PROMPT + annotation_inst + "\n\n## Conversation ##\n" + target_string

    if verbose:
        print(f"Step 2: Getting {majority_voting_max} annotations for majority voting...")

    _n = majority_voting_max - len(target_res_list)
    while True:
        voting_response = completion_with_backoff(
            model=model,
            messages=[{"role": "user", "content": prompt2}],
            response_format={"type": "json_object"},
            n=_n
        )

        for x in voting_response['choices']:
            _result = json.loads(x['message']['content']).get('annotations', [])
            if len(_result) == len(_res_extra):
                target_res_list.append(_result)
                if verbose:
                    print(f"  Added annotation {len(target_res_list)}/{majority_voting_max}")

        _n = majority_voting_max - len(target_res_list)

        if _n <= 0:
            # Perform majority voting
            final = []
            if verbose:
                print("Step 3: Performing majority voting...")

            for annotations in zip(*target_res_list):
                new_dict = {}

                # Normalize sentences
                sentences = [ann['sentence'].lower().strip(" ,. ") for ann in annotations]
                strategies = [ann['strategy'] for ann in annotations]

                # Find most common sentence
                sentence_counter = Counter(sentences)
                most_common_sentence = sentence_counter.most_common(1)[0][0]

                # Get indices of most common sentence
                indices = [i for i, s in enumerate(sentences) if s == most_common_sentence]

                if len(indices) == 1:
                    _n = 1
                    if verbose:
                        print("  Warning: Only one matching sentence found, retrying...")
                    continue

                # Use first matching annotation as reference
                ref = annotations[indices[0]]
                new_dict['speaker'] = ref['speaker']
                new_dict['sentence'] = ref['sentence']

                # Find most common strategy among matching sentences
                filtered_strategies = [strategies[idx] for idx in indices]
                strategy_counter = Counter(filtered_strategies)
                most_common_strategy = strategy_counter.most_common(1)[0][0]
                new_dict['strategy'] = most_common_strategy

                final.append(new_dict)

            return final
        else:
            continue


def annotate_model_conversations(
    input_path: str,
    output_dir: str,
    model: str = 'gpt-4o',
    majority_voting_max: int = 3,
    verbose: bool = False
) -> str:
    """Annotate model conversations with IRP strategies.

    Args:
        input_path: Path to input JSON file (model-merged_250_emo.json)
        output_dir: Directory to save individual annotations
        model: OpenAI model to use
        majority_voting_max: Number of annotations for majority voting
        verbose: Print detailed progress

    Returns:
        Path to output directory with individual annotation files
    """
    # Load input data
    with open(input_path, 'r') as f:
        data = json.load(f)

    conversations = data.get('conversation', [])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Annotating {len(conversations)} conversations...")

    for i, conversation in enumerate(tqdm(conversations, desc="IRP Annotation")):
        output_file = os.path.join(output_dir, f"irp_conv_{i}.json")

        # Skip if already exists
        if os.path.exists(output_file):
            if verbose:
                print(f"  Conversation {i} already annotated, skipping")
            continue

        try:
            annotations = annotate_conversation_with_irp(
                conversation,
                data_type='model',
                model=model,
                majority_voting_max=majority_voting_max,
                verbose=verbose
            )

            # Save individual annotation
            with open(output_file, 'w') as f:
                json.dump(annotations, f, indent=2)

            if verbose:
                print(f"  ✓ Saved: {output_file}")

        except Exception as e:
            print(f"  ✗ Error annotating conversation {i}: {e}")
            continue

    print(f"✓ Individual annotations saved to: {output_dir}")
    return output_dir


def annotate_kodis_conversations(
    input_path: str,
    output_dir: str,
    model: str = 'gpt-4o',
    majority_voting_max: int = 5,
    verbose: bool = False
) -> str:
    """Annotate KODIS conversations with IRP strategies.

    Args:
        input_path: Path to input JSON file (KODIS_combined_dialogues_emo.json)
        output_dir: Directory to save individual annotations
        model: OpenAI model to use
        majority_voting_max: Number of annotations for majority voting (default: 5 for KODIS)
        verbose: Print detailed progress

    Returns:
        Path to output directory with individual annotation files
    """
    # Load input data
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Annotating {len(data)} KODIS conversations...")

    for session_id, conversation in tqdm(data.items(), desc="IRP Annotation"):
        output_file = os.path.join(output_dir, f"{session_id}")

        # Skip if already exists
        if os.path.exists(output_file):
            if verbose:
                print(f"  {session_id} already annotated, skipping")
            continue

        try:
            annotations = annotate_conversation_with_irp(
                conversation,
                data_type='kodis',
                model=model,
                majority_voting_max=majority_voting_max,
                verbose=verbose
            )

            # Save individual annotation
            with open(output_file, 'w') as f:
                json.dump(annotations, f, indent=2)

            if verbose:
                print(f"  ✓ Saved: {output_file}")

        except Exception as e:
            print(f"  ✗ Error annotating {session_id}: {e}")
            continue

    print(f"✓ Individual annotations saved to: {output_dir}")
    return output_dir
