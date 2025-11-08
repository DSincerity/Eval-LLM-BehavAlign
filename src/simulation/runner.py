"""Simulation runner for L2L negotiations."""
import os
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

from .agent import DialogAgent
from .personality import select_adjectives, split_100_with_constraints, prompt_build_v2
from .scoring import calculate_final_score
from ..utils import Logger, is_terminated, parse_submission, load_json, compute_time


def run_single_negotiation(
    agent_1: DialogAgent,
    agent_2: DialogAgent,
    n_round: int,
    p1_importance: List[int],
    p2_importance: List[int],
    personality: tuple,
    issue_importance: tuple,
    logger: Logger,
    moderator: Optional[DialogAgent] = None
) -> Dict[str, Any]:
    """Run a single L2L negotiation simulation.

    Args:
        agent_1: First agent (Seller)
        agent_2: Second agent (Buyer)
        n_round: Maximum number of negotiation rounds
        p1_importance: Agent 1's issue importance weights
        p2_importance: Agent 2's issue importance weights
        personality: Tuple of personality traits for both agents
        issue_importance: Tuple of issue importance for both agents
        logger: Logger instance for recording conversation
        moderator: Optional moderator agent

    Returns:
        Dictionary containing negotiation outcome and metadata
    """
    conversation = []

    logger.write(f'---- AGENT 1: Personality: {agent_1.agent_type} ----\n')
    logger.write(f'---- AGENT 1 PROMPT:----\n{agent_1.last_response}\n\n')
    logger.write(f'---- AGENT 2: Personality: {agent_2.agent_type} ----\n')
    logger.write(f'---- AGENT 2 PROMPT:---- \n{agent_2.last_response}\n\n')
    logger.write('\n\n---- START CONVERSATION ----\n\n')

    # Agent2 (Buyer) starts conversation
    agent_2_run = agent_2.call("YOU: ")
    logger.write(f'\t[Agent2] : {agent_2.last_response}')
    conversation.append({"role": "Agent2", "content": agent_2.last_response})

    agent1_score, agent2_score = 0, 0
    deal = "None"
    threshold_max_round = int(0.75 * n_round)
    threshold_max_round2 = int(0.90 * n_round)

    for round_idx in range(n_round):
        # Warning message at 75% of max rounds
        if round_idx - 1 == threshold_max_round:
            print(">> Warning: reaching 75% of maximum number of rounds.")
            warning_msg = (
                "[System Warning] Time is running out. To reach a deal, you must now adjust your stance "
                "and make real compromises. Accept a fair offer, or propose a counteroffer that includes "
                "at least one clear concession. Without meaningful trade-offs from both sides, the "
                "negotiation will fail—and both will receive a zero score."
            )
            agent_1.dialog_history.append({"role": "user", "content": warning_msg})
            agent_2.dialog_history.append({"role": "user", "content": warning_msg})

        # Final warning at 90% of max rounds
        if round_idx - 1 == threshold_max_round2:
            print(">> Final Warning: reaching 90% of maximum number of rounds.")
            warning_msg = (
                "[System Warning – Final Chance] You've reached 90% of the maximum turns. This is your "
                "last chance. To avoid a zero score, you must now accept a fair offer or propose a serious "
                "counteroffer with a clear concession. No deal means zero points for both. Act now."
            )
            agent_1.dialog_history.append({"role": "user", "content": warning_msg})
            agent_2.dialog_history.append({"role": "user", "content": warning_msg})

        # Agent 1's turn
        agent_1_run = agent_1.call(f"THEM: {agent_2_run}\nYOU: ")
        logger.write(f'\t[Agent1] : {agent_1.last_response}')
        conversation.append({"role": "Agent1", "content": agent_1.last_response})

        # Check if agent_1 terminates
        terminated, case = is_terminated(agent_1.last_response)
        if terminated:
            if case == "ACCEPT-DEAL":
                parsed_output = parse_submission(agent_2.last_response)
                assert parsed_output is not None, f"Submission parsing error: {agent_2.last_response}"
                agent1_score, agent2_score = calculate_final_score(
                    parsed_output, p1_importance, p2_importance
                )
                deal = parsed_output
            # else: case == "WALK-AWAY"

            output = {
                "terminated": True,
                "case": f"Agent1-{case}",
                "final_deal": deal,
                "agent1_final_score": agent1_score,
                "agent2_final_score": agent2_score,
                "rounds": round_idx + 1,
                "turns": 2 * (round_idx + 1),
                "personality": personality,
                "issue_importance": issue_importance,
                "conversation": conversation
            }
            return output

        # Agent 2's turn
        agent_2_run = agent_2.call(f"THEM: {agent_1_run}\nYOU: ")
        logger.write(f'\t[Agent2] : {agent_2.last_response}')
        conversation.append({"role": "Agent2", "content": agent_2.last_response})

        # Check if agent_2 terminates
        terminated, case = is_terminated(agent_2.last_response)
        if terminated:
            if case == "ACCEPT-DEAL":
                parsed_output = parse_submission(agent_1.last_response)
                assert parsed_output is not None, f"Submission parsing error: {agent_1.last_response}"
                deal = parsed_output
                agent1_score, agent2_score = calculate_final_score(
                    deal, p1_importance, p2_importance
                )
            # else: case == "WALK-AWAY"

            output = {
                "terminated": True,
                "case": f"Agent2-{case}",
                "final_deal": deal,
                "agent1_final_score": agent1_score,
                "agent2_final_score": agent2_score,
                "rounds": round_idx + 1,
                "turns": 2 * (round_idx + 1),
                "personality": personality,
                "issue_importance": issue_importance,
                "conversation": conversation
            }
            return output

        # Check if max rounds reached
        if round_idx == n_round - 1:
            output = {
                "terminated": True,
                "case": "Max turn reached",
                "final_deal": deal,
                "agent1_final_score": agent1_score,
                "agent2_final_score": agent2_score,
                "rounds": round_idx + 1,
                "turns": 2 * (round_idx + 1),
                "personality": personality,
                "issue_importance": issue_importance,
                "conversation": conversation
            }
            return output


def run_experiment(
    agent_1: DialogAgent,
    agent_2: DialogAgent,
    n_exp: int,
    n_round: int,
    prompt_dir: str,
    personality_adj_dict: Dict,
    logger: Logger,
    personality_setting: bool = True,
    version: str = "test",
    moderator: Optional[DialogAgent] = None
) -> List[Dict[str, Any]]:
    """Run multiple L2L negotiation experiments.

    Args:
        agent_1: First agent
        agent_2: Second agent
        n_exp: Number of experiments to run
        n_round: Number of rounds per experiment
        prompt_dir: Directory containing prompt templates
        personality_adj_dict: Dictionary of personality adjectives
        logger: Logger instance
        personality_setting: Whether to enable personality variation
        version: Version identifier for logging
        moderator: Optional moderator agent

    Returns:
        List of output dictionaries from all experiments
    """
    print(f">> Agent Engine: Agent1 ({agent_1.engine}) vs Agent2 ({agent_2.engine})")
    if personality_setting:
        logger.write(">> Personality setting is ON!!")

    start_time = time.time()
    outputs = []

    for i in range(n_exp):
        conversation = []

        if personality_setting:
            logger.write("")
            # Select personality traits
            player1_selected_traits, personality_sent1 = select_adjectives(personality_adj_dict, 3)
            player2_selected_traits, personality_sent2 = select_adjectives(personality_adj_dict, 3)
            player1_personality = personality_sent1.replace("You are ", "")
            player2_personality = personality_sent2.replace("You are ", "")
        else:
            player1_personality = player2_personality = ""
            personality_sent1 = personality_sent2 = ""

        print(f"Player 1's personality: {personality_sent1}")
        print(f"Player 2's personality: {personality_sent2}")

        # Generate issue importance weights
        p1_ref, p1_self_rev, p1_part_rev, p1_apo = split_100_with_constraints(
            option=list(player1_selected_traits['AGR'].keys())[0].lower(),
            target_position=4
        )
        p2_ref, p2_self_rev, p2_part_rev, p2_apo = split_100_with_constraints(
            option=list(player2_selected_traits['AGR'].keys())[0].lower(),
            target_position=4
        )

        print(f"Player 1's importance: {p1_ref}, {p1_self_rev}, {p1_part_rev}, {p1_apo}")
        print(f"Player 2's importance: {p2_ref}, {p2_self_rev}, {p2_part_rev}, {p2_apo}")

        p1_importance = [p1_ref, p1_self_rev, p1_part_rev, p1_apo]
        p2_importance = [p2_ref, p2_self_rev, p2_part_rev, p2_apo]

        # Build prompts
        agent_1_initial_system_instruction = prompt_build_v2(
            "seller", personality_sent1, p1_importance, prompt_dir, verbose=False
        )
        agent_2_initial_system_instruction = prompt_build_v2(
            "buyer", personality_sent2, p2_importance, prompt_dir, verbose=False
        )

        agent_1.agent_type = player1_personality
        agent_2.agent_type = player2_personality

        logger.write(f"==== ver {version} CASE {i} / {n_exp}, {compute_time(start_time):.2f} min ====")
        logger.write(f'> Agent1 ({agent_1.agent_type})\n> Agent2 ({agent_2.agent_type})\n')

        # Update system instructions
        agent_1.system_instruction = agent_1_initial_system_instruction
        agent_2.system_instruction = agent_2_initial_system_instruction
        agent_1.dialog_history = [{"role": "system", "content": agent_1.system_instruction}]
        agent_2.dialog_history = [{"role": "system", "content": agent_2.system_instruction}]

        personality = (player1_selected_traits, player2_selected_traits)
        issue_importance = (p1_importance, p2_importance)

        try:
            _output = run_single_negotiation(
                agent_1, agent_2, n_round,
                p1_importance, p2_importance,
                personality, issue_importance,
                logger, moderator
            )
        except Exception as e:
            logger.write(f"Error: {e}")
            _output = {
                "terminated": False,
                "case": f"error ({e})",
                "final_deal": {},
                "agent1_final_score": 0,
                "agent2_final_score": 0,
                "rounds": 0,
                "turns": 0,
                "personality": personality,
                "issue_importance": issue_importance,
                "conversation": conversation
            }

        outputs.append(_output)
        logger.write("\n\n")

    return outputs


def summarize_results(outputs: List[Dict[str, Any]], agent_1_engine: str, agent_2_engine: str) -> Dict:
    """Summarize experiment results.

    Args:
        outputs: List of experiment outputs
        agent_1_engine: Name of agent 1's engine
        agent_2_engine: Name of agent 2's engine

    Returns:
        Dictionary with aggregated statistics
    """
    collected_data = defaultdict(list)
    collected_data['agent1_engine'] = agent_1_engine
    collected_data['agent2_engine'] = agent_2_engine

    for info in outputs:
        for key, value in info.items():
            collected_data[key].append(value)

    collected_data['terminated_rate'] = sum(collected_data['terminated']) / len(collected_data['terminated'])
    collected_data['avg_rounds'] = sum(collected_data['rounds']) / len(collected_data['rounds'])
    collected_data['avg_turns'] = sum(collected_data['turns']) / len(collected_data['turns'])
    collected_data['case_dist'] = dict(Counter(collected_data['case']))

    return collected_data
