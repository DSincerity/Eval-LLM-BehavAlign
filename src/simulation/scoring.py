"""Scoring utilities for negotiation outcomes."""
from typing import List, Dict


def calculate_final_score(
    final_deal: Dict[str, str],
    agent1_importance: List[int],
    agent2_importance: List[int],
    importance_normalize: bool = False
) -> tuple[float, float]:
    """Calculate final scores for both agents based on deal outcome.

    Args:
        final_deal: Dictionary with keys 'REF', 'SNR', 'BNR', 'SAP', 'BAP'
        agent1_importance: List of 4 importance weights for Agent1 (Seller)
        agent2_importance: List of 4 importance weights for Agent2 (Buyer)
        importance_normalize: Whether to normalize importance values to 0-1 range

    Returns:
        Tuple of (agent1_score, agent2_score)

    Raises:
        AssertionError: If importance weights or deal keys are invalid
        ValueError: If deal values are not in expected format
    """
    assert len(agent1_importance) == len(agent2_importance) == 4, \
        "Agent1 & Agent2 importance should have 4 values"
    assert final_deal.keys() == {'REF', 'SNR', 'BNR', 'SAP', 'BAP'}, \
        "final_deal should have 5 keys: REF, SNR, BNR, SAP, BAP"

    S_importance = agent1_importance  # Agent1: Seller
    B_importance = agent2_importance  # Agent2: Buyer

    if importance_normalize:
        S_importance = [i / 100 for i in S_importance]
        B_importance = [i / 100 for i in B_importance]

    S_final_score = 0
    B_final_score = 0

    # Process REF (Refund)
    ref_value = final_deal['REF']
    if isinstance(ref_value, list):
        ref_value = ref_value[0]

    if ref_value == 'None':
        S_final_score += S_importance[0] * 1  # Seller gets full score
    elif ref_value == 'partial':
        S_final_score += S_importance[0] * 0.5
        B_final_score += B_importance[0] * 0.5
    elif ref_value == 'full':
        B_final_score += B_importance[0] * 1  # Buyer gets full score
    else:
        raise ValueError("REF value should be 'None', 'partial', or 'full'")

    # Process SNR (Seller Negative Review)
    snr_value = final_deal['SNR']
    if isinstance(snr_value, list):
        snr_value = snr_value[0]

    if snr_value == 'remove':
        B_final_score += B_importance[2] * 1  # Buyer benefits from removing seller's negative review
    elif snr_value == 'not remove':
        S_final_score += S_importance[1] * 1  # Seller keeps their review
    else:
        raise ValueError("SNR value should be 'remove' or 'not remove'")

    # Process BNR (Buyer Negative Review)
    bnr_value = final_deal['BNR']
    if isinstance(bnr_value, list):
        bnr_value = bnr_value[0]

    if bnr_value == 'remove':
        S_final_score += S_importance[2] * 1  # Seller benefits from removing buyer's negative review
    elif bnr_value == 'not remove':
        B_final_score += B_importance[1] * 1  # Buyer keeps their review
    else:
        raise ValueError("BNR value should be 'remove' or 'not remove'")

    # Process SAP (Seller Apology)
    sap_value = final_deal['SAP']
    if isinstance(sap_value, list):
        sap_value = sap_value[0]

    if sap_value == 'apologize':
        B_final_score += B_importance[3] * 1  # Buyer receives apology from seller
    elif sap_value == 'not apologize':
        pass
    else:
        raise ValueError("SAP value should be 'apologize' or 'not apologize'")

    # Process BAP (Buyer Apology)
    bap_value = final_deal['BAP']
    if isinstance(bap_value, list):
        bap_value = bap_value[0]

    if bap_value == 'apologize':
        S_final_score += S_importance[3] * 1  # Seller receives apology from buyer
    elif bap_value == 'not apologize':
        pass
    else:
        raise ValueError("BAP value should be 'apologize' or 'not apologize'")

    return S_final_score, B_final_score
