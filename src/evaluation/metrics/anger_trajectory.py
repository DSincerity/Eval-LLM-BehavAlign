"""Anger trajectory analysis using DTW distance and AUC metrics."""
import numpy as np
from dtaidistance import dtw
from scipy.stats import ttest_ind
from typing import List, Dict, Tuple, Any
from tqdm import tqdm


ROLE_MAPPING = {
    'Agent2': 'buyer',
    'Agent1': 'seller',
    'Speaker1': 'buyer',
    'Speaker2': 'seller',
}


def get_emotion_trajectory(
    dialog: List[Dict],
    target_role: str,
    target_emotion: str,
    is_kodis: bool = False
) -> np.ndarray:
    """Extract emotion trajectory for a specific role and emotion.

    Args:
        dialog: List of dialogue turns with emotion annotations
        target_role: Target role ('buyer' or 'seller')
        target_emotion: Target emotion (e.g., 'anger')
        is_kodis: Whether the data is from KODIS dataset

    Returns:
        Array of emotion values over time
    """
    emotion_trajectory = []

    for turn in dialog:
        role_key = 'speaker' if is_kodis else 'role'
        role = turn[role_key]
        emo = turn['emotion']

        if ROLE_MAPPING[role] == target_role:
            emotion_trajectory.append(emo[target_emotion])

    return np.array(emotion_trajectory)


def extract_all_trajectories(
    conversations: List[List[Dict]],
    target_emotion: str = 'anger',
    is_kodis: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract emotion trajectories for both buyers and sellers.

    Args:
        conversations: List of conversations
        target_emotion: Target emotion to track
        is_kodis: Whether the data is from KODIS dataset

    Returns:
        Tuple of (buyer_trajectories, seller_trajectories)
    """
    buyer_trajs = []
    seller_trajs = []

    for dialog in conversations:
        buyer_traj = get_emotion_trajectory(dialog, 'buyer', target_emotion, is_kodis)
        seller_traj = get_emotion_trajectory(dialog, 'seller', target_emotion, is_kodis)

        buyer_trajs.append(buyer_traj)
        seller_trajs.append(seller_traj)

    return buyer_trajs, seller_trajs


def calculate_dtw_distances(
    group1: List[np.ndarray],
    group2: List[np.ndarray],
    desc: str = "Calculating DTW"
) -> List[float]:
    """Calculate DTW distances between two groups of trajectories.

    Args:
        group1: First group of trajectories
        group2: Second group of trajectories
        desc: Description for progress bar

    Returns:
        List of DTW distances
    """
    distances = []
    total = len(group1) * len(group2)

    with tqdm(total=total, desc=desc, leave=False) as pbar:
        for traj1 in group1:
            for traj2 in group2:
                dist = dtw.distance(traj1, traj2)
                distances.append(dist)
                pbar.update(1)

    return distances


def calculate_auc(trajectory: np.ndarray, normalize: bool = True) -> float:
    """Calculate Area Under Curve for a trajectory.

    Args:
        trajectory: Emotion trajectory array
        normalize: Whether to normalize by trajectory length

    Returns:
        AUC value
    """
    auc = np.trapz(trajectory)
    if normalize and len(trajectory) > 0:
        auc = auc / len(trajectory)
    return auc


def compute_dtw_metrics(
    human_trajs: List[np.ndarray],
    agent_trajs_dict: Dict[str, List[np.ndarray]]
) -> Dict[str, Any]:
    """Compute DTW distance metrics comparing human and agent trajectories.

    Args:
        human_trajs: Human emotion trajectories
        agent_trajs_dict: Dictionary mapping model names to trajectories

    Returns:
        Dictionary containing DTW metrics and statistical tests
    """
    results = {
        'within_group': {},
        'between_group': {},
        'statistical_tests': {}
    }

    print(f"  Computing Human-Human DTW distances ({len(human_trajs)} trajectories)...")
    # Human within-group distances
    human_within = calculate_dtw_distances(
        human_trajs, human_trajs,
        desc="Human-Human DTW"
    )
    results['within_group']['human'] = {
        'distances': human_within,
        'mean': float(np.mean(human_within)),
        'std': float(np.std(human_within))
    }
    print(f"    Mean: {np.mean(human_within):.4f}")

    # Agent within-group and between-group distances
    for model_name, agent_trajs in agent_trajs_dict.items():
        print(f"\n  Processing {model_name} ({len(agent_trajs)} trajectories)...")

        # Within-group
        print(f"    Computing {model_name}-{model_name} DTW distances...")
        agent_within = calculate_dtw_distances(
            agent_trajs, agent_trajs,
            desc=f"{model_name}-{model_name} DTW"
        )
        results['within_group'][model_name] = {
            'distances': agent_within,
            'mean': float(np.mean(agent_within)),
            'std': float(np.std(agent_within))
        }
        print(f"      Mean: {np.mean(agent_within):.4f}")

        # Between-group (Agent-Human)
        print(f"    Computing {model_name}-Human DTW distances...")
        agent_human = calculate_dtw_distances(
            agent_trajs, human_trajs,
            desc=f"{model_name}-Human DTW"
        )
        results['between_group'][model_name] = {
            'distances': agent_human,
            'mean': float(np.mean(agent_human)),
            'std': float(np.std(agent_human))
        }
        print(f"      Mean: {np.mean(agent_human):.4f}")

        # ATS (Alignment-to-Source) score
        ats_score = np.mean(human_within) / np.mean(agent_human)
        results['between_group'][model_name]['ats'] = float(ats_score)
        print(f"      ATS Score: {ats_score:.4f}")

        # ATG (Anger Trajectory Gap) - absolute difference between Human-Human and Human-Model DTW
        atg = abs(np.mean(human_within) - np.mean(agent_human))
        results['between_group'][model_name]['atg'] = float(atg)
        print(f"      ATG (Anger Trajectory Gap): {atg:.4f}")

        # Statistical test
        t_stat, p_val = ttest_ind(human_within, agent_human, equal_var=False)
        results['statistical_tests'][model_name] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val)
        }

    return results


def compute_auc_metrics(
    human_trajs: List[np.ndarray],
    agent_trajs_dict: Dict[str, List[np.ndarray]]
) -> Dict[str, Any]:
    """Compute AUC metrics for emotion trajectories.

    Args:
        human_trajs: Human emotion trajectories
        agent_trajs_dict: Dictionary mapping model names to trajectories

    Returns:
        Dictionary containing AUC metrics and statistical tests
    """
    results = {
        'auc_values': {},
        'auc_differences': {},
        'statistical_tests': {}
    }

    print(f"\n  Computing Human AUC values...")
    # Human AUC
    human_aucs = [calculate_auc(traj) for traj in tqdm(human_trajs, desc="Human AUC", leave=False)]
    results['auc_values']['human'] = {
        'aucs': human_aucs,
        'mean': float(np.mean(human_aucs)),
        'std': float(np.std(human_aucs))
    }
    print(f"    Mean: {np.mean(human_aucs):.4f}")

    # Agent AUC
    for model_name, agent_trajs in agent_trajs_dict.items():
        print(f"\n  Computing {model_name} AUC values...")
        agent_aucs = [calculate_auc(traj) for traj in tqdm(agent_trajs, desc=f"{model_name} AUC", leave=False)]
        results['auc_values'][model_name] = {
            'aucs': agent_aucs,
            'mean': float(np.mean(agent_aucs)),
            'std': float(np.std(agent_aucs))
        }
        print(f"    Mean: {np.mean(agent_aucs):.4f}")

        # AMG (Anger Magnitude Gap) - absolute difference between Human and Model average AUC
        amg = abs(np.mean(agent_aucs) - np.mean(human_aucs))
        results['auc_differences'][model_name] = float(amg)
        print(f"    AMG (Anger Magnitude Gap): {amg:.4f}")

        # Statistical test
        t_stat, p_val = ttest_ind(human_aucs, agent_aucs, equal_var=False)
        results['statistical_tests'][model_name] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val)
        }

    return results


def analyze_anger_trajectory(
    kodis_data: Dict[str, List[Dict]],
    agent_data_dict: Dict[str, Dict[str, List[List[Dict]]]]
) -> Dict[str, Any]:
    """Main function to analyze anger trajectories.

    Args:
        kodis_data: KODIS dataset with emotion annotations
        agent_data_dict: Dictionary mapping model names to conversation data

    Returns:
        Complete analysis results including DTW and AUC metrics
    """
    print("\n1. Extracting emotion trajectories...")

    # Extract KODIS trajectories
    print(f"  Extracting KODIS trajectories ({len(kodis_data)} conversations)...")
    kodis_conversations = list(kodis_data.values())
    kodis_buyer, kodis_seller = extract_all_trajectories(
        kodis_conversations,
        target_emotion='anger',
        is_kodis=True
    )
    human_trajs = kodis_buyer + kodis_seller
    print(f"    Total human trajectories: {len(human_trajs)} (buyer: {len(kodis_buyer)}, seller: {len(kodis_seller)})")

    # Extract agent trajectories
    agent_trajs_dict = {}
    for model_name, model_data in agent_data_dict.items():
        print(f"  Extracting {model_name} trajectories...")
        conversations = model_data['conversation']
        buyer_trajs, seller_trajs = extract_all_trajectories(
            conversations,
            target_emotion='anger',
            is_kodis=False
        )
        agent_trajs_dict[model_name] = buyer_trajs + seller_trajs
        print(f"    Total {model_name} trajectories: {len(agent_trajs_dict[model_name])} (buyer: {len(buyer_trajs)}, seller: {len(seller_trajs)})")

    # Compute metrics
    print("\n2. Computing DTW distance metrics...")
    dtw_results = compute_dtw_metrics(human_trajs, agent_trajs_dict)

    print("\n3. Computing AUC metrics...")
    auc_results = compute_auc_metrics(human_trajs, agent_trajs_dict)

    print("\n✓ Anger trajectory analysis complete!")

    return {
        'dtw_analysis': dtw_results,
        'auc_analysis': auc_results
    }
