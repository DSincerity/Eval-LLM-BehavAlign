"""Strategic behavior analysis using IRP (Interest, Rights, Power) framework."""
import numpy as np
from collections import Counter
from itertools import chain
from scipy.spatial.distance import jensenshannon
from scipy.stats import ttest_ind
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


# Ordered strategy list for consistent distribution calculation
STRATEGY_ORDER = [
    'POSITIVE EXPECTATION',
    'PROPOSAL',
    'CONCESSION',
    'INTEREST',
    'FACTS',
    'PROCEDURAL',
    'POWER',
    'RIGHTS',
    'RESIDUAL',
    'NULL'
]

# Strategy order without NULL for visualization
STRATEGY_ORDER_NO_NULL = STRATEGY_ORDER[:-1]


def normalize_strategy_name(strategy: str) -> str:
    """Normalize strategy names to fix common typos.

    Args:
        strategy: Raw strategy name

    Returns:
        Normalized strategy name
    """
    return strategy.replace("EXPECTATIONS", "EXPECTATION").replace("POSITVE", "POSITIVE")


def extract_strategies_from_session(
    session: List[Dict],
    is_kodis: bool = False
) -> List[str]:
    """Extract all strategies from a session.

    Args:
        session: Single conversation session
        is_kodis: Whether the data is from KODIS dataset

    Returns:
        List of strategies used in the session
    """
    strategy_key = 'strategy' if is_kodis else 'strategies'
    strategies = list(chain(*[turn[strategy_key] for turn in session]))
    return [normalize_strategy_name(s) for s in strategies]


def compute_strategy_distribution(
    strategies: List[str],
    include_null: bool = True
) -> List[float]:
    """Compute percentage distribution of strategies.

    Args:
        strategies: List of strategy names
        include_null: Whether to include NULL category

    Returns:
        List of percentages in STRATEGY_ORDER
    """
    counts = Counter(strategies)
    total = sum(counts.values())

    if total == 0:
        return [0.0] * len(STRATEGY_ORDER)

    percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}

    strategy_list = STRATEGY_ORDER if include_null else STRATEGY_ORDER_NO_NULL
    return [percentages.get(key, 0.0) for key in strategy_list]


def compute_overall_distribution(
    all_sessions: List[List[Dict]],
    is_kodis: bool = False,
    include_null: bool = True
) -> List[float]:
    """Compute overall strategy distribution across all sessions.

    Args:
        all_sessions: All conversation sessions
        is_kodis: Whether the data is from KODIS dataset
        include_null: Whether to include NULL category

    Returns:
        Overall percentage distribution
    """
    all_strategies = []
    for session in all_sessions:
        all_strategies.extend(extract_strategies_from_session(session, is_kodis))

    return compute_strategy_distribution(all_strategies, include_null)


def compute_session_distributions(
    all_sessions: List[List[Dict]],
    is_kodis: bool = False
) -> List[List[float]]:
    """Compute strategy distribution for each session separately.

    Args:
        all_sessions: All conversation sessions
        is_kodis: Whether the data is from KODIS dataset

    Returns:
        List of distributions (one per session)
    """
    distributions = []
    for session in all_sessions:
        strategies = extract_strategies_from_session(session, is_kodis)
        dist = compute_strategy_distribution(strategies, include_null=True)
        distributions.append(dist)

    return distributions


def compute_jsd_within_group(distributions: List[List[float]], desc: str = "Computing JSD") -> Tuple[float, float, List[float]]:
    """Compute Jensen-Shannon Divergence within a group (all pairs).

    Args:
        distributions: List of strategy distributions
        desc: Description for progress bar

    Returns:
        Tuple of (mean_jsd, std_jsd, all_jsd_values)
    """
    jsd_values = []
    n = len(distributions)
    total_pairs = n * (n - 1) // 2

    with tqdm(total=total_pairs, desc=desc, leave=False) as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                jsd = jensenshannon(distributions[i], distributions[j]) ** 2
                jsd_values.append(jsd)
                pbar.update(1)

    if not jsd_values:
        return 0.0, 0.0, []

    return float(np.mean(jsd_values)), float(np.std(jsd_values)), jsd_values


def compute_jsd_between_groups(
    group1_dists: List[List[float]],
    group2_dists: List[List[float]],
    desc: str = "Computing JSD"
) -> Tuple[float, float, List[float]]:
    """Compute Jensen-Shannon Divergence between two groups (cross-product).

    Args:
        group1_dists: Distributions from group 1
        group2_dists: Distributions from group 2
        desc: Description for progress bar

    Returns:
        Tuple of (mean_jsd, std_jsd, all_jsd_values)
    """
    jsd_values = []
    total_pairs = len(group1_dists) * len(group2_dists)

    with tqdm(total=total_pairs, desc=desc, leave=False) as pbar:
        for dist1 in group1_dists:
            for dist2 in group2_dists:
                jsd = jensenshannon(dist1, dist2) ** 2
                jsd_values.append(jsd)
                pbar.update(1)

    if not jsd_values:
        return 0.0, 0.0, []

    return float(np.mean(jsd_values)), float(np.std(jsd_values)), jsd_values


def analyze_strategic_behavior(
    kodis_sessions: List[List[Dict]],
    agent_sessions_dict: Dict[str, List[List[Dict]]]
) -> Dict[str, Any]:
    """Main function to analyze strategic behavior using IRP framework.

    Args:
        kodis_sessions: KODIS conversation sessions
        agent_sessions_dict: Dictionary mapping model names to their sessions

    Returns:
        Complete strategic behavior analysis results
    """
    results = {
        'overall_distributions': {},
        'jsd_within_group': {},
        'jsd_between_groups': {},
        'statistical_tests': {}
    }

    print("\n1. Computing overall strategy distributions...")
    # Compute overall distributions
    print(f"  Computing KODIS distribution ({len(kodis_sessions)} sessions)...")
    results['overall_distributions']['human'] = compute_overall_distribution(
        kodis_sessions,
        is_kodis=True,
        include_null=True
    )

    for model_name, sessions in agent_sessions_dict.items():
        print(f"  Computing {model_name} distribution ({len(sessions)} sessions)...")
        results['overall_distributions'][model_name] = compute_overall_distribution(
            sessions,
            is_kodis=False,
            include_null=True
        )

    print("\n2. Computing session-level distributions...")
    # Compute session-level distributions for JSD analysis
    print(f"  KODIS: Computing {len(kodis_sessions)} session distributions...")
    kodis_dists = compute_session_distributions(kodis_sessions, is_kodis=True)

    print("\n3. Computing JSD metrics...")
    # Human within-group JSD
    print(f"  Computing Human-Human JSD...")
    human_mean_jsd, human_std_jsd, human_jsd_values = compute_jsd_within_group(
        kodis_dists,
        desc="Human-Human JSD"
    )
    results['jsd_within_group']['human'] = {
        'mean': human_mean_jsd,
        'std': human_std_jsd,
        'values': human_jsd_values
    }
    print(f"    Mean: {human_mean_jsd:.4f}")

    # Agent within-group and between-group JSD
    for model_name, sessions in agent_sessions_dict.items():
        print(f"\n  Processing {model_name}...")
        print(f"    Computing {len(sessions)} session distributions...")
        agent_dists = compute_session_distributions(sessions, is_kodis=False)

        # Within-group
        print(f"    Computing {model_name}-{model_name} JSD...")
        agent_mean_jsd, agent_std_jsd, agent_jsd_values = compute_jsd_within_group(
            agent_dists,
            desc=f"{model_name}-{model_name} JSD"
        )
        results['jsd_within_group'][model_name] = {
            'mean': agent_mean_jsd,
            'std': agent_std_jsd,
            'values': agent_jsd_values
        }
        print(f"      Mean: {agent_mean_jsd:.4f}")

        # Between-group (Agent vs Human)
        print(f"    Computing {model_name}-Human JSD...")
        cross_mean_jsd, cross_std_jsd, cross_jsd_values = compute_jsd_between_groups(
            agent_dists,
            kodis_dists,
            desc=f"{model_name}-Human JSD"
        )

        # SBG (Strategic Behavior Gap) - absolute difference between Human-Human and Human-Model JSD
        sbg = abs(human_mean_jsd - cross_mean_jsd)

        results['jsd_between_groups'][model_name] = {
            'mean': cross_mean_jsd,
            'std': cross_std_jsd,
            'values': cross_jsd_values,
            'sbg': float(sbg)
        }
        print(f"      Mean: {cross_mean_jsd:.4f}")
        print(f"      SBG (Strategic Behavior Gap): {sbg:.4f}")

        # Statistical test (Agent-Human vs Human-Human)
        if human_jsd_values and cross_jsd_values:
            t_stat, p_val = ttest_ind(human_jsd_values, cross_jsd_values, equal_var=False)
            results['statistical_tests'][model_name] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val)
            }
        else:
            results['statistical_tests'][model_name] = {
                't_statistic': None,
                'p_value': None
            }

    print("\n✓ Strategic behavior analysis complete!")

    return results
