"""Linguistic Gap analysis using LIWC features."""
import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import jensenshannon
from scipy.stats import ttest_ind
from typing import Dict, Any, Tuple, List
from tqdm import tqdm


# LIWC categories for different analysis types
DISPUTE_CATEGORIES = ["Analytic", "Authentic", "Clout"]
IRP_CATEGORIES = ["insight", "prosocial", "affiliation", "power", "allnone", "polite"]


def normalize_liwc(df: pd.DataFrame, categories: List[str]) -> np.ndarray:
    """Normalize LIWC features by row sum.

    Args:
        df: DataFrame with LIWC features
        categories: List of LIWC category names to use

    Returns:
        Normalized numpy array (each row sums to 1)
    """
    arr = df[categories].to_numpy()
    row_sums = arr.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    return arr / row_sums[:, np.newaxis]


def compute_within_jsd(distributions: np.ndarray, desc: str = "Computing Within-JSD") -> Tuple[float, float, List[float]]:
    """Compute within-group JSD (all pairwise comparisons).

    Args:
        distributions: Normalized LIWC distributions (n_samples x n_features)
        desc: Description for progress bar

    Returns:
        Tuple of (mean_jsd, std_jsd, jsd_values)
    """
    jsd_pairs = []
    skipped = 0
    n = len(distributions)
    total_pairs = n * (n - 1) // 2

    with tqdm(total=total_pairs, desc=desc, leave=False) as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                p = distributions[i]
                q = distributions[j]

                # Skip if NaN present
                if np.isnan(p).any() or np.isnan(q).any():
                    skipped += 1
                    pbar.update(1)
                    continue

                # Suppress scipy warning for division by zero in jensenshannon
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                    jsd = jensenshannon(p, q) ** 2

                if not np.isnan(jsd):
                    jsd_pairs.append(jsd)
                else:
                    skipped += 1
                pbar.update(1)

    if skipped > 0:
        print(f"    Skipped {skipped} pairs due to zero/NaN values")

    if not jsd_pairs:
        return 0.0, 0.0, []

    return float(np.mean(jsd_pairs)), float(np.std(jsd_pairs)), jsd_pairs


def compute_cross_jsd(
    human_dist: np.ndarray,
    model_dist: np.ndarray,
    desc: str = "Computing Cross-JSD"
) -> Tuple[float, float, List[float]]:
    """Compute cross-group JSD between human and model distributions.

    Args:
        human_dist: Human LIWC distributions
        model_dist: Model LIWC distributions
        desc: Description for progress bar

    Returns:
        Tuple of (mean_jsd, std_jsd, jsd_values)
    """
    jsd_pairs = []
    skipped = 0
    total_pairs = len(human_dist) * len(model_dist)

    with tqdm(total=total_pairs, desc=desc, leave=False) as pbar:
        for h in human_dist:
            for m in model_dist:
                # Skip if NaN present
                if np.isnan(h).any() or np.isnan(m).any():
                    skipped += 1
                    pbar.update(1)
                    continue

                # Suppress scipy warning for division by zero in jensenshannon
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                    jsd = jensenshannon(h, m) ** 2

                if not np.isnan(jsd):
                    jsd_pairs.append(jsd)
                else:
                    skipped += 1
                pbar.update(1)

    if skipped > 0:
        print(f"    Skipped {skipped} pairs due to zero/NaN values")

    if not jsd_pairs:
        return 0.0, 0.0, []

    return float(np.mean(jsd_pairs)), float(np.std(jsd_pairs)), jsd_pairs


def analyze_linguistic_gap_dispute(
    kodis_df: pd.DataFrame,
    model_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Analyze Linguistic Gap for Dispute-related LIWC categories.

    Args:
        kodis_df: KODIS (human) LIWC data
        model_dfs: Dictionary mapping model names to their LIWC dataframes

    Returns:
        Analysis results including JSD metrics and LG-Dispute scores
    """
    results = {
        'categories': DISPUTE_CATEGORIES,
        'within_group': {},
        'between_group': {},
        'statistical_tests': {}
    }

    print(f"\n  Computing KODIS (Human) LIWC distributions for Dispute categories...")
    kodis_dist = normalize_liwc(kodis_df, DISPUTE_CATEGORIES)
    print(f"    Normalized {len(kodis_dist)} samples")

    # Human within-group JSD
    print(f"  Computing Human-Human JSD (Dispute)...")
    human_mean, human_std, human_jsd_values = compute_within_jsd(
        kodis_dist,
        desc="Human-Human JSD (Dispute)"
    )
    results['within_group']['human'] = {
        'mean': human_mean,
        'std': human_std,
        'values': human_jsd_values
    }
    print(f"    Mean: {human_mean:.4f}")

    # Model cross-group JSD
    for model_name, model_df in model_dfs.items():
        print(f"\n  Processing {model_name} (Dispute)...")
        model_dist = normalize_liwc(model_df, DISPUTE_CATEGORIES)
        print(f"    Normalized {len(model_dist)} samples")

        print(f"    Computing {model_name}-Human JSD...")
        cross_mean, cross_std, cross_jsd_values = compute_cross_jsd(
            kodis_dist,
            model_dist,
            desc=f"{model_name}-Human JSD (Dispute)"
        )

        # LG-Dispute: absolute difference between Human-Human and Human-Model JSD
        lg_dispute = abs(human_mean - cross_mean)

        results['between_group'][model_name] = {
            'mean': cross_mean,
            'std': cross_std,
            'values': cross_jsd_values,
            'lg_dispute': float(lg_dispute)
        }
        print(f"      Mean: {cross_mean:.4f}")
        print(f"      LG-Dispute: {lg_dispute:.4f}")

        # Statistical test
        if human_jsd_values and cross_jsd_values:
            t_stat, p_val = ttest_ind(human_jsd_values, cross_jsd_values, equal_var=False)
            results['statistical_tests'][model_name] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val)
            }

    return results


def analyze_linguistic_gap_irp(
    kodis_df: pd.DataFrame,
    model_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Analyze Linguistic Gap for IRP-related LIWC categories.

    Args:
        kodis_df: KODIS (human) LIWC data
        model_dfs: Dictionary mapping model names to their LIWC dataframes

    Returns:
        Analysis results including JSD metrics and LG-IRP scores
    """
    results = {
        'categories': IRP_CATEGORIES,
        'within_group': {},
        'between_group': {},
        'statistical_tests': {}
    }

    print(f"\n  Computing KODIS (Human) LIWC distributions for IRP categories...")
    kodis_dist = normalize_liwc(kodis_df, IRP_CATEGORIES)
    print(f"    Normalized {len(kodis_dist)} samples")

    # Human within-group JSD
    print(f"  Computing Human-Human JSD (IRP)...")
    human_mean, human_std, human_jsd_values = compute_within_jsd(
        kodis_dist,
        desc="Human-Human JSD (IRP)"
    )
    results['within_group']['human'] = {
        'mean': human_mean,
        'std': human_std,
        'values': human_jsd_values
    }
    print(f"    Mean: {human_mean:.4f}")

    # Model cross-group JSD
    for model_name, model_df in model_dfs.items():
        print(f"\n  Processing {model_name} (IRP)...")
        model_dist = normalize_liwc(model_df, IRP_CATEGORIES)
        print(f"    Normalized {len(model_dist)} samples")

        print(f"    Computing {model_name}-Human JSD...")
        cross_mean, cross_std, cross_jsd_values = compute_cross_jsd(
            kodis_dist,
            model_dist,
            desc=f"{model_name}-Human JSD (IRP)"
        )

        # LG-IRP: absolute difference between Human-Human and Human-Model JSD
        lg_irp = abs(human_mean - cross_mean)

        results['between_group'][model_name] = {
            'mean': cross_mean,
            'std': cross_std,
            'values': cross_jsd_values,
            'lg_irp': float(lg_irp)
        }
        print(f"      Mean: {cross_mean:.4f}")
        print(f"      LG-IRP: {lg_irp:.4f}")

        # Statistical test
        if human_jsd_values and cross_jsd_values:
            t_stat, p_val = ttest_ind(human_jsd_values, cross_jsd_values, equal_var=False)
            results['statistical_tests'][model_name] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val)
            }

    return results


def analyze_linguistic_gap(
    kodis_liwc_path: str,
    model_liwc_paths: Dict[str, str]
) -> Dict[str, Any]:
    """Main function to analyze linguistic gaps (both Dispute and IRP).

    Args:
        kodis_liwc_path: Path to KODIS LIWC CSV file
        model_liwc_paths: Dictionary mapping model names to their LIWC CSV paths

    Returns:
        Complete linguistic gap analysis results
    """
    print("\n" + "=" * 60)
    print("Linguistic Gap Analysis")
    print("=" * 60)

    # Load KODIS data
    print(f"\nLoading KODIS LIWC data from: {kodis_liwc_path}")
    kodis_df = pd.read_csv(kodis_liwc_path)
    print(f"  Loaded {len(kodis_df)} samples")

    # Load model data
    model_dfs = {}
    for model_name, path in model_liwc_paths.items():
        print(f"\nLoading {model_name} LIWC data from: {path}")
        model_dfs[model_name] = pd.read_csv(path)
        print(f"  Loaded {len(model_dfs[model_name])} samples")

    # Analyze Dispute-related linguistic gap
    print("\n" + "=" * 60)
    print("1. Analyzing LG-Dispute (Dispute-related LIWC categories)")
    print("=" * 60)
    dispute_results = analyze_linguistic_gap_dispute(kodis_df, model_dfs)

    # Analyze IRP-related linguistic gap
    print("\n" + "=" * 60)
    print("2. Analyzing LG-IRP (IRP-related LIWC categories)")
    print("=" * 60)
    irp_results = analyze_linguistic_gap_irp(kodis_df, model_dfs)

    print("\n✓ Linguistic gap analysis complete!")

    return {
        'lg_dispute': dispute_results,
        'lg_irp': irp_results
    }
