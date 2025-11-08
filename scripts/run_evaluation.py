"""Main script for running evaluation on L2L negotiation simulation results."""
import os
import sys
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import analyze_anger_trajectory, analyze_strategic_behavior, analyze_linguistic_gap, analyze_linguistic_entrainment


def find_latest_result_file(output_dir: str, result_type: str) -> str:
    """Find the most recent result file of a given type.

    Args:
        output_dir: Directory to search for result files
        result_type: Type of result file (e.g., 'anger_trajectory', 'strategic_behavior')

    Returns:
        Path to the most recent result file, or None if not found
    """
    pattern = os.path.join(output_dir, f"{result_type}_*.json")
    matching_files = glob.glob(pattern)

    if not matching_files:
        return None

    # Sort by modification time and return the most recent
    matching_files.sort(key=os.path.getmtime, reverse=True)
    return matching_files[0]


def load_json(filepath: str) -> dict:
    """Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded JSON data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    """Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✓ Saved: {filepath}")


def parse_arguments():
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run evaluation on L2L negotiation simulation results'
    )

    # Data paths
    parser.add_argument(
        '--kodis_emo_path',
        type=str,
        default='data/KODIS/KODIS_merged_20_samples_emo_irp.json',
        help='Path to KODIS emotion-annotated data'
    )
    parser.add_argument(
        '--kodis_irp_path',
        type=str,
        default='data/KODIS/KODIS_merged_20_samples_emo_irp.json',
        help='Path to KODIS IRP-annotated data'
    )

    # Agent data paths (emotion + IRP)
    parser.add_argument(
        '--agent_data_dir',
        type=str,
        default='data/complete',
        help='Directory containing complete simulation results (default: data/complete)'
    )

    # Model specifications
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='List of model names to evaluate (default: auto-detect all models in agent_data_dir)'
    )
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='Auto-detect all models in agent_data_dir'
    )

    # Output configuration
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--report_name',
        type=str,
        default='evaluation_report',
        help='Base name for evaluation report'
    )

    # Analysis options (use either --metrics or --skip-* flags, not both)
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        choices=['anger', 'strategic', 'linguistic', 'entrainment', 'all'],
        help='Specific metrics to run (e.g., --metrics anger strategic). Use "all" for all metrics.'
    )
    parser.add_argument(
        '--skip_anger',
        action='store_true',
        help='Skip anger trajectory analysis (deprecated: use --metrics instead)'
    )
    parser.add_argument(
        '--skip_strategic',
        action='store_true',
        help='Skip strategic behavior analysis (deprecated: use --metrics instead)'
    )
    parser.add_argument(
        '--skip_linguistic',
        action='store_true',
        help='Skip linguistic gap analysis (deprecated: use --metrics instead)'
    )
    parser.add_argument(
        '--skip_entrainment',
        action='store_true',
        help='Skip linguistic entrainment analysis (deprecated: use --metrics instead)'
    )

    # LIWC data paths
    parser.add_argument(
        '--liwc_dir',
        type=str,
        default='data/LIWC',
        help='Directory containing LIWC CSV files'
    )

    # Linguistic Entrainment paths
    parser.add_argument(
        '--entrainment_dir',
        type=str,
        default='data/linguistic_entrainment',
        help='Directory to cache/load linguistic entrainment values'
    )

    # Caching options
    parser.add_argument(
        '--use_cache',
        action='store_true',
        default=True,
        help='Use cached result files if available (default: True)'
    )
    parser.add_argument(
        '--no_cache',
        action='store_true',
        help='Force recomputation, ignore cached result files'
    )

    return parser.parse_args()


def generate_markdown_report(
    anger_results: dict,
    strategic_results: dict,
    linguistic_results: dict,
    entrainment_results: dict,
    output_path: str,
    models: list
):
    """Generate a Markdown report summarizing evaluation results.

    Args:
        anger_results: Anger trajectory analysis results
        strategic_results: Strategic behavior analysis results
        linguistic_results: Linguistic gap analysis results
        entrainment_results: Linguistic entrainment analysis results
        output_path: Path to save the report
        models: List of model names evaluated
    """
    report_lines = []

    # Header
    report_lines.append("# L2L Negotiation Evaluation Report")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n**Models Evaluated:** {', '.join(models)}")
    report_lines.append("\n---\n")

    # Anger Trajectory Analysis
    if anger_results:
        report_lines.append("## 1. Anger Trajectory Analysis\n")

        # DTW Analysis
        report_lines.append("### 1.1 DTW Distance Analysis\n")
        report_lines.append("**Within-Group DTW Distances:**\n")

        dtw_data = anger_results['dtw_analysis']

        # Table header
        report_lines.append("| Group | Mean DTW | Std DTW |")
        report_lines.append("|-------|----------|---------|")

        # Human within
        human_within = dtw_data['within_group']['human']
        report_lines.append(
            f"| Human-Human | {human_within['mean']:.3f} | {human_within['std']:.3f} |"
        )

        # Agent within
        for model in models:
            if model in dtw_data['within_group']:
                within = dtw_data['within_group'][model]
                report_lines.append(
                    f"| {model}-{model} | {within['mean']:.3f} | {within['std']:.3f} |"
                )

        report_lines.append("\n**Between-Group DTW Distances (Agent vs Human):**\n")
        report_lines.append("| Model | Mean DTW | Std DTW | ATS Score | ATG | T-statistic | P-value |")
        report_lines.append("|-------|----------|---------|-----------|-----|-------------|---------|")

        for model in models:
            if model in dtw_data['between_group']:
                between = dtw_data['between_group'][model]
                test = dtw_data['statistical_tests'][model]
                report_lines.append(
                    f"| {model} | {between['mean']:.3f} | {between['std']:.3f} | "
                    f"{between['ats']:.3f} | {between['atg']:.3f} | "
                    f"{test['t_statistic']:.3f} | {test['p_value']:.3f} |"
                )

        # AUC Analysis
        report_lines.append("\n### 1.2 AUC Analysis (Anger Magnitude)\n")
        report_lines.append("| Group | Mean AUC | Std AUC | AMG | T-statistic | P-value |")
        report_lines.append("|-------|----------|---------|-----|-------------|---------|")

        auc_data = anger_results['auc_analysis']

        # Human
        human_auc = auc_data['auc_values']['human']
        report_lines.append(
            f"| Human | {human_auc['mean']:.3f} | {human_auc['std']:.3f} | - | - | - |"
        )

        # Agents
        for model in models:
            if model in auc_data['auc_values']:
                auc = auc_data['auc_values'][model]
                amg = auc_data['auc_differences'][model]  # This is AMG
                test = auc_data['statistical_tests'][model]
                report_lines.append(
                    f"| {model} | {auc['mean']:.3f} | {auc['std']:.3f} | "
                    f"{amg:.3f} | {test['t_statistic']:.3f} | {test['p_value']:.3f} |"
                )

        report_lines.append("\n---\n")

    # Strategic Behavior Analysis
    if strategic_results:
        report_lines.append("## 2. Strategic Behavior Analysis\n")

        # Overall distribution
        report_lines.append("### 2.1 Overall IRP Strategy Distribution\n")
        from src.evaluation import STRATEGY_ORDER

        report_lines.append("| Strategy | Human | " + " | ".join(models) + " |")
        report_lines.append("|----------|-------|" + "|".join(["-------"] * len(models)) + "|")

        dists = strategic_results['overall_distributions']
        for i, strategy in enumerate(STRATEGY_ORDER):
            row = f"| {strategy} | {dists['human'][i]:.2f}% |"
            for model in models:
                if model in dists:
                    row += f" {dists[model][i]:.2f}% |"
            report_lines.append(row)

        # JSD Analysis
        report_lines.append("\n### 2.2 Jensen-Shannon Divergence Analysis\n")
        report_lines.append("| Comparison | Mean JSD | Std JSD | SBG | T-statistic | P-value |")
        report_lines.append("|------------|----------|---------|-----|-------------|---------|")

        # Human within
        human_jsd = strategic_results['jsd_within_group']['human']
        report_lines.append(
            f"| Human-Human | {human_jsd['mean']:.3f} | {human_jsd['std']:.3f} | - | - | - |"
        )

        # Agent vs Human
        for model in models:
            if model in strategic_results['jsd_between_groups']:
                between_jsd = strategic_results['jsd_between_groups'][model]
                test = strategic_results['statistical_tests'][model]

                t_stat = test['t_statistic'] if test['t_statistic'] is not None else 'N/A'
                p_val = test['p_value'] if test['p_value'] is not None else 'N/A'

                if isinstance(t_stat, float):
                    t_stat = f"{t_stat:.3f}"
                if isinstance(p_val, float):
                    p_val = f"{p_val:.3f}"

                sbg = between_jsd.get('sbg', 0.0)
                report_lines.append(
                    f"| {model} vs Human | {between_jsd['mean']:.3f} | "
                    f"{between_jsd['std']:.3f} | {sbg:.3f} | {t_stat} | {p_val} |"
                )

        report_lines.append("\n---\n")

    # Linguistic Gap Analysis
    if not linguistic_results:
        # Add a note if LIWC data is missing
        report_lines.append("## 3. Linguistic Gap Analysis\n\n")
        report_lines.append("**LIWC data not available.** Please generate LIWC-22 CSV files to compute Linguistic Gap metrics (LG-Dispute and LG-IRP).\n\n")
        report_lines.append("See the LIWC Analysis section in README.md for instructions on generating LIWC files.\n\n")
        report_lines.append("---\n\n")

    if linguistic_results:
        report_lines.append("## 3. Linguistic Gap Analysis\n")

        # LG-Dispute
        if 'lg_dispute' in linguistic_results:
            report_lines.append("### 3.1 LG-Dispute (Dispute-related LIWC categories)\n")
            report_lines.append(f"**Categories:** {', '.join(linguistic_results['lg_dispute']['categories'])}\n\n")

            dispute_data = linguistic_results['lg_dispute']

            # Human within
            human_dispute = dispute_data['within_group']['human']
            report_lines.append("| Comparison | Mean JSD | Std JSD | LG-Dispute | T-statistic | P-value |")
            report_lines.append("|------------|----------|---------|------------|-------------|---------|")
            report_lines.append(
                f"| Human-Human | {human_dispute['mean']:.3f} | {human_dispute['std']:.3f} | - | - | - |"
            )

            # Models
            for model in models:
                if model in dispute_data['between_group']:
                    between = dispute_data['between_group'][model]
                    test = dispute_data['statistical_tests'].get(model, {})

                    t_stat = test.get('t_statistic', 'N/A')
                    p_val = test.get('p_value', 'N/A')

                    if isinstance(t_stat, float):
                        t_stat = f"{t_stat:.3f}"
                    if isinstance(p_val, float):
                        p_val = f"{p_val:.3f}"

                    lg_dispute = between.get('lg_dispute', 0.0)
                    report_lines.append(
                        f"| {model} vs Human | {between['mean']:.3f} | "
                        f"{between['std']:.3f} | {lg_dispute:.3f} | {t_stat} | {p_val} |"
                    )

        # LG-IRP
        if 'lg_irp' in linguistic_results:
            report_lines.append("\n### 3.2 LG-IRP (IRP-related LIWC categories)\n")
            report_lines.append(f"**Categories:** {', '.join(linguistic_results['lg_irp']['categories'])}\n\n")

            irp_data = linguistic_results['lg_irp']

            # Human within
            human_irp = irp_data['within_group']['human']
            report_lines.append("| Comparison | Mean JSD | Std JSD | LG-IRP | T-statistic | P-value |")
            report_lines.append("|------------|----------|---------|--------|-------------|---------|")
            report_lines.append(
                f"| Human-Human | {human_irp['mean']:.3f} | {human_irp['std']:.3f} | - | - | - |"
            )

            # Models
            for model in models:
                if model in irp_data['between_group']:
                    between = irp_data['between_group'][model]
                    test = irp_data['statistical_tests'].get(model, {})

                    t_stat = test.get('t_statistic', 'N/A')
                    p_val = test.get('p_value', 'N/A')

                    if isinstance(t_stat, float):
                        t_stat = f"{t_stat:.3f}"
                    if isinstance(p_val, float):
                        p_val = f"{p_val:.3f}"

                    lg_irp = between.get('lg_irp', 0.0)
                    report_lines.append(
                        f"| {model} vs Human | {between['mean']:.3f} | "
                        f"{between['std']:.3f} | {lg_irp:.3f} | {t_stat} | {p_val} |"
                    )

        report_lines.append("\n---\n")

    # Linguistic Entrainment Analysis
    if entrainment_results:
        report_lines.append("## 4. Linguistic Entrainment Gap (LEG) Analysis\n")
        report_lines.append("**Metric**: nCLiD (normalized Conversational Linguistic Distance)\n\n")

        report_lines.append("| Group/Model | Mean LE | Std LE | LEG | T-statistic | P-value |")
        report_lines.append("|-------------|---------|--------|-----|-------------|---------|")

        # Human baseline
        kodis_data = entrainment_results.get('kodis', {})
        report_lines.append(
            f"| Human (KODIS) | {kodis_data.get('mean', 0.0):.3f} | "
            f"{kodis_data.get('std', 0.0):.3f} | - | - | - |"
        )

        # Models
        for model in models:
            if model in entrainment_results.get('models', {}):
                model_data = entrainment_results['models'][model]
                test_data = entrainment_results.get('statistical_tests', {}).get(model, {})

                leg = model_data.get('leg', 0.0)
                t_stat = test_data.get('t_statistic', 'N/A')
                p_val = test_data.get('p_value', 'N/A')

                if isinstance(t_stat, float):
                    t_stat = f"{t_stat:.3f}"
                if isinstance(p_val, float):
                    p_val = f"{p_val:.3f}"

                report_lines.append(
                    f"| {model} | {model_data.get('mean', 0.0):.3f} | "
                    f"{model_data.get('std', 0.0):.3f} | {leg:.3f} | {t_stat} | {p_val} |"
                )

        report_lines.append("\n**Note**: LEG (Linguistic Entrainment Gap) = |Human Mean LE - Model Mean LE|\n")
        report_lines.append("Lower LEG indicates more human-like linguistic coordination patterns.\n")

        report_lines.append("\n---\n")

    # Interpretation
    report_lines.append("## 5. Interpretation\n")
    report_lines.append("### Gap Metrics (Lower is Better)\n")
    report_lines.append(
        "- **ATG (Anger Trajectory Gap)**: Absolute difference between Human-Human DTW and Human-Model DTW\n"
        "  - Measures how different the model's emotion trajectory patterns are from human baseline\n"
        "- **AMG (Anger Magnitude Gap)**: Absolute difference between Human and Model average AUC\n"
        "  - Measures how different the overall anger intensity is from humans\n"
        "- **SBG (Strategic Behavior Gap)**: Absolute difference between Human-Human JSD and Human-Model JSD\n"
        "  - Measures how different the model's strategy usage variance is from human baseline\n"
        "- **LG-Dispute**: Absolute difference between Human-Human and Human-Model JSD for dispute-related LIWC features\n"
        "  - Measures linguistic divergence in dispute-related language patterns\n"
        "- **LG-IRP**: Absolute difference between Human-Human and Human-Model JSD for IRP-related LIWC features\n"
        "  - Measures linguistic divergence in Interest/Rights/Power-related language patterns\n"
        "- **LEG (Linguistic Entrainment Gap)**: Absolute difference between Human and Model mean nCLiD values\n"
        "  - Measures how different the model's linguistic coordination patterns are from humans\n"
    )
    report_lines.append("\n### DTW Distance (Anger Trajectory)\n")
    report_lines.append(
        "- **Lower DTW distance** = More similar emotion trajectories\n"
        "- **Higher ATS score** = Better alignment with human emotional dynamics\n"
    )
    report_lines.append("\n### AUC (Anger Magnitude)\n")
    report_lines.append(
        "- **Lower AMG** = Similar overall anger levels to humans\n"
    )
    report_lines.append("\n### JSD (Strategic Behavior)\n")
    report_lines.append(
        "- **Lower JSD** = More similar strategy usage patterns\n"
        "- **Lower SBG** = Strategy variance closer to Human-Human baseline\n"
    )

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Report saved: {output_path}")


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Auto-detect models if not specified
    if args.models is None or args.auto_detect:
        print(f"Auto-detecting models from {args.agent_data_dir}...")
        detected_models = []
        if os.path.exists(args.agent_data_dir):
            for file in os.listdir(args.agent_data_dir):
                if file.endswith('_complete.json'):
                    model_name = file.replace('_complete.json', '')
                    detected_models.append(model_name)

        if detected_models:
            args.models = detected_models
            print(f"Detected models: {', '.join(args.models)}")
        else:
            print(f"No models found in {args.agent_data_dir}")
            if args.models is None:
                print("Using default models: gpt-4.1, gpt-4.1-mini, claude-3-7-sonnet-20250219, gemini-2.0-flash")
                args.models = ['gpt-4.1', 'gpt-4.1-mini', 'claude-3-7-sonnet-20250219', 'gemini-2.0-flash']

    # Determine caching behavior
    use_cache = args.use_cache and not args.no_cache

    # Process metrics selection
    if args.metrics:
        # If --metrics is specified, use it
        if 'all' in args.metrics:
            run_anger = True
            run_strategic = True
            run_linguistic = True
            run_entrainment = True
        else:
            run_anger = 'anger' in args.metrics
            run_strategic = 'strategic' in args.metrics
            run_linguistic = 'linguistic' in args.metrics
            run_entrainment = 'entrainment' in args.metrics
    else:
        # Fall back to --skip_* flags (deprecated)
        run_anger = not args.skip_anger
        run_strategic = not args.skip_strategic
        run_linguistic = not args.skip_linguistic
        run_entrainment = not args.skip_entrainment

    print("=" * 60)
    print("L2L Negotiation Evaluation")
    print("=" * 60)
    print(f"\nMetrics to run:")
    print(f"  - Anger Trajectory: {'Yes' if run_anger else 'No'}")
    print(f"  - Strategic Behavior: {'Yes' if run_strategic else 'No'}")
    print(f"  - Linguistic Gap: {'Yes' if run_linguistic else 'No'}")
    print(f"  - Linguistic Entrainment: {'Yes' if run_entrainment else 'No'}")
    print(f"\nCache usage: {'Enabled' if use_cache else 'Disabled (force recomputation)'}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}

    # Load KODIS data
    print("\nLoading KODIS data...")
    kodis_emo_data = None
    kodis_irp_data = None

    if run_anger:
        if os.path.exists(args.kodis_emo_path):
            kodis_emo_data = load_json(args.kodis_emo_path)
            print(f"  ✓ Loaded emotion data: {args.kodis_emo_path}")
        else:
            print(f"  ✗ Emotion data not found: {args.kodis_emo_path}")

    if run_strategic:
        if os.path.exists(args.kodis_irp_path):
            kodis_irp_data = load_json(args.kodis_irp_path)
            kodis_irp_data = [session for session in kodis_irp_data.values()]
            print(f"  ✓ Loaded IRP data: {args.kodis_irp_path}")
        else:
            print(f"  ✗ IRP data not found: {args.kodis_irp_path}")

    # Load agent data
    print("\nLoading agent simulation data...")
    agent_emo_data = {}
    agent_irp_data = {}

    import glob

    for model in args.models:
        # Primary path: data/complete/<model>_complete.json
        complete_file = f"{args.agent_data_dir}/{model}_complete.json"

        # Fallback search patterns
        search_patterns = [
            complete_file,
            f"{args.agent_data_dir}/{model}-merged_*_emo*irp*.json",
            f"{args.agent_data_dir}/{model}-merged_*_emo*.json",
            f"{args.agent_data_dir}/{model}_emo*.json",
        ]

        model_data = None
        matched_file = None

        # Try to find a file with the required data
        for pattern in search_patterns:
            matches = glob.glob(pattern) if '*' in pattern else ([pattern] if os.path.exists(pattern) else [])

            for match in matches:
                try:
                    test_data = load_json(match)

                    # Check if file has the required data
                    has_emotion = ('conversation' in test_data and
                                 test_data['conversation'] and
                                 isinstance(test_data['conversation'][0], list) and
                                 test_data['conversation'][0] and
                                 'emotion' in test_data['conversation'][0][0])

                    has_irp = ('irp_1' in test_data or 'irp_2' in test_data)

                    # Use this file if it has the data we need
                    if (run_anger and has_emotion) or (run_strategic and has_irp):
                        model_data = test_data
                        matched_file = match

                        if run_anger and has_emotion:
                            agent_emo_data[model] = model_data
                            print(f"  ✓ {model} emotion data: {match}")

                        if run_strategic and has_irp:
                            agent_irp_data[model] = model_data
                            print(f"  ✓ {model} IRP data: {match}")

                        break  # Found a good file, no need to continue
                except Exception as e:
                    continue

            if model_data:
                break  # Found data for this model

        # Report if data not found
        if run_anger and model not in agent_emo_data:
            print(f"  ✗ {model} emotion data not found")
            print(f"     Expected: {complete_file}")
        if run_strategic and model not in agent_irp_data:
            print(f"  ✗ {model} IRP data not found")
            print(f"     Expected: {complete_file}")

    # Run Anger Trajectory Analysis
    if run_anger and kodis_emo_data and agent_emo_data:
        print("\n" + "=" * 60)
        print("Running Anger Trajectory Analysis...")
        print("=" * 60)

        try:
            # Check for cached results
            cached_file = None
            if use_cache:
                cached_file = find_latest_result_file(args.output_dir, 'anger_trajectory')
                if cached_file:
                    print(f"  ✓ Found cached results: {cached_file}")
                    print("  → Loading from cache...")
                    anger_results = load_json(cached_file)
                    results['anger_trajectory'] = anger_results
                    print("✓ Anger trajectory analysis loaded from cache")

            # Compute if no cache or cache disabled
            if not use_cache or not cached_file:
                if not use_cache:
                    print("  → Cache disabled, computing...")
                else:
                    print("  → No cached results found, computing...")

                anger_results = analyze_anger_trajectory(kodis_emo_data, agent_emo_data)
                results['anger_trajectory'] = anger_results

                # Save results
                output_file = os.path.join(
                    args.output_dir,
                    f"anger_trajectory_{timestamp}.json"
                )
                save_json(anger_results, output_file)
                print("✓ Anger trajectory analysis completed")

        except Exception as e:
            print(f"✗ Error in anger trajectory analysis: {e}")
            import traceback
            traceback.print_exc()

    # Run Strategic Behavior Analysis
    if run_strategic and kodis_irp_data and agent_irp_data:
        print("\n" + "=" * 60)
        print("Running Strategic Behavior Analysis...")
        print("=" * 60)

        try:
            # Check for cached results
            cached_file = None
            if use_cache:
                cached_file = find_latest_result_file(args.output_dir, 'strategic_behavior')
                if cached_file:
                    print(f"  ✓ Found cached results: {cached_file}")
                    print("  → Loading from cache...")
                    strategic_results = load_json(cached_file)
                    results['strategic_behavior'] = strategic_results
                    print("✓ Strategic behavior analysis loaded from cache")

            # Compute if no cache or cache disabled
            if not use_cache or not cached_file:
                if not use_cache:
                    print("  → Cache disabled, computing...")
                else:
                    print("  → No cached results found, computing...")

                # Convert agent IRP data format
                agent_irp_sessions = {}
                for model, data in agent_irp_data.items():
                    # Handle different data formats
                    if 'irp_1' in data:
                        agent_irp_sessions[model] = data['irp_1']
                    elif isinstance(data, list):
                        agent_irp_sessions[model] = data
                    else:
                        print(f"  ⚠ Unknown IRP data format for {model}")
                        continue

                strategic_results = analyze_strategic_behavior(
                    kodis_irp_data,
                    agent_irp_sessions
                )
                results['strategic_behavior'] = strategic_results

                # Save results
                output_file = os.path.join(
                    args.output_dir,
                    f"strategic_behavior_{timestamp}.json"
                )
                save_json(strategic_results, output_file)
                print("✓ Strategic behavior analysis completed")

        except Exception as e:
            print(f"✗ Error in strategic behavior analysis: {e}")
            import traceback
            traceback.print_exc()

    # Run Linguistic Gap Analysis
    if run_linguistic:
        print("\n" + "=" * 60)
        print("Running Linguistic Gap Analysis...")
        print("=" * 60)

        try:
            # Check for cached results
            cached_file = None
            if use_cache:
                cached_file = find_latest_result_file(args.output_dir, 'linguistic_gap')
                if cached_file:
                    print(f"  ✓ Found cached results: {cached_file}")
                    print("  → Loading from cache...")
                    linguistic_results = load_json(cached_file)
                    results['linguistic_gap'] = linguistic_results
                    print("✓ Linguistic gap analysis loaded from cache")

            # Compute if no cache or cache disabled
            if not use_cache or not cached_file:
                if not use_cache:
                    print("  → Cache disabled, computing...")
                else:
                    print("  → No cached results found, computing...")

                # Build LIWC file paths using glob pattern to find files like LIWC_22_Aggregated_KODIS*.csv
                kodis_liwc_pattern = os.path.join(args.liwc_dir, "LIWC_22_Aggregated_KODIS*.csv")
                kodis_liwc_matches = glob.glob(kodis_liwc_pattern)
                kodis_liwc_path = kodis_liwc_matches[0] if kodis_liwc_matches else None

                # Map model names to LIWC file names
                model_name_mapping = {
                    'gpt-4.1': 'gpt-4.1',
                    'gpt-4.1-mini': 'gpt-4.1-mini',
                    'claude-3-7-sonnet-20250219': 'claude',
                    'gemini-2.0-flash': 'gemini'
                }

                model_liwc_paths = {}
                for model in args.models:
                    liwc_model_name = model_name_mapping.get(model, model)
                    liwc_pattern = os.path.join(args.liwc_dir, f"LIWC_22_Aggregated_{liwc_model_name}*.csv")
                    liwc_matches = glob.glob(liwc_pattern)

                    if liwc_matches:
                        model_liwc_paths[model] = liwc_matches[0]
                    else:
                        print(f"  ⚠ LIWC file not found for {model}: {liwc_pattern}")

                if kodis_liwc_path and model_liwc_paths:
                    linguistic_results = analyze_linguistic_gap(
                        kodis_liwc_path,
                        model_liwc_paths
                    )
                    results['linguistic_gap'] = linguistic_results

                    # Save results
                    output_file = os.path.join(
                        args.output_dir,
                        f"linguistic_gap_{timestamp}.json"
                    )
                    save_json(linguistic_results, output_file)
                    print("✓ Linguistic gap analysis completed")
                else:
                    if not kodis_liwc_path:
                        print(f"  ✗ KODIS LIWC file not found: {kodis_liwc_pattern}")
                    if not model_liwc_paths:
                        print("  ✗ No model LIWC files found")

        except Exception as e:
            print(f"✗ Error in linguistic gap analysis: {e}")
            import traceback
            traceback.print_exc()

    # Run Linguistic Entrainment Analysis
    if run_entrainment:
        print("\n" + "=" * 60)
        print("Running Linguistic Entrainment Analysis...")
        print("=" * 60)

        try:
            # Check for cached results
            cached_file = None
            if use_cache:
                cached_file = find_latest_result_file(args.output_dir, 'linguistic_entrainment')
                if cached_file:
                    print(f"  ✓ Found cached results: {cached_file}")
                    print("  → Loading from cache...")
                    entrainment_results = load_json(cached_file)
                    results['linguistic_entrainment'] = entrainment_results
                    print("✓ Linguistic entrainment analysis loaded from cache")

            # Compute if no cache or cache disabled
            if not use_cache or not cached_file:
                if not use_cache:
                    print("  → Cache disabled, computing...")
                else:
                    print("  → No cached results found, computing...")

                # Build model data paths
                model_name_mapping = {
                    'gpt-4.1': 'gpt-4.1',
                    'gpt-4.1-mini': 'gpt-4.1-mini',
                    'claude-3-7-sonnet-20250219': 'claude',
                    'gemini-2.0-flash': 'gemini'
                }

                model_data_paths = {}
                for model in args.models:
                    # Try to find model data file
                    search_patterns = [
                        f"{args.agent_data_dir}/{model}-merged_*_emo*irp*.json",
                        f"{args.agent_data_dir}/{model}-merged_*_emo*.json",
                        f"{args.agent_data_dir}/{model}_emo*.json",
                    ]

                    for pattern in search_patterns:
                        matches = glob.glob(pattern)
                        if matches:
                            model_data_paths[model] = matches[0]
                            break

                    if model not in model_data_paths:
                        print(f"  ⚠ Data file not found for {model}")

                if os.path.exists(args.kodis_emo_path) and model_data_paths:
                    entrainment_results = analyze_linguistic_entrainment(
                        args.kodis_emo_path,
                        model_data_paths,
                        output_dir=args.entrainment_dir
                    )
                    results['linguistic_entrainment'] = entrainment_results

                    # Save results
                    output_file = os.path.join(
                        args.output_dir,
                        f"linguistic_entrainment_{timestamp}.json"
                    )
                    save_json(entrainment_results, output_file)
                    print("✓ Linguistic entrainment analysis completed")
                else:
                    if not os.path.exists(args.kodis_emo_path):
                        print(f"  ✗ KODIS emotion data not found: {args.kodis_emo_path}")
                    if not model_data_paths:
                        print("  ✗ No model data files found")

        except Exception as e:
            print(f"✗ Error in linguistic entrainment analysis: {e}")
            import traceback
            traceback.print_exc()

    # Generate combined report
    if results:
        print("\n" + "=" * 60)
        print("Generating Report...")
        print("=" * 60)

        report_path = os.path.join(
            args.output_dir,
            f"{args.report_name}_{timestamp}.md"
        )

        generate_markdown_report(
            results.get('anger_trajectory'),
            results.get('strategic_behavior'),
            results.get('linguistic_gap'),
            results.get('linguistic_entrainment'),
            report_path,
            args.models
        )

        # Save complete results
        complete_results_path = os.path.join(
            args.output_dir,
            f"complete_results_{timestamp}.json"
        )
        save_json(results, complete_results_path)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
