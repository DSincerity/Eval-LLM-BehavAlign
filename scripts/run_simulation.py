"""Main script for running L2L negotiation simulations."""
import os
import sys
import time
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation import DialogAgent, run_experiment, summarize_results
from src.utils import Logger, load_json, save_dict_to_json


def parse_arguments():
    """Define and parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run L2L negotiation simulation')

    # Agent configuration
    parser.add_argument('--agent_1_engine', type=str, default="gpt-4o-mini",
                        help='LLM engine for agent 1 (seller)')
    parser.add_argument('--agent_2_engine', type=str, default="gpt-4o-mini",
                        help='LLM engine for agent 2 (buyer)')

    # API keys (optional, can use environment variables)
    parser.add_argument('--api_key', type=str, default="",
                        help='API key for OpenAI (optional if set in env)')

    # Simulation parameters
    parser.add_argument('--n_exp', type=int, default=1,
                        help='Number of experiments to run')
    parser.add_argument('--n_round', type=int, default=10,
                        help='Maximum number of rounds per negotiation')
    parser.add_argument('--personality_setting', type=bool, default=True,
                        help='Enable personality variation')

    # I/O parameters
    parser.add_argument('--log_path', type=str, default="logs/",
                        help='Path to log directory')
    parser.add_argument('--data_output_dir', type=str, default="data/simulations",
                        help='Directory to save simulation data files')
    parser.add_argument('--save_metrics', type=str, default="metrics.json",
                        help='Filename for output metrics')
    parser.add_argument('--ver', type=str, default="default",
                        help='Version identifier for this run')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=quiet, 1=normal)')

    return parser.parse_args()


def main():
    """Main function to run the simulation."""
    args = parse_arguments()

    # Setup API keys from environment if not provided
    if not args.api_key:
        if "gpt" in args.agent_1_engine.lower() or "gpt" in args.agent_2_engine.lower():
            args.api_key = os.environ.get("OPENAI_API_KEY", "")
            if not args.api_key:
                print("ERROR: OPENAI_API_KEY environment variable not set!")
                print("Please set it with: export OPENAI_API_KEY='your-key-here'")
                print("Or run: python scripts/check_env.py")
                sys.exit(1)

    # Validate Claude API key
    if "claude" in args.agent_1_engine.lower() or "claude" in args.agent_2_engine.lower():
        # Support both ANTHROPIC_API_KEY (standard) and anthropic_key (legacy)
        claude_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("anthropic_key", "")
        if not claude_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
            print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
            print("Or run: python scripts/check_env.py")
            sys.exit(1)

    # Validate Gemini API key
    if "gemini" in args.agent_1_engine.lower() or "gemini" in args.agent_2_engine.lower():
        # Support both GEMINI_API_KEY (standard) and gemini_key (legacy)
        gemini_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("gemini_key", "")
        if not gemini_key:
            print("ERROR: GEMINI_API_KEY environment variable not set!")
            print("Please set it with: export GEMINI_API_KEY='your-key-here'")
            print("Or run: python scripts/check_env.py")
            sys.exit(1)

    # Create log and data directories
    args.log_path = os.path.join(args.log_path, args.agent_1_engine)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.data_output_dir, exist_ok=True)

    # Setup logger
    time_string = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(
        args.log_path,
        f"agent_simulation_{time_string}_{args.agent_1_engine}_{args.ver}.txt"
    )

    # Save metrics in logs (for debugging)
    args.save_metrics = os.path.join(
        args.log_path,
        f"agent_simulation_{time_string}_{args.agent_1_engine}_{args.ver}.json"
    )

    # Also save in data/simulations for pipeline processing
    data_output_file = os.path.join(
        args.data_output_dir,
        f"{args.agent_1_engine}.json"
    )

    logger = Logger(log_file, args.verbose == 1)

    # Load personality adjectives
    prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "final")
    personality_adj_path = os.path.join(prompt_dir, "personality_adjective.json")
    personality_adj_dict = load_json(personality_adj_path)

    # Initialize agents
    logger.write(f"Initializing agents...")
    logger.write(f"  Agent 1 (Seller): {args.agent_1_engine}")
    logger.write(f"  Agent 2 (Buyer): {args.agent_2_engine}")

    agent_1 = DialogAgent(
        agent_type="",
        engine=args.agent_1_engine,
        api_key=args.api_key,
        system_instruction=""
    )

    agent_2 = DialogAgent(
        agent_type="",
        engine=args.agent_2_engine,
        api_key=args.api_key,
        system_instruction=""
    )

    # Run experiments
    logger.write(f"\nStarting {args.n_exp} experiments with {args.n_round} rounds each...")
    logger.write("=" * 80)

    outputs = run_experiment(
        agent_1=agent_1,
        agent_2=agent_2,
        n_exp=args.n_exp,
        n_round=args.n_round,
        prompt_dir=prompt_dir,
        personality_adj_dict=personality_adj_dict,
        logger=logger,
        personality_setting=args.personality_setting,
        version=args.ver
    )

    # Summarize results
    logger.write("\n" + "=" * 80)
    logger.write("SUMMARY")
    logger.write("=" * 80)

    collected_data = summarize_results(outputs, args.agent_1_engine, args.agent_2_engine)

    logger.write(f"\nTotal experiments: {len(outputs)}")
    logger.write(f"Termination rate: {collected_data['terminated_rate']:.2%}")
    logger.write(f"Average rounds: {collected_data['avg_rounds']:.2f}")
    logger.write(f"Average turns: {collected_data['avg_turns']:.2f}")
    logger.write(f"\nCase distribution:")
    for case, count in collected_data['case_dist'].items():
        logger.write(f"  {case}: {count}")

    # Save results
    save_dict_to_json(collected_data, args.save_metrics)
    logger.write(f"\nResults saved to: {args.save_metrics}")

    # Also save to data/simulations for pipeline processing
    save_dict_to_json(collected_data, data_output_file)
    logger.write(f"Data file saved to: {data_output_file}")
    logger.write(f"\nNext steps:")
    logger.write(f"  1. Emotion annotation: python scripts/annotate_emotion.py --model {args.agent_1_engine}")
    logger.write(f"  2. IRP annotation: python scripts/annotate_irp.py --model {args.agent_1_engine}")
    logger.write(f"  3. Evaluation: python scripts/run_evaluation.py --models {args.agent_1_engine}")


if __name__ == "__main__":
    main()
