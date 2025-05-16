#!/usr/bin/env python3
"""
Example script demonstrating how to use the SpecEval framework
to rank pre-generated model outputs using either OpenAI or Anthropic models as the ranker.

This script reads generated outputs from different models (structured under generations-dir)
and uses a ranking model to perform pairwise comparisons based on alignment with statements
from a specification.
"""

import os
import argparse
from pathlib import Path

# Import necessary components from speceval
from speceval import JsonlParser
from speceval.pipelines.ranking_pipeline import RankingPipeline

# flake8: noqa: F401
from speceval.models.openai import OpenAIRankingModel, GPT_4_1_MINI, GPT_4_1
from speceval.models.anthropic import (
    AnthropicRankingModel,
    CLAUDE_3_5_HAIKU,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_7_SONNET,
)
from speceval.models.google import GoogleRankingModel, GEMINI_1_5_FLASH, GEMINI_2_0_FLASH
from speceval.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Rank pre-generated model outputs using SpecEval and a ranking model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected input directory structure ({generations-dir}):
  IF NOT --batched-generations-mode:
    - {evaluator_model_name}x{candidate_model_A_name}/
      - {statement_id_1}.json
      - {statement_id_2}.json
      ...
    - {evaluator_model_name}x{candidate_model_B_name}/
      - {statement_id_1}.json
      - {statement_id_2}.json
      ...
  IF --batched-generations-mode ({generations-dir} points to provider level, e.g., data/batched_generations/openai/):
    - {model_name_A}/
      - {run_id_1}/results/
        - {statement_id_1}.json
        ...
      - {run_id_2}/results/
        ...
    - {model_name_B}/
      - {run_id_X}/results/
        ...

Output directory structure ({output-dir}):
  - ranker_{ranking_model_name}/
    - {evaluator_model_name}/
      - {candidate_model_A_name}VS{candidate_model_B_name}/
        - {statement_id_1}.json
        - {statement_id_2}.json
        ...
      - {candidate_model_A_name}VS{candidate_model_C_name}/
        ...
""",
    )
    parser.add_argument(
        "--spec-path",
        type=str,
        default="data/specs/openai/jsonl/openai.jsonl",
        help="Path to the specification JSONL file (used for statement context).",
    )
    parser.add_argument(
        "--generations-dir",
        type=str,
        default="data/generations",
        help="Base directory containing pre-generated model outputs (e.g., 'data/generations').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rankings",
        help="Base directory to store ranking results.",
    )
    parser.add_argument(
        "--ranking-org",
        type=str,
        choices=["openai", "anthropic", "google"],
        default="openai",
        help="Organization providing the ranking model (openai or anthropic).",
    )
    parser.add_argument(
        "--ranking-model-name",
        type=str,
        default=None,  # Will be set based on ranking-org if not provided
        help=f"Name of the model to use as the ranker (e.g., {GPT_4_1_MINI} for OpenAI or {CLAUDE_3_5_HAIKU} for Anthropic).",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (alternatively, set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (alternatively, set ANTHROPIC_API_KEY env var).",
    )
    parser.add_argument(
        "--reports-dir",  # Keep this for consistency, might store summary later
        type=str,
        default="data/reports",
        help="Directory to potentially store ranking summary reports.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Generate rankings for all pairs in both directions (A vs B and B vs A).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ranking for statement/model pairs that already have results files.",
    )
    parser.add_argument(
        "--batched-generations-mode",
        action="store_true",
        help="Enable batched generations mode. 'generations-dir' should point to the provider level (e.g., data/batched_generations/openai/).",
    )

    args = parser.parse_args()

    # Configure logging
    logger = setup_logging(args.verbose, folder_name=f"rank_outputs_{args.ranking_org}")

    # Set default model name based on ranking organization if not provided
    if args.ranking_model_name is None:
        if args.ranking_org == "openai":
            args.ranking_model_name = GPT_4_1_MINI
        elif args.ranking_org == "anthropic":  # anthropic
            args.ranking_model_name = CLAUDE_3_5_HAIKU
        elif args.ranking_org == "google":  # google
            args.ranking_model_name = GEMINI_2_0_FLASH
        logger.info(f"Using default {args.ranking_org} model: {args.ranking_model_name}")

    # Load the Specification
    spec_parser = JsonlParser()
    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        logger.error(f"Specification file not found: {spec_path}")
        exit(1)
    try:
        specification = spec_parser.from_file(spec_path)
        logger.info(
            f"Loaded specification with {len(specification.statements)} statements from {args.spec_path}"
        )
    except Exception as e:
        logger.error(f"Error loading specification from {spec_path}: {e}")
        exit(1)

    # Initialize the Ranking Model based on the organization
    try:
        if args.ranking_org == "openai":
            # Handle OpenAI API Key
            api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error(
                    "OpenAI API key is required. Either provide --openai-api-key or set OPENAI_API_KEY env var."
                )
                exit(1)
            ranking_model = OpenAIRankingModel(model_name=args.ranking_model_name, api_key=api_key)
        elif args.ranking_org == "anthropic":  # anthropic
            # Handle Anthropic API Key
            api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error(
                    "Anthropic API key is required. Either provide --anthropic-api-key or set ANTHROPIC_API_KEY env var."
                )
                exit(1)
            ranking_model = AnthropicRankingModel(
                model_name=args.ranking_model_name, api_key=api_key
            )
        elif args.ranking_org == "google":  # google
            # Handle Google API Key
            logger.info("Using Google, we don't need an API key here")
            ranking_model = GoogleRankingModel(model_name=args.ranking_model_name)
        logger.info(f"Using Ranking Model: {ranking_model.get_info()}")
    except ValueError as e:  # Catches API key error from model base classes
        logger.error(f"Failed to initialize ranking model ({args.ranking_model_name}): {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred initializing the ranking model: {e}")
        exit(1)

    # Initialize the Ranking Pipeline
    generations_base_dir = Path(args.generations_dir)
    output_dir_base = Path(args.output_dir)
    try:
        pipeline = RankingPipeline(
            specification=specification,
            generations_dir=str(generations_base_dir),  # Ensure it's a string path
            ranking_model=ranking_model,
            output_dir_base=str(output_dir_base),  # Ensure it's a string path
            verbose=args.verbose,
            all_pairs=args.all_pairs,
            skip_existing=args.skip_existing,
            batched_generations_mode=args.batched_generations_mode,
        )
        # The pipeline init will parse the generations_dir and identify evaluator/candidates
        logger.info(f"Identified Evaluator/Provider: {pipeline.evaluator_model_name}")
        logger.info(f"Identified Candidates: {pipeline.candidate_model_names}")
        logger.info(f"Ranking output base directory: {pipeline.rankings_output_dir.absolute()}")
        if args.all_pairs:
            logger.info(
                "Using all-pairs mode: will generate rankings in both directions (A vs B and B vs A)"
            )
        else:
            logger.info(
                "Using unique-pairs mode: will only generate rankings in one direction (A vs B)"
            )
        if args.skip_existing:
            logger.info(
                "Skipping existing rankings: will only process pairs/statements without existing result files"
            )

    except ValueError as e:  # Catches errors from _parse_generation_dir
        logger.error(f"Pipeline initialization failed: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline initialization: {e}")
        exit(1)

    # Run the ranking pipeline
    logger.info(f"Starting ranking pipeline for generations in {generations_base_dir}...")
    try:
        pipeline.run()
        logger.info("Ranking pipeline finished successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline execution: {e}")
        # Consider more specific error handling if needed
        exit(1)

    # --- Optional: Summarization/Reporting ---
    # Add code here later to aggregate results from the output JSON files
    # and potentially save a summary report to args.reports_dir
    logger.info(f"Ranking results saved under: {pipeline.rankings_output_dir.absolute()}")
    ranking_model_name = ranking_model.get_info().get("model_name", "unknown_ranker")
    logger.info(
        f"Output structure: ranker_{ranking_model_name}/{pipeline.evaluator_model_name}/[model_pairs]"
    )

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
