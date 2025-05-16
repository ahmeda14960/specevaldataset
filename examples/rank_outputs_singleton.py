#!/usr/bin/env python3
"""
CLI for ranking pre-generated model outputs for a single specification using a YAML config.
Uses the standard RankingPipeline (synchronous mode).
"""

import os
import argparse
import yaml
from pathlib import Path

from speceval import JsonlParser
from speceval.pipelines.ranking_pipeline import RankingPipeline
from speceval.models.openai import OpenAIRankingModel, GPT_4_1_MINI
from speceval.models.anthropic import AnthropicRankingModel, CLAUDE_3_5_HAIKU
from speceval.models.google import GoogleRankingModel, GEMINI_2_0_FLASH
from speceval.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Run ranking pipeline for a single spec via YAML config."
    )
    parser.add_argument(
        "--config-file", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # Load YAML config
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Required config fields
    spec_path = config.get("spec_path")
    # optional list of candidate dirs for batched mode
    candidate_dirs = config.get("candidate_generation_dirs")
    # legacy single base dir mode
    generations_dir = config.get("generations_dir")
    ranking_org = config.get("ranking_org")
    ranking_model_name = config.get("ranking_model_name")
    output_dir_base = config.get("output_dir_base", "data/rankings")
    verbose = config.get("verbose", False)
    all_pairs = config.get("all_pairs", True)
    skip_existing = config.get("skip_existing", False)
    batched_gen = config.get("batched_generations_mode", False)
    openai_api_key = config.get("openai_api_key")
    anthropic_api_key = config.get("anthropic_api_key")

    # Validate required fields
    if not spec_path or not ranking_org:
        print("Configuration must include 'spec_path' and 'ranking_org'")
        exit(1)
    # Ensure one of generations_dir or candidate_dirs is provided
    if not generations_dir and not candidate_dirs:
        print("Configuration must include 'generations_dir' or 'candidate_generation_dirs'")
        exit(1)

    # Setup logging
    logger = setup_logging(verbose, folder_name=f"rank_outputs_{ranking_org}")

    # Resolve default ranking model name
    if ranking_model_name is None:
        if ranking_org == "openai":
            ranking_model_name = GPT_4_1_MINI
        elif ranking_org == "anthropic":
            ranking_model_name = CLAUDE_3_5_HAIKU
        elif ranking_org == "google":
            ranking_model_name = GEMINI_2_0_FLASH
        logger.info(f"Using default {ranking_org} model: {ranking_model_name}")

    # Load specification
    spec_file = Path(spec_path)
    if not spec_file.exists():
        logger.error(f"Specification file not found: {spec_file}")
        exit(1)
    try:
        spec = JsonlParser().from_file(spec_file)
        logger.info(f"Loaded spec '{spec_file.stem}' with {len(spec.statements)} statements.")
    except Exception as e:
        logger.error(f"Failed to load spec: {e}")
        exit(1)

    # Initialize the ranking model
    try:
        if ranking_org == "openai":
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key is required.")
                exit(1)
            ranking_model = OpenAIRankingModel(model_name=ranking_model_name, api_key=api_key)
        elif ranking_org == "anthropic":
            api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("Anthropic API key is required.")
                exit(1)
            ranking_model = AnthropicRankingModel(model_name=ranking_model_name, api_key=api_key)
        else:  # google
            logger.info("Using Google ranking model (no API key needed)")
            ranking_model = GoogleRankingModel(model_name=ranking_model_name)
        logger.info(f"Ranking model info: {ranking_model.get_info()}")
    except Exception as e:
        logger.error(f"Failed to initialize ranking model: {e}")
        exit(1)

    # Prepare output directory base including spec name
    spec_name = spec_file.stem
    output_base = Path(output_dir_base) / ranking_model_name / spec_name

    # Instantiate the pipeline
    try:
        if candidate_dirs:
            # explicit candidate dirs mode (batched-like)
            pipeline = RankingPipeline(
                specification=spec,
                candidate_generation_dirs=candidate_dirs,
                evaluator_model_name=ranking_org,
                ranking_model=ranking_model,
                output_dir_base=str(output_base),
                verbose=verbose,
                all_pairs=all_pairs,
                skip_existing=skip_existing,
                batched_generations_mode=True,
            )
        else:
            # legacy generations_dir mode
            pipeline = RankingPipeline(
                specification=spec,
                generations_dir=generations_dir,
                ranking_model=ranking_model,
                output_dir_base=str(output_base),
                verbose=verbose,
                all_pairs=all_pairs,
                skip_existing=skip_existing,
                batched_generations_mode=batched_gen,
            )
        logger.info(
            f"Initialized pipeline: evaluator='{pipeline.evaluator_model_name}', candidates={pipeline.candidate_model_names}"
        )
        pipeline.run()
        logger.info("Ranking completed successfully.")
    except Exception as e:
        logger.error(f"Ranking pipeline failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
