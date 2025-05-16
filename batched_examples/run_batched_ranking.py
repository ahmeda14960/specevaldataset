"""
Example script for running the BatchedRankingPipeline.

This script takes generation outputs from multiple candidate models and uses a
batch-capable "judge" model (OpenAI or Anthropic) to rank pairs of these outputs
based on their alignment with statements from specification files.

Supports configuration via a YAML file and/or direct CLI arguments.
"""

import os
import argparse
import yaml
from pathlib import Path
import logging  # Import standard logging first

# SpecEval imports
from speceval.pipelines.batched_ranking_pipeline import BatchedRankingPipeline
from speceval.models.openai import OpenAIBatchedModel, GPT_4O_MINI  # Default judge
from speceval.models.anthropic import AnthropicBatchedModel, CLAUDE_3_5_HAIKU  # Default judge
from speceval.models.google import GoogleBatchedModel, GEMINI_1_5_FLASH, GEMINI_2_0_FLASH
from speceval.utils.logging import setup_logging  # Custom setup

# Global logger, will be configured by setup_logging
logger = logging.getLogger(__name__)

# Define valid model names for each provider to assist with dynamic model loading
# These would typically be more extensive in a real scenario
VALID_OPENAI_JUDGES = {GPT_4O_MINI, "gpt-4o-2024-11-20", "gpt-4.1-2025-04-14"}  # Add more as needed
VALID_ANTHROPIC_JUDGES = {
    CLAUDE_3_5_HAIKU,
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20240620",
}
VALID_GOOGLE_JUDGES = {GEMINI_1_5_FLASH, GEMINI_2_0_FLASH}

ANTHROPIC_MAX_BATCH_SIZE = 100_000  # From Anthropic docs
OPENAI_MAX_BATCH_SIZE = 50_000  # From OpenAI docs


def main():
    parser = argparse.ArgumentParser(
        description="Run Batched Ranking Pipeline for model outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Example YAML configuration (config.yaml):
---
spec_files:
  - data/specs/openai/jsonl/openai.jsonl
  - data/specs/anthropic/jsonl/anthropic_spec.jsonl
candidate_generation_dirs:
  - data/batched_generations/claude-3-7-sonnet-20250219/20250506_115521/results # Model A results
  - data/batched_generations/gpt-4o-mini-2024-07-18/20250507_093000/results   # Model B results
ranking_judge_model_name: {GPT_4O_MINI}
ranking_judge_org: openai
output_dir_base: data/batched_rankings
batch_size: 50
verbose: true
# openai_api_key: YOUR_KEY (or set OPENAI_API_KEY env var)
# anthropic_api_key: YOUR_KEY (or set ANTHROPIC_API_KEY env var)
""",
    )

    # Configuration file argument
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to a YAML configuration file. CLI arguments will override YAML values.",
    )

    # Arguments that can be in YAML or CLI
    parser.add_argument(
        "--spec-path",
        type=str,
        help="Path to the specification JSONL file. Required either via CLI or YAML.",
    )
    parser.add_argument(
        "--spec-name",
        type=str,
        choices=["openai", "anthropic", "google"],
        help="Name of the specification organization (openai, anthropic, google). Required either via CLI or YAML.",
    )
    parser.add_argument(
        "--candidate-generation-dirs",
        type=str,
        nargs="+",
        help="List of paths to directories containing candidate model generations. Basename is model name.",
    )
    parser.add_argument(
        "--ranking-judge-model-name",
        type=str,
        help=f"Name of the batch-capable model to use as judge (e.g., {GPT_4O_MINI}, {CLAUDE_3_5_HAIKU}).",
    )
    parser.add_argument(
        "--ranking-judge-org",
        type=str,
        choices=["openai", "anthropic", "google"],
        help="Organization/Provider for the ranking judge model (openai, anthropic, google).",
    )
    parser.add_argument(
        "--output-dir-base",
        type=str,
        default="data/batched_rankings",
        help="Base directory to store batch ranking metadata and results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of ranking prompts per batch request for the judge model (default: 100).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the ranking judge model (default: 0.0). Optional.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key.")
    parser.add_argument("--anthropic-api-key", type=str, help="Anthropic API key.")
    parser.add_argument(
        "--google-api-key", type=str, help="Google API key (or set GOOGLE_API_KEY env var)."
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        help="GCS bucket name or URI for ranking batch input JSONL (google only).",
    )
    parser.add_argument(
        "--output-bucket",
        type=str,
        help="GCS bucket or prefix for ranking batch output (google only).",
    )

    args = parser.parse_args()

    # Load from YAML if provided
    config = {}
    if args.config_file:
        config_file_path = Path(args.config_file)
        if not config_file_path.exists():
            parser.error(f"Configuration file not found: {config_file_path}")
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Single spec path and spec name (required)
    cfg_spec_path = args.spec_path
    if cfg_spec_path is None:
        cfg_spec_path = config.get("spec_path")
    if cfg_spec_path is None:
        parser.error("Argument --spec-path (or 'spec_path' in YAML config) is required.")

    cfg_spec_name = args.spec_name
    if cfg_spec_name is None:
        cfg_spec_name = config.get("spec_name")
    if cfg_spec_name is None:
        parser.error("Argument --spec-name (or 'spec_name' in YAML config) is required.")

    cfg_candidate_dirs = config.get("candidate_generation_dirs", [])
    if args.candidate_generation_dirs is not None:
        cfg_candidate_dirs = args.candidate_generation_dirs
    elif not cfg_candidate_dirs:
        parser.error("Argument --candidate-generation-dirs is required.")

    cfg_judge_model = (
        args.ranking_judge_model_name
        if args.ranking_judge_model_name is not None
        else config.get("ranking_judge_model_name")
    )
    if not cfg_judge_model:
        parser.error("Argument --ranking-judge-model-name is required.")

    cfg_judge_org = (
        args.ranking_judge_org
        if args.ranking_judge_org is not None
        else config.get("ranking_judge_org")
    )
    if not cfg_judge_org:
        parser.error("Argument --ranking-judge-org is required.")

    cfg_output_dir = (
        args.output_dir_base
        if args.output_dir_base != "data/batched_rankings"
        else config.get("output_dir_base", "data/batched_rankings")
    )
    # Prepend spec and judge subdirectories to output base
    output_base = Path(cfg_output_dir) / f"spec_{cfg_spec_name}" / f"judge_{cfg_judge_model}"
    cfg_output_dir = str(output_base)
    cfg_batch_size = args.batch_size if args.batch_size != 100 else config.get("batch_size", 100)
    cfg_temperature = (
        args.temperature if args.temperature != 0.0 else config.get("temperature", 0.0)
    )
    cfg_verbose = args.verbose or config.get("verbose", False)
    cfg_openai_api_key = (
        args.openai_api_key or config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    )
    cfg_anthropic_api_key = (
        args.anthropic_api_key
        or config.get("anthropic_api_key")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    cfg_google_api_key = (
        args.google_api_key or config.get("google_api_key") or os.environ.get("GOOGLE_API_KEY")
    )
    # Resolve GCS buckets for Google ranking (from CLI args or config)
    cfg_input_bucket = (
        args.input_bucket if args.input_bucket is not None else config.get("input_bucket")
    )
    cfg_output_bucket = (
        args.output_bucket if args.output_bucket is not None else config.get("output_bucket")
    )

    # Configure logging (re-assigns the global logger)
    global logger
    logger = setup_logging(verbose=cfg_verbose, folder_name="run_batched_ranking")

    logger.info("Starting Batched Ranking Script...")
    logger.debug(f"Effective Configuration:")
    logger.debug(f"  Spec Path: {cfg_spec_path}")
    logger.debug(f"  Spec Name: {cfg_spec_name}")
    logger.debug(f"  Candidate Dirs: {cfg_candidate_dirs}")
    logger.debug(f"  Judge Model: {cfg_judge_model}")
    logger.debug(f"  Judge Org: {cfg_judge_org}")
    logger.debug(f"  Output Dir Base: {cfg_output_dir}")
    logger.debug(f"  Batch Size: {cfg_batch_size}")
    logger.debug(f"  Temperature: {cfg_temperature}")
    logger.debug(f"  Verbose: {cfg_verbose}")

    # --- Validate Judge Model and API Key ---
    ranking_judge_api_key = None
    batched_judge_model_instance = None

    # --- API Key Resolution ---
    api_key = None
    api_key_env_var = ""
    if cfg_judge_org == "openai":
        api_key_env_var = "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env_var)
    elif cfg_judge_org == "anthropic":
        api_key_env_var = "ANTHROPIC_API_KEY"
        api_key = os.environ.get(api_key_env_var)

    if not api_key and cfg_judge_org != "google":
        raise ValueError(
            f"{cfg_judge_org.capitalize()} API key is required. Please set the {api_key_env_var} environment variable."
        )

    # --- Batch Size Check (Anthropic) ---
    if cfg_judge_org == "anthropic" and args.batch_size > ANTHROPIC_MAX_BATCH_SIZE:
        logger.warning(
            f"Requested batch size ({args.batch_size}) exceeds Anthropic's documented limit ({ANTHROPIC_MAX_BATCH_SIZE}). "
            f"The API request may fail."
        )

    # --- Batch Size Check (OpenAI) ---
    if cfg_judge_org == "openai" and args.batch_size > OPENAI_MAX_BATCH_SIZE:
        logger.warning(
            f"Requested batch size ({args.batch_size}) exceeds OpenAI's documented limit ({OPENAI_MAX_BATCH_SIZE}). "
            f"The API request may fail."
        )

    if cfg_judge_org == "openai":
        if cfg_judge_model not in VALID_OPENAI_JUDGES:
            logger.warning(
                f"Judge model '{cfg_judge_model}' not in known OpenAI judges. Proceeding, but it might fail."
            )
        try:
            batched_judge_model_instance = OpenAIBatchedModel(
                model_name=cfg_judge_model, api_key=api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIBatchedModel for judge: {e}")
            return
    elif cfg_judge_org == "anthropic":
        if cfg_judge_model not in VALID_ANTHROPIC_JUDGES:
            logger.warning(
                f"Judge model '{cfg_judge_model}' not in known Anthropic judges. Proceeding."
            )
        try:
            batched_judge_model_instance = AnthropicBatchedModel(
                model_name=cfg_judge_model, api_key=api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize AnthropicBatchedModel for judge: {e}")
            return
    elif cfg_judge_org == "google":
        if cfg_judge_model not in VALID_GOOGLE_JUDGES:
            logger.warning(
                f"Judge model '{cfg_judge_model}' not in known Google judges. Proceeding."
            )
        # Ensure GCS buckets are provided via args or config
        if not cfg_input_bucket or not cfg_output_bucket:
            raise ValueError(
                "Google ranking requires 'input_bucket' and 'output_bucket' in CLI args or config file."
            )
        try:
            batched_judge_model_instance = GoogleBatchedModel(
                model_name=cfg_judge_model,
                api_key=api_key,
                input_bucket=cfg_input_bucket,
                output_bucket=cfg_output_bucket,
            )
        except Exception as e:
            logger.error(f"Failed to initialize GoogleBatchedModel for judge: {e}")
            return
    else:
        # Should be caught by argparse choices, but good to have a fallback
        logger.error(f"Unsupported ranking judge organization: {cfg_judge_org}")
        return

    if not batched_judge_model_instance:
        logger.error("Failed to instantiate a ranking judge model. Exiting.")
        return

    # --- Instantiate and Run Pipeline ---
    logger.info("Initializing BatchedRankingPipeline...")
    try:
        pipeline = BatchedRankingPipeline(
            spec_files=[cfg_spec_path],
            candidate_generation_dirs=cfg_candidate_dirs,
            ranking_judge_model=batched_judge_model_instance,
            output_dir_base=cfg_output_dir,
            batch_size=cfg_batch_size,
            temperature=cfg_temperature,
            verbose=cfg_verbose,
        )
    except Exception as e:
        logger.error(f"Failed to initialize BatchedRankingPipeline: {e}", exc_info=True)
        return

    logger.info("Starting pipeline run (submission and monitoring)...")
    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"BatchedRankingPipeline run failed: {e}", exc_info=True)
        return

    logger.info("Pipeline run finished. Check logs and output directory for status.")
    # Output location is complex due to subdirectories, pipeline logs will give specifics.


if __name__ == "__main__":
    main()
