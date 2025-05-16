#!/usr/bin/env python3
"""
Example script demonstrating how to run the BatchedGenerationPipeline.

This script takes pre-generated inputs and submits them to a batch-capable
model (currently supporting OpenAI and Anthropic Batch APIs) for asynchronous processing.

It only submits the jobs and monitors their progress until completion or failure.
A separate script/pipeline would be needed to retrieve and process the results.
"""

import os
import argparse
from pathlib import Path

# SpecEval imports
from speceval import JsonlParser

# Import model constants and classes from both providers
from speceval.models.openai import (
    OpenAIBatchedModel,
    GPT_4O_LATEST,
    GPT_4O_0820204,
    GPT_4O_052024,
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
    GPT_4O_MINI,
)
from speceval.models.anthropic import (
    AnthropicBatchedModel,
    CLAUDE_3_7_SONNET,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_5_HAIKU,
)  # Add other Anthropic models if needed
from speceval.models.google import (
    GoogleBatchedModel,
    GEMINI_1_5_FLASH,
    GEMINI_2_0_FLASH,
)
from speceval.pipelines.batched_generation_pipeline import BatchedGenerationPipeline
from speceval.utils.logging import setup_logging

# Define valid model names for each provider
VALID_OPENAI_MODELS = {
    GPT_4O_LATEST,
    GPT_4O_0820204,
    GPT_4O_052024,
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
    GPT_4O_MINI,
}  # Add more as needed
VALID_ANTHROPIC_MODELS = {
    CLAUDE_3_7_SONNET,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_5_HAIKU,
}  # Add more as needed
VALID_GOOGLE_MODELS = {
    GEMINI_1_5_FLASH,
    GEMINI_2_0_FLASH,
}
ALL_VALID_MODELS = VALID_OPENAI_MODELS.union(VALID_ANTHROPIC_MODELS).union(VALID_GOOGLE_MODELS)

ANTHROPIC_MAX_BATCH_SIZE = 100_000  # From Anthropic docs
OPENAI_MAX_BATCH_SIZE = 50_000  # From OpenAI docs


def main():
    """Run the main batched generation pipeline.

    This function sets up and executes the batched generation pipeline for either OpenAI
    or Anthropic models. It handles command line arguments, validates inputs, initializes
    the appropriate model, and runs the pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run Batched Generation Pipeline for OpenAI or Anthropic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Models:
  OpenAI: {", ".join(sorted(VALID_OPENAI_MODELS))}
  Anthropic: {", ".join(sorted(VALID_ANTHROPIC_MODELS))}
  Google: {", ".join(sorted(VALID_GOOGLE_MODELS))}

Example Usage (OpenAI):
  export OPENAI_API_KEY='your_key_here'
  python -m batched_examples.run_batched_generation \\
    --spec-path data/specs/openai/jsonl/openai.jsonl \\
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/gpt-4.1-2025-04-14 \\
    --model-name {GPT_4_1_MINI} \\
    --batch-size 500 \\
    --verbose

Example Usage (Anthropic):
  export ANTHROPIC_API_KEY='your_key_here'
  python -m batched_examples.run_batched_generation \\
    --spec-path data/specs/anthropic/jsonl/anthropic_spec.jsonl \\
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/claude_something \\
    --model-name {CLAUDE_3_5_SONNET} \\
    --batch-size 50000 \\
    --verbose

Example Usage (Google):
  export GOOGLE_API_KEY='your_key_here'
  python -m batched_examples.run_batched_generation \\
    --spec-path data/specs/google/jsonl/google_spec.jsonl \\
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/gemini_something \\
    --model-name {GEMINI_1_5_FLASH} \\
    --batch-size 50000 \\
    --input-bucket my-bucket \\
    --output-bucket my-bucket/output-prefix \\
    --verbose
""",
    )
    parser.add_argument(
        "--spec-path",
        type=str,
        required=True,
        help="Path to the Specification JSONL file (used to map inputs to statements).",
    )
    parser.add_argument(
        "--pregenerated-inputs-dir",
        type=str,
        required=True,
        help="Directory containing pre-generated JSON inputs (filename stem must match statement ID).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,  # Model name is now required
        help="Name of the batch-capable model to use.",
    )
    parser.add_argument(
        "--org",
        type=str,
        choices=["openai", "anthropic", "google"],
        default=None,
        help="Organization/Provider (openai, anthropic, or google). If not provided, inferred from model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help=f"Number of prompts per batch request (default: 500). OpenAI max: {OPENAI_MAX_BATCH_SIZE}, Anthropic max: {ANTHROPIC_MAX_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="data/batched_generations",
        help="Base directory to store batch metadata and intermediate files (default: data/batched_generations).",
    )
    parser.add_argument(
        "--spec-name",
        type=str,
        required=True,
        choices=["openai", "anthropic", "google"],
        help="Name of the specification to test against (openai, anthropic, or google).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature setting for the model. (Optional)",
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        default="gs://levanter-data/model_spec/",
        help="GCS bucket name for batch input JSONL (e.g. 'my-bucket'). Required for google provider.",
    )
    parser.add_argument(
        "--output-bucket",
        type=str,
        default="gs://levanter-data/model_spec/output",
        help="GCS bucket or prefix for batch output (e.g. 'my-bucket/output-prefix'). Required for google provider.",
    )

    args = parser.parse_args()

    # Prepend the spec-name to the output base directory
    output_base = Path(args.output_base_dir) / args.spec_name

    # Configure logging
    logger = setup_logging(args.verbose, folder_name="batched_generation")

    # --- Validate Model and Determine Organization ---
    if args.model_name not in ALL_VALID_MODELS:
        logger.error(f"Unsupported model: {args.model_name}")
        logger.error(f"Supported models are: {ALL_VALID_MODELS}")
        return

    org = args.org
    if not org:
        if args.model_name in VALID_OPENAI_MODELS:
            org = "openai"
        elif args.model_name in VALID_ANTHROPIC_MODELS:
            org = "anthropic"
        elif args.model_name in VALID_GOOGLE_MODELS:
            org = "google"
        else:
            # This case should be caught by the previous check, but belt-and-suspenders
            logger.error(f"Could not determine organization for model: {args.model_name}")
            return
        logger.info(f"Inferred organization '{org}' based on model name '{args.model_name}'.")
    else:
        # Validate provided org matches model name
        if org == "openai" and args.model_name not in VALID_OPENAI_MODELS:
            logger.error(
                f"Model '{args.model_name}' is not a valid OpenAI model, but --org=openai was specified."
            )
            return
        if org == "anthropic" and args.model_name not in VALID_ANTHROPIC_MODELS:
            logger.error(
                f"Model '{args.model_name}' is not a valid Anthropic model, but --org=anthropic was specified."
            )
            return
        if org == "google" and args.model_name not in VALID_GOOGLE_MODELS:
            logger.error(
                f"Model '{args.model_name}' is not a valid Google model, but --org=google was specified."
            )
            return
        logger.info(f"Using specified organization: {org}")

    # --- API Key Resolution ---
    api_key = None
    api_key_env_var = ""
    if org == "openai":
        api_key_env_var = "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env_var)
    elif org == "anthropic":
        api_key_env_var = "ANTHROPIC_API_KEY"
        api_key = os.environ.get(api_key_env_var)
    elif org == "google":
        api_key_env_var = "GOOGLE_API_KEY"
        api_key = os.environ.get(api_key_env_var)

    if not api_key:
        # Google client may work without an explicit API key if configured via ADC
        if org != "google":
            raise ValueError(
                f"{org.capitalize()} API key is required. Please set the {api_key_env_var} environment variable."
            )

    # --- Batch Size Check (Anthropic) ---
    if org == "anthropic" and args.batch_size > ANTHROPIC_MAX_BATCH_SIZE:
        logger.warning(
            f"Requested batch size ({args.batch_size}) exceeds Anthropic's documented limit ({ANTHROPIC_MAX_BATCH_SIZE}). "
            f"The API request may fail."
        )

    # --- Batch Size Check (OpenAI) ---
    if org == "openai" and args.batch_size > OPENAI_MAX_BATCH_SIZE:
        logger.warning(
            f"Requested batch size ({args.batch_size}) exceeds OpenAI's documented limit ({OPENAI_MAX_BATCH_SIZE}). "
            f"The API request may fail."
        )

    # --- Load Specification ---
    parser = JsonlParser()
    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")
    spec = parser.from_file(spec_path)
    logger.info(f"Loaded specification with {len(spec.statements)} statements from {spec_path}")

    # --- Check Inputs Directory ---
    inputs_dir = Path(args.pregenerated_inputs_dir)
    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"Pregenerated inputs directory not found: {inputs_dir}")

    # --- Instantiate Model ---
    logger.info(f"Using {org.capitalize()} model: {args.model_name}")
    batched_model = None
    try:
        if org == "openai":
            batched_model = OpenAIBatchedModel(model_name=args.model_name, api_key=api_key)
        elif org == "anthropic":
            batched_model = AnthropicBatchedModel(model_name=args.model_name, api_key=api_key)
        elif org == "google":
            # Ensure buckets specified
            if not args.input_bucket or not args.output_bucket:
                raise ValueError(
                    "Google provider requires --input-bucket and --output-bucket arguments."
                )
            batched_model = GoogleBatchedModel(
                model_name=args.model_name,
                api_key=api_key,
                input_bucket=args.input_bucket,
                output_bucket=args.output_bucket,
            )
        else:
            raise ValueError(f"Internal error: Unexpected organization '{org}'")

    except Exception as e:
        logger.error(f"Failed to initialize {org.capitalize()}BatchedModel: {e}")
        return  # Exit if model can't be initialized

    # --- Instantiate Pipeline ---
    logger.info("Initializing BatchedGenerationPipeline...")
    pipeline = BatchedGenerationPipeline(
        specification=spec,
        batched_model=batched_model,
        pregenerated_inputs_dir=str(inputs_dir),
        batch_size=args.batch_size,
        output_base_dir=str(output_base),
        verbose=args.verbose,
        temperature=args.temperature,
    )

    # --- Run the Pipeline ---
    logger.info("Starting pipeline run (submission and monitoring)...")
    pipeline.run()
    logger.info("Pipeline run finished. Check logs and output directory for status.")
    logger.info(f"Output data located in subdirectories under: {pipeline.run_output_dir}")
    logger.info(
        "Note: This script only submits and monitors. Run a separate process to retrieve results."
    )


if __name__ == "__main__":
    main()
