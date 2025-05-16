#!/usr/bin/env python3
"""
Generates specific LLM-as-a-judge prompts for evaluating statements
on a Likert scale, using a specified generator model.

Reads a specification (JSONL format), iterates through each statement,
and calls a generator LLM (OpenAI, Anthropic, or Google) using a meta-prompt
to create a tailored judge prompt for that statement.

The generated judge prompts are saved to an output directory.
"""

import os
import argparse
import datetime
import logging
from pathlib import Path
from typing import Optional, Type

# --- Model Imports ---
# OpenAI
from speceval.models.openai import (
    OpenAIModel,
    GPT_4O,
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
)

# Anthropic
from speceval.models.anthropic import (
    AnthropicModel,
    CLAUDE_3_7_SONNET,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_5_HAIKU,
)

# Google
from speceval.models.google import (
    GoogleModel,
    GEMINI_2_0_FLASH,
    GEMINI_1_5_FLASH,
)

# --- SpecEval Core Imports ---
from speceval import JsonlParser, Statement
from speceval.base import BaseModel  # Corrected base type hint
from speceval.utils.logging import setup_logging
from speceval.utils.prompts import (
    META_PROMPT_LIKERT_RESPONSE_JUDGE_GENERATOR,
    META_PROMPT_LIKERT_INPUT_JUDGE_GENERATOR,
    format_statement_examples,
)


# --- Model Mappings ---
AVAILABLE_MODELS = {
    "openai": {
        "class": OpenAIModel,
        "models": {GPT_4O, GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO},
        "api_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "class": AnthropicModel,
        "models": {CLAUDE_3_7_SONNET, CLAUDE_3_5_SONNET, CLAUDE_3_5_HAIKU},
        "api_env": "ANTHROPIC_API_KEY",
    },
    "google": {
        "class": GoogleModel,
        "models": {GEMINI_2_0_FLASH, GEMINI_1_5_FLASH},
        "api_env": "GOOGLE_API_KEY",  # Note: Google key often handled differently (gcloud auth)
    },
}


def get_api_key(org: str, args_key: Optional[str], logger: logging.Logger) -> Optional[str]:
    """Retrieves API key from args or environment variable."""
    if args_key:
        return args_key
    env_var = AVAILABLE_MODELS[org]["api_env"]
    key = os.environ.get(env_var)
    if not key and org != "google":  # Google often uses gcloud auth login
        logger.warning(
            f"API key for {org.capitalize()} not found in arguments or environment variable {env_var}. "
            f"Attempting to proceed without explicit key (may fail if not configured)."
        )
    return key


def validate_model(org: str, model_name: str) -> bool:
    """Checks if the model name is valid for the given organization."""
    if org not in AVAILABLE_MODELS:
        return False
    return model_name in AVAILABLE_MODELS[org]["models"]


def create_generator_model(
    org: str, model_name: str, api_key: Optional[str], logger: logging.Logger
) -> BaseModel:
    """Instantiates the correct model class."""
    model_cls: Type[BaseModel] = AVAILABLE_MODELS[org]["class"]
    try:
        # Pass api_key only if it's not None or if the class expects it
        # (GoogleModel handles None api_key internally if auth is configured)
        if api_key or org == "google":
            return model_cls(model_name=model_name, api_key=api_key)
        else:
            # For OpenAI/Anthropic if no key is found, raise error here explicitly
            raise ValueError(
                f"API Key required for {org} but none provided via args or environment."
            )
    except Exception as e:
        logger.error(f"Failed to initialize model {org}/{model_name}: {e}")
        raise


def generate_judge_prompt_for_statement(
    statement: Statement, generator_model: BaseModel, logger: logging.Logger, mode: str
) -> Optional[str]:
    """Uses the generator model to create a Likert judge prompt for the statement."""
    try:
        # Select the meta-prompt template based on mode
        if mode == "response_judge":
            prompt_template = META_PROMPT_LIKERT_RESPONSE_JUDGE_GENERATOR
        else:
            prompt_template = META_PROMPT_LIKERT_INPUT_JUDGE_GENERATOR

        # Format the meta-prompt with statement details
        meta_prompt = prompt_template.format(
            statement_id=statement.id,
            statement_type=statement.type.value,
            statement_authority=statement.authority_level.value,
            statement_section=statement.section,
            statement_subsection=statement.subsection or "N/A",
            statement_text=statement.text,
            statement_examples=format_statement_examples(statement),
        )

        # Generate the judge prompt
        logger.debug(f"Generating judge prompt for statement: {statement.id}")
        generated_prompt = generator_model.generate(prompt=meta_prompt)
        logger.debug(
            f"Raw generated prompt for {statement.id}:\n{generated_prompt[:500]}..."
        )  # Log snippet
        return generated_prompt

    except Exception as e:
        logger.error(f"Failed to generate judge prompt for statement {statement.id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-as-a-judge prompts for Likert scale evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spec-path",
        type=str,
        required=True,
        help="Path to the specification JSONL file.",
    )
    parser.add_argument(
        "--org",
        type=str,
        required=True,
        choices=AVAILABLE_MODELS.keys(),
        help="Organization/provider for the generator model.",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        required=True,
        help="Name of the generator model to use (must be valid for the specified org).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["response_judge", "input_judge"],
        help="Mode of judge prompt generation: 'response_judge' or 'input_judge'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the generated judge prompts (defaults to data/generated_judge_prompts/likert_<mode>).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the generator model (overrides environment variable).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    # Determine output directory based on mode if not provided
    if not args.output_dir:
        args.output_dir = f"data/generated_judge_prompts/likert_{args.mode}"

    # Configure logging
    script_name = Path(__file__).stem
    logger = setup_logging(args.verbose, folder_name=f"meta_prompts/{script_name}")

    # Validate organization and model
    if not validate_model(args.org, args.generator_model):
        valid_models = ", ".join(sorted(list(AVAILABLE_MODELS[args.org]["models"])))
        logger.error(
            f"Invalid model '{args.generator_model}' for organization '{args.org}'. "
            f"Available models: {valid_models}"
        )
        exit(1)
    logger.info(f"Using generator model: {args.org}/{args.generator_model}")

    # Get API key
    api_key = get_api_key(args.org, args.api_key, logger)

    # Initialize generator model
    try:
        generator_model = create_generator_model(args.org, args.generator_model, api_key, logger)
    except Exception:
        logger.error("Exiting due to model initialization failure.")
        exit(1)

    # Load the specification
    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        logger.error(f"Specification file not found: {spec_path}")
        exit(1)

    try:
        parser = JsonlParser()
        spec = parser.from_file(spec_path)
        logger.info(
            f"Loaded specification '{spec_path.name}' with {len(spec.statements)} statements."
        )
    except Exception as e:
        logger.error(f"Failed to load or parse specification file: {e}")
        exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
    except OSError as e:
        logger.error(f"Failed to create output directory '{output_dir}': {e}")
        exit(1)

    # Process each statement
    success_count = 0
    fail_count = 0
    start_time = datetime.datetime.now()
    logger.info("Starting judge prompt generation...")

    for statement in spec.statements:
        logger.info(f"Processing statement: {statement.id}")
        generated_prompt = generate_judge_prompt_for_statement(
            statement, generator_model, logger, args.mode
        )

        if generated_prompt:
            # Sanitize statement ID for filename if needed (e.g., replace slashes)
            safe_filename = statement.id.replace("/", "_").replace("\\", "_")
            output_filename = f"{safe_filename}_likert_judge.txt"
            output_path = output_dir / output_filename
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(generated_prompt)
                logger.info(f"Saved judge prompt to: {output_path}")
                success_count += 1
            except IOError as e:
                logger.error(
                    f"Failed to write judge prompt for {statement.id} to {output_path}: {e}"
                )
                fail_count += 1
        else:
            logger.warning(f"Skipping statement {statement.id} due to generation error.")
            fail_count += 1

    # Log summary
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logger.info("\n--- Generation Summary ---")
    logger.info(f"Total statements processed: {len(spec.statements)}")
    logger.info(f"Successfully generated prompts: {success_count}")
    logger.info(f"Failed generations: {fail_count}")
    logger.info(f"Total time taken: {duration}")
    logger.info(f"Generated prompts saved in: {output_dir.absolute()}")
    logger.info("------------------------")


if __name__ == "__main__":
    main()
