#!/usr/bin/env python3
"""
Script for generating responses using Google models synchronously.
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from tqdm import tqdm
from speceval import JsonlParser
from speceval.models.google import (
    GoogleModel,
    GEMINI_1_5_FLASH,
    GEMINI_2_0_FLASH,
    GEMINI_1_5_PRO,
    GEMINI_2_5_PRO,
)
from speceval.utils.logging import setup_logging

VALID_GOOGLE_MODELS = {
    GEMINI_1_5_FLASH,
    GEMINI_2_0_FLASH,
    GEMINI_1_5_PRO,
    GEMINI_2_5_PRO,
}


def process_statement_file(
    model: GoogleModel,
    input_file: Path,
    output_dir: Path,
    temperature: Optional[float] = None,
    logger: Any = None,
) -> None:
    """Process a single statement file containing inputs."""
    if logger:
        logger.info(f"Starting to process {input_file.name}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            loaded_json_data = json.load(f)
    except json.JSONDecodeError as e:
        if logger:
            logger.error(f"Error decoding JSON from {input_file.name}: {e}")
        return
    except Exception as e:
        if logger:
            logger.error(f"Error reading {input_file.name}: {e}")
        return

    input_list_for_generation = (
        []
    )  # Stores dicts like {"input_text": "...", "original_index_for_output": N}

    if isinstance(loaded_json_data, dict) and "inputs" in loaded_json_data:
        potential_inputs_list = loaded_json_data.get("inputs")
        if isinstance(potential_inputs_list, list):
            for idx, entry in enumerate(potential_inputs_list):
                if isinstance(entry, str):
                    input_list_for_generation.append(
                        {"input_text": entry, "original_index_for_output": idx}
                    )
                elif isinstance(entry, dict) and "input_text" in entry:
                    # If the entry itself is a dict with "input_text" and possibly "original_index"
                    original_idx = entry.get("original_index", idx)
                    input_list_for_generation.append(
                        {
                            "input_text": entry["input_text"],
                            "original_index_for_output": original_idx,
                        }
                    )
                else:
                    if logger:
                        logger.warning(
                            f"Skipping malformed entry in 'inputs' list of {input_file.name}: {str(entry)[:100]}"
                        )
        else:
            if logger:
                logger.error(f"The 'inputs' key in {input_file.name} does not contain a list.")
            return
    elif isinstance(loaded_json_data, list):
        # Handles case where the file itself is a list of items
        for idx, entry in enumerate(loaded_json_data):
            if isinstance(entry, dict) and "input_text" in entry:
                original_idx = entry.get("original_index", idx)
                input_list_for_generation.append(
                    {"input_text": entry["input_text"], "original_index_for_output": original_idx}
                )
            else:
                if logger:
                    logger.warning(
                        f"Skipping malformed entry in top-level list of {input_file.name}: {str(entry)[:100]}"
                    )
    else:
        if logger:
            logger.error(
                f"Unsupported JSON structure in {input_file.name}. Expected a list of items or a dict with an 'inputs' key."
            )
        return

    if not input_list_for_generation:
        if logger:
            logger.info(f"No valid inputs found to process in {input_file.name}")
        return

    results = []
    for item_data in tqdm(
        input_list_for_generation, desc=f"Processing inputs in {input_file.name}", leave=False
    ):
        input_text = item_data["input_text"]
        current_original_index = item_data["original_index_for_output"]

        try:
            if logger:
                logger.debug(
                    f"Generating for input (original_index: {current_original_index}) from {input_file.name}"
                )

            generation_kwargs = {}
            if temperature is not None:
                # Pass temperature within generation_config
                generation_kwargs["generation_config"] = {"temperature": temperature}

            response_text = model.generate(prompt=input_text, **generation_kwargs)

            results.append(
                {
                    "original_index": current_original_index,
                    "input_text": input_text,
                    "output_text": response_text,
                }
            )
        except Exception as e:
            if logger:
                logger.error(
                    f"Error generating response for input (original_index: {current_original_index}) in {input_file.name}: {e}"
                )
            # Re-raise the exception to be caught by the main loop for immediate failure
            raise

    output_file_path = output_dir / input_file.name
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if logger:
            logger.info(f"Successfully wrote {len(results)} results to {output_file_path}")
    except Exception as e:
        if logger:
            logger.error(f"Error writing results to {output_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using Google models synchronously",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Models:
  Google: {", ".join(sorted(VALID_GOOGLE_MODELS))}

Example Usage:
  export GOOGLE_API_KEY='your_key_here'
  python generate_google_models.py \\
    --spec-path data/specs/openai/jsonl/openai.jsonl \\
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/gpt-4.1-2025-04-14 \\
    --model-name {GEMINI_1_5_FLASH} \\
    --spec-name openai \\
    --verbose
""",
    )

    parser.add_argument(
        "--spec-path",
        type=str,
        required=True,
        help="Path to the Specification JSONL file.",
    )
    parser.add_argument(
        "--pregenerated-inputs-dir",
        type=str,
        required=True,
        help="Directory containing pre-generated JSON inputs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the Google model to use.",
    )
    parser.add_argument(
        "--spec-name",
        type=str,
        required=True,
        choices=["openai", "anthropic", "google"],
        help="Name of the specification to test against.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="data/generations",
        help="Base directory to store generated outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature setting for the model.",
    )

    args = parser.parse_args()

    # Configure logging
    logger = setup_logging(args.verbose, folder_name="google_generation")

    # Validate model
    if args.model_name not in VALID_GOOGLE_MODELS:
        logger.error(f"Unsupported model: {args.model_name}")
        logger.error(f"Supported models are: {VALID_GOOGLE_MODELS}")
        return

    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "No GOOGLE_API_KEY found in environment. Will try to use default credentials."
        )

    # Load specification
    parser = JsonlParser()
    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")
    spec = parser.from_file(spec_path)
    logger.info(f"Loaded specification with {len(spec.statements)} statements from {spec_path}")

    # Check inputs directory
    inputs_dir = Path(args.pregenerated_inputs_dir)
    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"Pregenerated inputs directory not found: {inputs_dir}")

    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = (
        Path(args.output_base_dir) / args.spec_name / args.model_name / timestamp / "results"
    )
    output_base.mkdir(parents=True, exist_ok=True)

    # Initialize model
    try:
        model = GoogleModel(model_name=args.model_name, api_key=api_key)
        logger.info(f"Initialized Google model: {args.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Google model: {e}")
        return

    # Process each statement file
    input_files = list(inputs_dir.glob("*.json"))
    logger.info(f"Found {len(input_files)} statement files to process")

    for input_file in tqdm(input_files, desc="Processing statement files"):
        logger.info(f"Processing statement file: {input_file.name}")
        try:
            process_statement_file(
                model=model,
                input_file=input_file,
                output_dir=output_base,
                temperature=args.temperature,
                logger=logger,
            )
        except Exception as e:
            logger.error(
                f"Critical error during processing of {input_file.name}. Script will now exit. Error: {e}"
            )
            return  # Exit main function, thus stopping the script

    logger.info(f"Generation complete. Outputs saved to: {output_base}")


if __name__ == "__main__":
    main()
