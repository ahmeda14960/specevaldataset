#!/usr/bin/env python3
"""Script to consolidate individual statement YAML files from a directory into a single JSONL file, with an optional check against a reference JSONL file."""
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Set, Optional
import logging
from speceval.utils.logging import setup_logging

# Add the project root to the path so we can potentially import project modules if needed
# (Although not strictly necessary for this specific script, it's good practice)
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def read_jsonl_as_set(filepath: Path, logger: logging.Logger) -> Optional[Set[str]]:
    """Read a JSONL file, parse each line, canonicalize, and return a set of JSON strings."""
    if not filepath.is_file():
        logger.error(f"Error: File not found: {filepath}")
        return None

    json_set = set()
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        # Parse and dump with sort_keys for canonical representation
                        data = json.loads(line)
                        canonical_json = json.dumps(data, sort_keys=True)
                        json_set.add(canonical_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Warning: Invalid JSON on line {i+1} in {filepath.name}")
        return json_set
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None


def main():
    """Consolidate YAML statement files into a JSONL file."""
    parser = argparse.ArgumentParser(
        description="Consolidate individual statement YAML files into a single JSONL file and optionally compare with a reference."
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing the individual statement YAML files."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional path to a reference JSONL file to compare against the output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",  # Set flag to True if present
        help="Enable verbose logging and print differences if the reference comparison fails.",
    )
    args = parser.parse_args()

    # Configure logging
    logger = setup_logging(args.verbose, folder_name="consolidate_yaml_spec")

    input_dir = args.input_dir.resolve()  # Get absolute path
    reference_file_path = args.reference.resolve() if args.reference else None

    if not input_dir.is_dir():
        logger.error(f"Error: Input path '{input_dir}' is not a valid directory.")
        sys.exit(1)

    dir_name = input_dir.name  # Get the name of the input directory (e.g., "openai_model_spec")

    # Define output directory and file path
    output_dir = input_dir / "jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create subdir if it doesn't exist
    output_jsonl_path = output_dir / f"{dir_name}.jsonl"

    statement_count = 0
    logger.info(f"Consolidating YAML files from: {input_dir}")
    logger.info(f"Outputting JSONL to: {output_jsonl_path}")

    # Open the output JSONL file for writing
    with open(output_jsonl_path, "w") as outfile:
        # Iterate through all .yaml files in the input directory
        for yaml_file in sorted(input_dir.glob("*.yaml")):  # Sort for consistent order
            try:
                with open(yaml_file, "r") as infile:
                    # Load the statement data from the YAML file
                    statement_data = yaml.safe_load(infile)

                    if statement_data:  # Ensure the file wasn't empty
                        # Convert the dictionary to a JSON string (compact format)
                        # Using sort_keys here isn't strictly necessary for the output file,
                        # but doesn't hurt. The comparison logic handles order differences.
                        json_line = json.dumps(
                            statement_data, separators=(",", ":"), sort_keys=True
                        )
                        # Write the JSON string as a line in the output file
                        outfile.write(json_line + "\n")
                        statement_count += 1
                    else:
                        logger.warning(
                            f"Warning: Skipping empty or invalid YAML file: {yaml_file.name}"
                        )

            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {yaml_file.name}: {e}")
            except Exception as e:
                logger.error(f"Error processing file {yaml_file.name}: {e}")

    logger.info(f"Successfully consolidated {statement_count} statements into {output_jsonl_path}")

    # --- Comparison Logic ---
    if reference_file_path:
        logger.info(f"\nComparing output file with reference: {reference_file_path}")

        output_set = read_jsonl_as_set(output_jsonl_path, logger)
        reference_set = read_jsonl_as_set(reference_file_path, logger)

        if output_set is not None and reference_set is not None:
            if output_set == reference_set:
                logger.info("Success: Output JSONL matches the reference file.")
            else:
                logger.error("Mismatch: Output JSONL does NOT match the reference file.")

                # Print differences if verbose flag is set
                if args.verbose:
                    logger.debug("\n--- Differences --- ")
                    diff_out_ref = output_set - reference_set
                    diff_ref_out = reference_set - output_set

                    if diff_out_ref:
                        logger.debug(
                            f"Statements in OUTPUT but not in REFERENCE ({len(diff_out_ref)}):"
                        )
                        for item in sorted(list(diff_out_ref)):
                            logger.debug(f"  + {item}")

                    if diff_ref_out:
                        logger.debug(
                            f"Statements in REFERENCE but not in OUTPUT ({len(diff_ref_out)}):"
                        )
                        for item in sorted(list(diff_ref_out)):
                            logger.debug(f"  - {item}")
                    logger.debug("--- End Differences --- \n")

                sys.exit(1)  # Exit with error code on mismatch
        else:
            logger.error("Comparison skipped due to errors reading files.")
            sys.exit(1)  # Exit with error code if files couldn't be read


if __name__ == "__main__":
    main()
