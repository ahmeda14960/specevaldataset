#!/usr/bin/env python3
"""Script to read a single statement YAML file and print its representation as a Statement object."""
import sys
import yaml
import argparse
from pathlib import Path
from dataclasses import asdict  # For cleaner printing if needed
import json
from speceval.utils.logging import setup_logging
from speceval.base.statement import Statement, StatementType, AuthorityLevel

# Add the project root to the path so we can import project modules
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def main():
    """Read a YAML file and print the corresponding Statement object."""
    parser = argparse.ArgumentParser(
        description="Read a statement YAML file and print the corresponding Statement object."
    )
    parser.add_argument("yaml_file", type=Path, help="Path to the input statement YAML file.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    logger = setup_logging(args.verbose, folder_name="test_extract_yaml")

    yaml_file_path = args.yaml_file.resolve()

    if not yaml_file_path.is_file():
        logger.error(f"Error: Input file not found: '{yaml_file_path}'")
        sys.exit(1)

    logger.info(f"Reading statement from: {yaml_file_path}")

    try:
        with open(yaml_file_path, "r") as infile:
            # Load the statement data from the YAML file
            statement_data = yaml.safe_load(infile)

            if not statement_data or not isinstance(statement_data, dict):
                logger.error(
                    f"Error: YAML file '{yaml_file_path}' is empty or not a valid dictionary."
                )
                sys.exit(1)

            # --- Map string values to Enum types ---
            try:
                statement_type_str = statement_data.get("type")
                if statement_type_str:
                    statement_data["type"] = StatementType(statement_type_str)
                else:
                    raise ValueError("Missing 'type' field in YAML")
            except ValueError as e:
                logger.error(
                    f"Error: Invalid or missing 'type' value in {yaml_file_path.name}: {e}"
                )
                logger.error(f"Valid types are: {[t.value for t in StatementType]}")
                sys.exit(1)

            try:
                authority_level_str = statement_data.get("authority_level")
                if authority_level_str:
                    statement_data["authority_level"] = AuthorityLevel(authority_level_str)
                else:
                    raise ValueError("Missing 'authority_level' field in YAML")
            except ValueError as e:
                logger.error(
                    f"Error: Invalid or missing 'authority_level' value in {yaml_file_path.name}: {e}"
                )
                logger.error(f"Valid levels are: {[a.value for a in AuthorityLevel]}")
                sys.exit(1)

            # --- Create Statement object ---
            # Use dict unpacking, letting the dataclass handle defaults for missing optional fields
            statement = Statement(**statement_data)

            # --- Print the Statement object ---
            logger.info("--- Statement Object ---")
            # Use print for the statement object itself as it's the main output
            print(statement)
            logger.info("----------------------")

            logger.info("--- Statement as Dictionary ---")
            # Use print for the JSON dump as it's the main output
            print(json.dumps(asdict(statement), indent=2, default=str))  # Use default=str for enums
            logger.info("-----------------------------")

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {yaml_file_path.name}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Error: File not found {yaml_file_path.name}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred processing file {yaml_file_path.name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
