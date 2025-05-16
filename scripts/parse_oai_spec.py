#!/usr/bin/env python3
"""Script to demonstrate using the JsonlParser to parse OpenAI Model Spec statements from JSONL."""
import sys
import yaml  # Added import
import argparse
from pathlib import Path
from speceval.parsers import JsonlParser
from speceval.base import StatementType
from speceval.utils.logging import setup_logging

# Add the project root to the path so we can import speceval
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def main():
    """Parse OpenAI Model Spec JSONL and write each statement to a separate YAML file."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Parse OpenAI Model Spec JSONL and write each statement to a separate YAML file."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/specs/openai_model_spec.jsonl",
        help="Path to the OpenAI Model Spec JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to scr/<jsonl_name>)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    logger = setup_logging(args.verbose, folder_name="parse_oai_spec")

    # Path to the data file relative to project root
    data_path = project_root / args.input_file
    jsonl_name = data_path.stem  # Extract the base name, e.g., "openai_model_spec"

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "scr" / jsonl_name

    # Use JsonlParser to parse the JSONL file
    try:
        specification = JsonlParser.from_file(data_path)
        statements = specification.statements
    except FileNotFoundError:
        logger.error(f"Input file not found: {data_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing JSONL file: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(statements)} statements from {data_path.name}.")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing statements to {output_dir}")

    # Write each statement to its own YAML file
    for statement in statements:
        output_path = output_dir / f"{statement.id}.yaml"  # Use statement ID for filename

        # Convert the current Statement object to a dictionary
        statement_dict = {
            "id": statement.id,
            "text": statement.text,
            "type": statement.type.name,  # Convert enum to string name
            "authority_level": statement.authority_level.name,  # Convert enum to string name
            "section": statement.section,
            "subsection": statement.subsection,
            "related_statements": statement.related_statements,
            "metadata": statement.metadata,  # Keep metadata as is
        }

        # Write the dictionary for the current statement to its YAML file
        try:
            with open(output_path, "w") as outfile:
                yaml.dump(statement_dict, outfile, default_flow_style=False, sort_keys=False)
            if args.verbose:
                logger.debug(f"Wrote statement {statement.id} to {output_path}")
        except Exception as e:
            logger.error(f"Error writing {statement.id} to {output_path}: {e}")

    logger.info(
        f"Successfully wrote {len(statements)} statements to individual YAML files in {output_dir}"
    )

    # --- Additional Analysis Examples (only shown in verbose mode) ---
    if args.verbose:
        # Example: Print all PLATFORM level prohibitions
        platform_prohibitions = specification.get_statements_by_authority("platform")
        platform_prohibitions = [
            s for s in platform_prohibitions if s.type == StatementType.PROHIBITION
        ]

        logger.debug(f"\nFound {len(platform_prohibitions)} PLATFORM level prohibitions:")
        for i, statement in enumerate(platform_prohibitions[:5], 1):
            logger.debug(f"{i}. {statement.id}: {statement.text[:100]}...")

        # Example: Find statements related to 'privacy'
        privacy_statements = [s for s in statements if "privacy" in s.id.lower()]
        logger.debug(f"\nFound {len(privacy_statements)} statements related to 'privacy':")
        for i, statement in enumerate(privacy_statements[:5], 1):
            logger.debug(f"{i}. {statement.id}: {statement.text[:100]}...")

        # Example: Get all statements from a specific section
        section = "Stay in bounds"
        section_statements = specification.get_statements_by_section(section)
        logger.debug(f"\nFound {len(section_statements)} statements in section '{section}'")

        # Example: Analyze the distribution of statement types
        type_counts = {}
        for s in statements:
            type_counts[s.type.name] = type_counts.get(s.type.name, 0) + 1

        logger.debug("\nDistribution of statement types:")
        for type_name, count in type_counts.items():
            logger.debug(f"{type_name}: {count}")


if __name__ == "__main__":
    main()
