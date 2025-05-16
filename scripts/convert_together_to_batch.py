#!/usr/bin/env python3
"""
Convert Together generations format to Claude format.

This script takes data from the 'together_generations' directory and converts
it to a format similar to Claude's with separate JSON files for each statement ID,
following the same structure as seen in the Claude example.
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Together generations to Claude format")
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/convert_config.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()


def ensure_directory(directory):
    """Ensure the directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def convert_together_to_claude_format(config):
    """
    Convert Together generations format to Claude format.

    Args:
        config (dict): Configuration from YAML file
    """
    src_dir = config["src_dir"]
    dst_dir = config["dst_dir"]

    # Get all JSON files recursively within the source directory
    json_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    for json_file in json_files:
        print(f"Processing: {json_file}")
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract metadata and results
        metadata = data.get("metadata", {})
        results = data.get("results", [])

        if not results:
            print(f"No results found in {json_file}")
            continue

        # Calculate the relative path to preserve directory structure
        rel_path = os.path.relpath(os.path.dirname(json_file), src_dir)
        model_id = os.path.basename(os.path.dirname(json_file))

        # Create the timestamp directory name from run_timestamp in metadata
        timestamp_dir = metadata.get("run_timestamp", model_id)

        # Create destination directory path
        dst_model_dir = os.path.join(dst_dir, rel_path)

        # Create metadata directory and file
        metadata_dir = os.path.join(dst_model_dir, "metadata")
        ensure_directory(metadata_dir)
        with open(os.path.join(metadata_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Create results directory
        results_dir = os.path.join(dst_model_dir, "results")
        ensure_directory(results_dir)

        # Group results by statement_id
        grouped_results = defaultdict(list)
        for result in results:
            statement_id = result.get("statement_id")
            if statement_id:
                # Convert to Claude format
                claude_format_result = {
                    "original_index": result.get("original_index", 0),
                    "input_text": result.get("input_text", ""),
                    "output_text": result.get("output_text", ""),
                    "batch_id": f"batch_{timestamp_dir}",  # Generate a batch ID
                }
                grouped_results[statement_id].append(claude_format_result)

        # Write each statement_id to its own file
        for statement_id, results_list in grouped_results.items():
            result_file = os.path.join(results_dir, f"{statement_id}.json")
            with open(result_file, "w") as f:
                json.dump(results_list, f, indent=2)

        print(f"Converted {len(results)} results into {len(grouped_results)} statement files")


def main():
    args = parse_args()

    # Load configuration from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    convert_together_to_claude_format(config)
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
