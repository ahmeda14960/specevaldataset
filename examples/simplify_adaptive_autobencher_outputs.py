#!/usr/bin/env python3
"""
This module is used to simplify the outputs of the Adaptive AutoBencher.
It contains functions to parse the output files and process them.
"""


import json
import glob
import os
import argparse
from typing import Dict, Any


def parse_adaptive_autobencher_output(file_path: str) -> Dict[str, Any]:
    """
    Parse an adaptive autobencher output file and extract relevant information.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary with simplified data containing statement, scenarios with descriptions,
        questions, and accuracy
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract the statement information
    statement_id = data.get("statement_id", "")
    statement_text = data.get("statement_text", "")

    # Extract scenarios from the trajectory
    scenarios = []
    for scenario in data.get("trajectory", []):
        # Get the high-level description
        description = scenario.get("description", "")

        # Get the accuracy
        accuracy = scenario.get("accuracy", 0.0)

        # Get the questions
        questions = scenario.get("questions", [])

        scenarios.append({"description": description, "accuracy": accuracy, "questions": questions})

    return {"statement_id": statement_id, "statement_text": statement_text, "scenarios": scenarios}


def process_files(input_pattern: str) -> Dict[str, Dict[str, Any]]:
    """
    Process all JSON files matching the input pattern.

    Args:
        input_pattern: Glob pattern for input files

    Returns:
        Dictionary of simplified outputs keyed by statement ID
    """
    results = {}

    # Find all files matching the pattern
    file_paths = glob.glob(input_pattern, recursive=True)
    if not file_paths:
        print(f"No files found matching pattern: {input_pattern}")
        return results

    print(f"Found {len(file_paths)} files to process")

    # Process each file
    for file_path in file_paths:
        try:
            print(f"Processing: {file_path}")
            result = parse_adaptive_autobencher_output(file_path)
            results[result["statement_id"]] = result
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results


def save_simplified_output(results: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Save the simplified outputs to JSON files.

    Args:
        results: Dictionary of results keyed by statement ID
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save individual files for each statement
    for statement_id, data in results.items():
        output_path = os.path.join(output_dir, f"{statement_id}_simplified.json")

        # Format the output like be_creative_inputs.json
        output_data = {
            "metadata": {"statement_id": statement_id, "statement_text": data["statement_text"]},
            "scenarios": [],
        }

        # Add each scenario with its description, accuracy, and questions
        for scenario in data["scenarios"]:
            output_data["scenarios"].append(
                {
                    "description": scenario["description"],
                    "accuracy": scenario["accuracy"],
                    "questions": scenario["questions"],
                }
            )

        # Save to JSON file
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved: {output_path}")

    # Save a combined file with all results
    combined_output_path = os.path.join(output_dir, "all_simplified.json")
    with open(combined_output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved combined results: {combined_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simplify adaptive autobencher outputs")
    parser.add_argument(
        "input_pattern",
        help="Glob pattern for input files (e.g., 'data/adaptive_autobencher/**/*.json')",
    )
    parser.add_argument(
        "--output-dir",
        default="data/simplified_outputs",
        help="Output directory for simplified files",
    )

    args = parser.parse_args()

    # Process all files matching the pattern
    results = process_files(args.input_pattern)

    if results:
        # Save the simplified outputs
        save_simplified_output(results, args.output_dir)
        print(f"Successfully processed {len(results)} statements")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
