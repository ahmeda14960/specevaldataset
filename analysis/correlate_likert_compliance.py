#!/usr/bin/env python3
"""Correlation script for Likert scale judgments across judge models.

This script:
1. Reads Likert scale judgement data from the data/likert_judgements directory
2. Groups judgements by evaluator, candidate, and statement (question)
3. Calculates agreement between different judge models using Krippendorff's Alpha
4. Generates a correlation matrix and summary statistics
5. Saves the results as CSV files in the analysis directory

Modes:
- default: Compute agreement for each evaluator-candidate-statement combination
- global: Compute agreement across all items (evaluator-candidate-statement combinations)
- model: Compute agreement for each evaluator-candidate combination, averaging over statements
"""

import json
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

# Import Krippendorff's alpha implementation
try:
    import krippendorff
except ImportError:
    print("Please install krippendorff: pip install krippendorff")
    raise


def extract_metadata_from_json(
    file_path: Path,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract evaluator, candidate, judge model information, and statement ID from a judgment JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

            # Extract metadata
            metadata = data.get("metadata", {})

            # Extract evaluator model info
            evaluator_info = metadata.get("evaluator_model", {})
            evaluator_provider = evaluator_info.get("provider", "")
            evaluator_model = evaluator_info.get("model_name", "")
            evaluator = (
                f"{evaluator_provider.lower()}-{evaluator_model}"
                if evaluator_provider and evaluator_model
                else evaluator_model
            )

            # Extract candidate model info
            candidate_info = metadata.get("candidate_model", {})
            candidate_provider = candidate_info.get("provider", "")
            candidate_model = candidate_info.get("model_name", "")
            candidate = (
                f"{candidate_provider.lower()}-{candidate_model}"
                if candidate_provider and candidate_model
                else candidate_model
            )

            # Extract judge model from path components if not in metadata
            judge = None
            # Judge identifier is likely in the directory structure for Likert judgements
            # Path format: data/likert_judgements/JUDGE_provider_model/date/evaluatorxcandidate/...
            try:
                path_parts = file_path.parts
                judge_part_index = [
                    i for i, part in enumerate(path_parts) if part.startswith("JUDGE_")
                ]
                if judge_part_index:
                    judge_dir = path_parts[judge_part_index[0]]
                    # Extract judge from directory name (e.g., "JUDGE_google_gemini-2.0-flash-001")
                    judge_parts = judge_dir.split("_")[1:]  # Remove "JUDGE_" prefix
                    if len(judge_parts) >= 2:
                        provider = judge_parts[0]
                        model = "_".join(judge_parts[1:])  # Handle underscores in model name
                        judge = f"{provider.lower()}-{model}"
            except Exception as e:
                print(f"Error extracting judge from path {file_path}: {e}")

            # Extract statement ID
            statement_id = metadata.get("statement_id", "")
            if not statement_id:
                # Try to get statement ID from filename
                statement_id = file_path.stem

            return evaluator, candidate, judge, statement_id
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        return None, None, None, None


def extract_judgements_from_json(file_path: Path) -> List[Dict[str, Any]]:
    """Extract judgement data from a JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data.get("judgements", [])
    except Exception as e:
        print(f"Error extracting judgements from {file_path}: {e}")
        return []


def scan_judgement_files(
    likert_judgements_base_dir: Path,
) -> Dict[Tuple[str, str, str], Dict[str, List[Dict]]]:
    """
    Scan all Likert judgement files in the directory and organize by evaluator, candidate, and statement.

    Returns:
        A nested dictionary: {(evaluator, candidate, statement): {judge: [judgements]}}
    """
    judgements_by_key = defaultdict(dict)

    # Recursively find all JSON files
    for json_file in likert_judgements_base_dir.glob("**/*.json"):
        # Extract metadata
        evaluator, candidate, judge, statement_id = extract_metadata_from_json(json_file)

        if not all([evaluator, candidate, judge, statement_id]):
            print(f"Skipping {json_file}: Missing required metadata")
            continue

        # Extract judgements
        judgements = extract_judgements_from_json(json_file)

        if not judgements:
            print(f"Skipping {json_file}: No judgements found")
            continue

        # Store judgements by key
        key = (evaluator, candidate, statement_id)
        judgements_by_key[key][judge] = judgements

    return judgements_by_key


def prepare_data_for_krippendorff(
    judgements_by_judge: Dict[str, List[Dict]]
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """
    Prepare data for Krippendorff's Alpha calculation.

    Returns:
        Tuple of (judges, data_by_input)
        Where data_by_input maps each input to a dict mapping judges to their ratings
    """
    # Get all judges
    judges = list(judgements_by_judge.keys())

    # Get all unique inputs across all judges
    all_inputs = set()
    for judge_judgements in judgements_by_judge.values():
        for judgement in judge_judgements:
            all_inputs.add(judgement.get("input", ""))

    # Create a dictionary mapping inputs to judge ratings
    data_by_input = {}
    for input_str in all_inputs:
        data_by_input[input_str] = {}

    # Fill the data dictionary
    for judge, judge_judgements in judgements_by_judge.items():
        for judgement in judge_judgements:
            input_str = judgement.get("input", "")
            if input_str in data_by_input:
                # Get the Likert rating (1-5)
                rating = judgement.get("rating")
                if rating is not None:
                    data_by_input[input_str][judge] = rating

    return judges, data_by_input


def calculate_krippendorff_alpha(
    judges: List[str], data_by_input: Dict[str, Dict[str, int]]
) -> float:
    """
    Calculate Krippendorff's Alpha from prepared data.

    Args:
        judges: List of judge identifiers
        data_by_input: Dictionary mapping inputs to judge ratings

    Returns:
        Krippendorff's Alpha value
    """
    # Convert to format expected by krippendorff.alpha
    # Each row in reliability_data is one coder's ratings
    reliability_data = []

    for judge in judges:
        # Create a list of ratings for this judge
        judge_ratings = []
        for input_str, ratings in data_by_input.items():
            # Use nan for missing values
            judge_ratings.append(ratings.get(judge, np.nan))
        reliability_data.append(judge_ratings)

    try:
        # Calculate Krippendorff's alpha with ordinal metric (for Likert scale)
        alpha = krippendorff.alpha(
            reliability_data=reliability_data, level_of_measurement="ordinal"
        )
        return alpha
    except Exception as e:
        print(f"Error calculating Krippendorff's Alpha: {e}")
        return float("nan")


def calculate_pairwise_correlation(
    judges: List[str], data_by_input: Dict[str, Dict[str, int]]
) -> Dict[Tuple[str, str], float]:
    """
    Calculate pairwise correlation (Pearson's r) between judges.

    Returns:
        Dictionary mapping judge pairs to correlation coefficients
    """
    correlation_by_pair = {}

    # For each pair of judges
    for i, judge1 in enumerate(judges):
        for j, judge2 in enumerate(judges):
            if i >= j:  # Skip duplicate pairs and self-pairs
                continue

            # Collect pairs of ratings
            ratings1 = []
            ratings2 = []

            for input_str, ratings in data_by_input.items():
                if judge1 in ratings and judge2 in ratings:
                    ratings1.append(ratings[judge1])
                    ratings2.append(ratings[judge2])

            # Calculate correlation if we have enough data points
            if len(ratings1) > 2:
                try:
                    correlation = np.corrcoef(ratings1, ratings2)[0, 1]
                    correlation_by_pair[(judge1, judge2)] = correlation
                except Exception as e:
                    print(f"Error calculating correlation for {judge1} and {judge2}: {e}")
                    correlation_by_pair[(judge1, judge2)] = float("nan")
            else:
                correlation_by_pair[(judge1, judge2)] = float("nan")

    return correlation_by_pair


def sanitize_filename(name: str) -> str:
    """Sanitize a string to make it safe for use in filenames."""
    # Replace slashes and other problematic characters
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame, output_path: Path, title: str = "Judge Correlation Matrix"
):
    """Generate a heatmap of correlation between judge models."""
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)  # Mask upper triangle

    # Create the heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        mask=mask,
        vmin=-1,
        vmax=1,
        center=0,
        cbar_kws={"label": "Correlation Coefficient"},
    )

    plt.title(title)
    plt.tight_layout()

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()


def save_results(
    alpha_results: Dict[Tuple[str, str, str], float],
    pairwise_results: Dict[Tuple[str, str, str], Dict[Tuple[str, str], float]],
    judge_counts: Dict[str, int],
    output_dir: Path,
):
    """Save correlation results to CSV files."""
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert alpha_results to DataFrame
    alpha_df = pd.DataFrame(
        [
            {
                "evaluator": evaluator,
                "candidate": candidate,
                "statement": statement,
                "krippendorff_alpha": alpha,
            }
            for (evaluator, candidate, statement), alpha in alpha_results.items()
        ]
    )

    if not alpha_df.empty:
        # Save alpha_results
        alpha_df.to_csv(output_dir / "krippendorff_alpha_results.csv", index=False)

        # Create a pivot table for easier analysis
        pivot_df = alpha_df.pivot_table(
            index=["evaluator", "candidate"],
            columns="statement",
            values="krippendorff_alpha",
            aggfunc="first",
        )

        # Add a mean column
        pivot_df["mean_alpha"] = pivot_df.mean(axis=1)

        # Save the pivot table
        pivot_df.to_csv(output_dir / "krippendorff_alpha_by_statement.csv")

    # Save judge counts
    judge_df = pd.DataFrame(
        [{"judge": judge, "count": count} for judge, count in judge_counts.items()]
    )
    judge_df.to_csv(output_dir / "judge_counts.csv", index=False)


def extract_judgements_by_model(
    judgements_by_key: Dict[Tuple[str, str, str], Dict[str, List[Dict]]]
):
    """
    Reorganize judgements by evaluator-candidate (model) pairs.

    Returns:
        Dictionary mapping (evaluator, candidate) to a dictionary of judges to all their judgements
    """
    judgements_by_model = defaultdict(lambda: defaultdict(list))

    # Reorganize data by grouping all statements for the same model
    for (evaluator, candidate, statement), judgements_by_judge in judgements_by_key.items():
        model_key = (evaluator, candidate)

        # For each judge, collect all judgements across statements
        for judge, judgements in judgements_by_judge.items():
            judgements_by_model[model_key][judge].extend(judgements)

    return judgements_by_model


def process_global_mode(
    judgements_by_key: Dict[Tuple[str, str, str], Dict[str, List[Dict]]], output_dir: Path
):
    """
    Process data in global mode - calculate agreement across all items.

    Args:
        judgements_by_key: Nested dictionary of judgements by evaluator, candidate, statement
        output_dir: Base output directory
    """
    # Create output directory for global mode
    global_output_dir = output_dir / "global"
    global_output_dir.mkdir(parents=True, exist_ok=True)

    print("\nProcessing in GLOBAL mode...")

    # Collect all judges and their judgements across all items
    all_judges = set()
    all_inputs = set()
    judge_ratings = defaultdict(dict)

    # First pass: collect all judges and inputs
    for (evaluator, candidate, statement), judges_data in judgements_by_key.items():
        for judge, judgements in judges_data.items():
            all_judges.add(judge)
            for judgement in judgements:
                input_str = judgement.get("input", "")
                if input_str:
                    # Create a unique identifier for this input across all data
                    global_input_id = f"{evaluator}_{candidate}_{statement}_{input_str}"
                    all_inputs.add(global_input_id)

                    # Store the rating
                    rating = judgement.get("rating")
                    if rating is not None:
                        judge_ratings[judge][global_input_id] = rating

    # Convert to format needed for Krippendorff's alpha
    judge_list = sorted(list(all_judges))
    input_list = sorted(list(all_inputs))
    reliability_data = []

    for judge in judge_list:
        judge_data = []
        for input_id in input_list:
            judge_data.append(judge_ratings[judge].get(input_id, np.nan))
        reliability_data.append(judge_data)

    # Calculate Krippendorff's alpha
    try:
        global_alpha = krippendorff.alpha(
            reliability_data=reliability_data, level_of_measurement="ordinal"
        )
        print(f"Global Krippendorff's Alpha across all items: {global_alpha:.4f}")

        # Save global results
        global_results = pd.DataFrame([{"mode": "global", "krippendorff_alpha": global_alpha}])
        global_results.to_csv(global_output_dir / "global_krippendorff_alpha.csv", index=False)

        # Calculate and save pairwise correlations
        pairwise_data = []

        for i, judge1 in enumerate(judge_list):
            for j, judge2 in enumerate(judge_list):
                if i >= j:  # Skip duplicate pairs and self-pairs
                    continue

                # Collect pairs of ratings where both judges rated the same input
                ratings1 = []
                ratings2 = []

                for input_id in input_list:
                    if input_id in judge_ratings[judge1] and input_id in judge_ratings[judge2]:
                        ratings1.append(judge_ratings[judge1][input_id])
                        ratings2.append(judge_ratings[judge2][input_id])

                # Calculate correlation if we have enough data points
                if len(ratings1) > 2:
                    try:
                        correlation = np.corrcoef(ratings1, ratings2)[0, 1]

                        # Add to data for DataFrame
                        pairwise_data.append(
                            {
                                "judge1": judge1,
                                "judge2": judge2,
                                "correlation": correlation,
                                "common_inputs": len(ratings1),
                            }
                        )
                    except Exception as e:
                        print(f"Error calculating correlation for {judge1} and {judge2}: {e}")

        # Save pairwise correlation results
        if pairwise_data:
            pairwise_df = pd.DataFrame(pairwise_data)
            pairwise_df.to_csv(global_output_dir / "global_pairwise_correlation.csv", index=False)

            # Create correlation matrix
            correlation_matrix = pd.DataFrame(0.0, index=judge_list, columns=judge_list)

            # Fill the diagonal with 1.0 (perfect correlation with self)
            for judge in judge_list:
                correlation_matrix.loc[judge, judge] = 1.0

            # Fill the matrix with pairwise correlations
            for row in pairwise_data:
                judge1 = row["judge1"]
                judge2 = row["judge2"]
                correlation = row["correlation"]
                correlation_matrix.loc[judge1, judge2] = correlation
                correlation_matrix.loc[judge2, judge1] = correlation

            # Plot the correlation matrix
            output_path = global_output_dir / "global_correlation_matrix.png"
            plot_correlation_matrix(
                correlation_matrix,
                output_path,
                title="Global Judge Correlation Matrix Across All Items",
            )

            # Also save the correlation matrix as CSV
            correlation_matrix.to_csv(global_output_dir / "global_correlation_matrix.csv")

    except Exception as e:
        print(f"Error in global Krippendorff's Alpha calculation: {e}")


def process_model_mode(
    judgements_by_key: Dict[Tuple[str, str, str], Dict[str, List[Dict]]], output_dir: Path
):
    """
    Process data in model mode - calculate agreement for each evaluator-candidate combination.

    Args:
        judgements_by_key: Nested dictionary of judgements by evaluator, candidate, statement
        output_dir: Base output directory
    """
    # Create output directory for model mode
    model_output_dir = output_dir / "model"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    print("\nProcessing in MODEL mode...")

    # Reorganize judgements by model (evaluator-candidate pair)
    judgements_by_model = extract_judgements_by_model(judgements_by_key)

    # Calculate Krippendorff's Alpha for each model
    model_alpha_results = {}
    model_pairwise_results = {}

    for model_key, judgements_by_judge in judgements_by_model.items():
        evaluator, candidate = model_key

        # Skip if there's only one judge
        if len(judgements_by_judge) < 2:
            print(f"Skipping model {model_key}: Only {len(judgements_by_judge)} judge(s)")
            continue

        # Prepare data for Krippendorff's Alpha
        judges, data_by_input = prepare_data_for_krippendorff(judgements_by_judge)

        # Calculate Krippendorff's Alpha
        alpha = calculate_krippendorff_alpha(judges, data_by_input)
        print(f"Model Krippendorff's Alpha for {model_key}: {alpha:.4f}")
        model_alpha_results[model_key] = alpha

        # Calculate pairwise correlation
        pairwise_correlation = calculate_pairwise_correlation(judges, data_by_input)
        model_pairwise_results[model_key] = pairwise_correlation

        # Create correlation matrix and plot it
        if len(judges) > 2:
            # Create a DataFrame for the correlation matrix
            correlation_matrix = pd.DataFrame(0.0, index=judges, columns=judges)

            # Fill the diagonal with 1.0 (perfect correlation with self)
            for judge in judges:
                correlation_matrix.loc[judge, judge] = 1.0

            # Fill the matrix with pairwise correlation values
            for (judge1, judge2), correlation in pairwise_correlation.items():
                correlation_matrix.loc[judge1, judge2] = correlation
                correlation_matrix.loc[judge2, judge1] = correlation

            # Sanitize filename components
            safe_evaluator = sanitize_filename(evaluator)
            safe_candidate = sanitize_filename(candidate)

            # Plot the correlation matrix
            output_path = (
                model_output_dir / f"model_correlation_matrix_{safe_evaluator}_{safe_candidate}.png"
            )
            plot_correlation_matrix(
                correlation_matrix,
                output_path,
                title=f"Model Judge Correlation Matrix\nEvaluator: {evaluator}\nCandidate: {candidate}",
            )

            # Also save the correlation matrix as CSV
            correlation_matrix.to_csv(
                model_output_dir / f"model_correlation_matrix_{safe_evaluator}_{safe_candidate}.csv"
            )

    # Save model alpha results
    model_alpha_df = pd.DataFrame(
        [
            {"evaluator": evaluator, "candidate": candidate, "krippendorff_alpha": alpha}
            for (evaluator, candidate), alpha in model_alpha_results.items()
        ]
    )

    if not model_alpha_df.empty:
        model_alpha_df.to_csv(
            model_output_dir / "model_krippendorff_alpha_results.csv", index=False
        )

        # Create model-based visualizations if there's enough data
        if len(model_alpha_df) > 1:
            # Create a pivot table
            pivot_df = model_alpha_df.pivot_table(
                index="evaluator", columns="candidate", values="krippendorff_alpha"
            )

            # Save the pivot table
            pivot_df.to_csv(model_output_dir / "model_krippendorff_alpha_pivot.csv")

            # Create a heatmap for models
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                pivot_df,
                annot=True,
                cmap="YlGnBu",
                fmt=".2f",
                linewidths=0.5,
                vmin=-1,
                vmax=1,
                center=0,
                cbar_kws={"label": "Krippendorff's Alpha"},
            )

            plt.title("Judge Agreement (Krippendorff's Alpha) by Model")
            plt.tight_layout()
            plt.savefig(model_output_dir / "model_alpha_heatmap.png", dpi=300)
            plt.close()


def main():
    """
    Run correlation analysis for Likert scale judgements across different judge models.

    Reads judgement data, calculates Krippendorff's Alpha for each evaluator-candidate-statement combination,
    generates visualizations, and saves CSV outputs.
    """
    parser = argparse.ArgumentParser(
        description="Analyze correlation between judge models for Likert scale judgements."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="analysis/likert_correlation_config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "global", "model", "all"],
        default="default",
        help="Analysis mode: default (per statement), global (all items), model (per model), or all modes",
    )
    args = parser.parse_args()

    # Set paths
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent

    # Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at {config_path}")
        return
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        return
    except Exception as e:
        print(f"Error reading configuration file {config_path}: {e}")
        return

    # Get judgements_base_dir from config (default to data/likert_judgements)
    judgements_base_dir = project_dir / config.get(
        "likert_judgements_dir", "data/likert_judgements"
    )

    # Get output_dir from config (default to analysis/likert_correlation)
    output_dir = project_dir / config.get("output_dir", "analysis/likert_correlation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up plot style
    sns.set_theme(style="whitegrid")

    # Scan all judgement files
    print(f"Scanning Likert judgement files in {judgements_base_dir}...")
    judgements_by_key = scan_judgement_files(judgements_base_dir)

    if not judgements_by_key:
        print("No valid judgements found. Exiting.")
        return

    print(f"Found {len(judgements_by_key)} evaluator-candidate-statement combinations")

    # Process data according to the selected mode
    if args.mode in ["default", "all"]:
        # Default mode - process per statement (original behavior)
        print("\nProcessing in DEFAULT mode...")
        default_output_dir = output_dir / "default"
        default_output_dir.mkdir(parents=True, exist_ok=True)

        alpha_results = {}
        pairwise_results = {}
        judge_counts = defaultdict(int)

        for key, judgements_by_judge in judgements_by_key.items():
            evaluator, candidate, statement = key

            # Skip if there's only one judge
            if len(judgements_by_judge) < 2:
                print(f"Skipping {key}: Only {len(judgements_by_judge)} judge(s)")
                continue

            # Update judge counts
            for judge in judgements_by_judge.keys():
                judge_counts[judge] += 1

            # Calculate Krippendorff's Alpha
            judges, data_by_input = prepare_data_for_krippendorff(judgements_by_judge)
            alpha = calculate_krippendorff_alpha(judges, data_by_input)

            print(f"Krippendorff's Alpha for {key}: {alpha:.4f}")
            alpha_results[key] = alpha

            # Calculate pairwise correlation
            pairwise_correlation = calculate_pairwise_correlation(judges, data_by_input)
            pairwise_results[key] = pairwise_correlation

            # If this combination has more than 2 judges, create a correlation matrix and plot it
            if len(judges) > 2:
                # Create a DataFrame for the correlation matrix
                correlation_matrix = pd.DataFrame(0.0, index=judges, columns=judges)

                # Fill the diagonal with 1.0 (perfect correlation with self)
                for judge in judges:
                    correlation_matrix.loc[judge, judge] = 1.0

                # Fill the matrix with pairwise correlation values
                for (judge1, judge2), correlation in pairwise_correlation.items():
                    correlation_matrix.loc[judge1, judge2] = correlation
                    correlation_matrix.loc[judge2, judge1] = correlation

                # Sanitize filename components
                safe_evaluator = sanitize_filename(evaluator)
                safe_candidate = sanitize_filename(candidate)
                safe_statement = sanitize_filename(statement)

                # Plot the correlation matrix with sanitized filenames
                output_path = (
                    default_output_dir
                    / f"correlation_matrix_{safe_evaluator}_{safe_candidate}_{safe_statement}.png"
                )
                plot_correlation_matrix(
                    correlation_matrix,
                    output_path,
                    title=f"Judge Correlation Matrix\nEvaluator: {evaluator}\nCandidate: {candidate}\nStatement: {statement}",
                )

        # Save results for default mode
        save_results(alpha_results, pairwise_results, judge_counts, default_output_dir)
        print(f"Default mode results saved to {default_output_dir}")

    # Global mode - process across all items
    if args.mode in ["global", "all"]:
        process_global_mode(judgements_by_key, output_dir)

    # Model mode - process by model (evaluator-candidate pair)
    if args.mode in ["model", "all"]:
        process_model_mode(judgements_by_key, output_dir)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
