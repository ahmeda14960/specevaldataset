#!/usr/bin/env python3
"""Correlation script for binary compliance judgments across judge models.

This script:
1. Reads binary compliance judgement data from the data/judgements directory
2. Groups judgements by evaluator, candidate, and statement (question)
3. Calculates agreement between different judge models using Fleiss' Kappa
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
from statsmodels.stats.inter_rater import fleiss_kappa
from typing import Dict, List, Tuple, Optional, Any


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

            # Extract judge model info
            judge_info = metadata.get("judge_model", {})
            judge_provider = judge_info.get("provider", "")
            judge_model = judge_info.get("model_name", "")
            judge = (
                f"{judge_provider.lower()}-{judge_model}"
                if judge_provider and judge_model
                else judge_model
            )

            # Extract statement ID
            statement_id = metadata.get("statement_id", "")

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


def scan_judgement_files(judgements_base_dir: Path) -> Dict[str, Dict]:
    """
    Scan all judgement files in the directory and organize by evaluator, candidate, and statement.

    Returns:
        A nested dictionary: {(evaluator, candidate, statement): {judge: [judgements]}}
    """
    judgements_by_key = defaultdict(dict)

    # Recursively find all JSON files
    for json_file in judgements_base_dir.glob("**/*.json"):
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


def prepare_data_for_fleiss_kappa(
    judgements_by_judge: Dict[str, List[Dict]]
) -> Tuple[List[str], np.ndarray]:
    """
    Prepare data for Fleiss' Kappa calculation.

    Returns:
        Tuple of (judge_names, data_matrix)
        Where data_matrix is an N x 2 matrix, N is the number of examples
        Each row contains counts of [non-compliant, compliant] for that example
    """
    # Get all judges
    judges = list(judgements_by_judge.keys())

    # Get all unique inputs across all judges
    all_inputs = set()
    for judge_judgements in judgements_by_judge.values():
        for judgement in judge_judgements:
            all_inputs.add(judgement.get("input", ""))

    # Create a mapping of inputs to their position in the matrix
    input_to_idx = {input_str: i for i, input_str in enumerate(all_inputs)}

    # Initialize the data matrix with zeros
    # Shape: (num_inputs, 2) where columns are [non_compliant, compliant]
    data_matrix = np.zeros((len(all_inputs), 2), dtype=int)

    # Fill the data matrix
    for judge, judge_judgements in judgements_by_judge.items():
        for judgement in judge_judgements:
            input_str = judgement.get("input", "")
            if input_str in input_to_idx:
                idx = input_to_idx[input_str]
                # Convert boolean compliant to integer (0 for False, 1 for True)
                is_compliant = int(judgement.get("compliant", False))
                # Increment the appropriate column (0 for non-compliant, 1 for compliant)
                data_matrix[idx, is_compliant] += 1

    return judges, data_matrix


def calculate_fleiss_kappa(data_matrix: np.ndarray) -> float:
    """
    Calculate Fleiss' Kappa from a data matrix.

    Args:
        data_matrix: An N x 2 matrix where each row contains counts of [non-compliant, compliant]

    Returns:
        Fleiss' Kappa value
    """
    try:
        kappa = fleiss_kappa(data_matrix)
        return kappa
    except Exception as e:
        print(f"Error calculating Fleiss' Kappa: {e}")
        return float("nan")


def calculate_pairwise_agreement(
    judgements_by_judge: Dict[str, List[Dict]]
) -> Dict[Tuple[str, str], float]:
    """
    Calculate pairwise agreement between judges.

    Returns:
        Dictionary mapping judge pairs to agreement percentages
    """
    judges = list(judgements_by_judge.keys())
    agreement_by_pair = {}

    # For each pair of judges
    for i, judge1 in enumerate(judges):
        for j, judge2 in enumerate(judges):
            if i >= j:  # Skip duplicate pairs and self-pairs
                continue

            # Get judgements for both judges
            judge1_judgements = judgements_by_judge[judge1]
            judge2_judgements = judgements_by_judge[judge2]

            # Create dictionaries mapping inputs to compliant values
            judge1_dict = {j.get("input", ""): j.get("compliant", False) for j in judge1_judgements}
            judge2_dict = {j.get("input", ""): j.get("compliant", False) for j in judge2_judgements}

            # Find common inputs
            common_inputs = set(judge1_dict.keys()) & set(judge2_dict.keys())

            if not common_inputs:
                agreement_by_pair[(judge1, judge2)] = float("nan")
                continue

            # Count agreements
            agreements = sum(
                1 for input_str in common_inputs if judge1_dict[input_str] == judge2_dict[input_str]
            )

            # Calculate agreement percentage
            agreement_percentage = agreements / len(common_inputs) if common_inputs else 0
            agreement_by_pair[(judge1, judge2)] = agreement_percentage

    return agreement_by_pair


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


def plot_agreement_matrix(
    agreement_matrix: pd.DataFrame, output_path: Path, title: str = "Judge Agreement Matrix"
):
    """Generate a heatmap of agreement between judge models."""
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)  # Mask upper triangle

    # Create the heatmap
    sns.heatmap(
        agreement_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.5,
        mask=mask,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Agreement Score"},
    )

    plt.title(title)
    plt.tight_layout()

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()


def save_results(
    fleiss_results: Dict[Tuple[str, str, str], float],
    pairwise_results: Dict[Tuple[str, str, str], Dict[Tuple[str, str], float]],
    judge_counts: Dict[str, int],
    output_dir: Path,
):
    """Save correlation results to CSV files."""
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert fleiss_results to DataFrame
    fleiss_df = pd.DataFrame(
        [
            {
                "evaluator": evaluator,
                "candidate": candidate,
                "statement": statement,
                "fleiss_kappa": kappa,
            }
            for (evaluator, candidate, statement), kappa in fleiss_results.items()
        ]
    )

    if not fleiss_df.empty:
        # Save fleiss_results
        fleiss_df.to_csv(output_dir / "fleiss_kappa_results.csv", index=False)

        # Create a pivot table for easier analysis
        pivot_df = fleiss_df.pivot_table(
            index=["evaluator", "candidate"],
            columns="statement",
            values="fleiss_kappa",
            aggfunc="first",
        )

        # Add a mean column
        pivot_df["mean_kappa"] = pivot_df.mean(axis=1)

        # Save the pivot table
        pivot_df.to_csv(output_dir / "fleiss_kappa_by_statement.csv")

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
    judgements_by_judge = defaultdict(list)

    for (evaluator, candidate, statement), judges_data in judgements_by_key.items():
        for judge, judgements in judges_data.items():
            all_judges.add(judge)
            judgements_by_judge[judge].extend(judgements)

    # Process judges who have enough judgements in common
    if len(all_judges) < 2:
        print("Not enough judges for global analysis. Exiting global mode.")
        return

    # Create a dictionary mapping inputs to judge decisions
    judge_decisions = defaultdict(dict)
    for judge, judgements in judgements_by_judge.items():
        for judgement in judgements:
            input_str = judgement.get("input", "")
            is_compliant = judgement.get("compliant", False)
            judge_decisions[(judge, input_str)] = is_compliant

    # Get all unique inputs
    all_inputs = set(input_str for (judge, input_str) in judge_decisions.keys())

    # Create data matrix for Fleiss' Kappa calculation
    judge_list = sorted(list(all_judges))
    input_list = sorted(list(all_inputs))

    # Create a mapping of inputs to their position in the matrix
    input_to_idx = {input_str: i for i, input_str in enumerate(input_list)}

    # Initialize the data matrix with zeros - shape: (num_inputs, 2)
    data_matrix = np.zeros((len(input_list), 2), dtype=int)

    # Fill the matrix with decisions from each judge
    for judge in judge_list:
        for input_str in input_list:
            if (judge, input_str) in judge_decisions:
                idx = input_to_idx[input_str]
                is_compliant = int(judge_decisions[(judge, input_str)])
                data_matrix[idx, is_compliant] += 1

    # Calculate Fleiss' Kappa
    try:
        global_kappa = fleiss_kappa(data_matrix)
        print(f"Global Fleiss' Kappa across all items: {global_kappa:.4f}")

        # Save global results
        global_results = pd.DataFrame([{"mode": "global", "fleiss_kappa": global_kappa}])
        global_results.to_csv(global_output_dir / "global_fleiss_kappa.csv", index=False)

        # Calculate and save pairwise agreement
        pairwise_agreement = {}
        pairwise_data = []

        for i, judge1 in enumerate(judge_list):
            for j, judge2 in enumerate(judge_list):
                if i >= j:  # Skip duplicate pairs and self-pairs
                    continue

                # Find common inputs between the two judges
                common_inputs = set(
                    input_str
                    for input_str in input_list
                    if (judge1, input_str) in judge_decisions
                    and (judge2, input_str) in judge_decisions
                )

                if not common_inputs:
                    pairwise_agreement[(judge1, judge2)] = float("nan")
                    continue

                # Count agreements
                agreements = sum(
                    1
                    for input_str in common_inputs
                    if judge_decisions[(judge1, input_str)] == judge_decisions[(judge2, input_str)]
                )

                # Calculate agreement percentage
                agreement_percentage = agreements / len(common_inputs)
                pairwise_agreement[(judge1, judge2)] = agreement_percentage

                # Add to data for DataFrame
                pairwise_data.append(
                    {
                        "judge1": judge1,
                        "judge2": judge2,
                        "agreement": agreement_percentage,
                        "common_inputs": len(common_inputs),
                    }
                )

        # Save pairwise agreement results
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df.to_csv(global_output_dir / "global_pairwise_agreement.csv", index=False)

        # Create and save agreement matrix
        agreement_matrix = pd.DataFrame(0.0, index=judge_list, columns=judge_list)

        # Fill the diagonal with 1.0 (perfect agreement with self)
        for judge in judge_list:
            agreement_matrix.loc[judge, judge] = 1.0

        # Fill the matrix with pairwise agreement values
        for (judge1, judge2), agreement in pairwise_agreement.items():
            agreement_matrix.loc[judge1, judge2] = agreement
            agreement_matrix.loc[judge2, judge1] = agreement

        # Plot the agreement matrix
        output_path = global_output_dir / "global_agreement_matrix.png"
        plot_agreement_matrix(
            agreement_matrix, output_path, title="Global Judge Agreement Matrix Across All Items"
        )

        # Also save the agreement matrix as CSV
        agreement_matrix.to_csv(global_output_dir / "global_agreement_matrix.csv")

    except Exception as e:
        print(f"Error in global Fleiss' Kappa calculation: {e}")


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

    # Calculate Fleiss' Kappa for each model
    model_kappa_results = {}
    model_pairwise_results = {}

    for model_key, judgements_by_judge in judgements_by_model.items():
        evaluator, candidate = model_key

        # Skip if there's only one judge
        if len(judgements_by_judge) < 2:
            print(f"Skipping model {model_key}: Only {len(judgements_by_judge)} judge(s)")
            continue

        # Extract data for Fleiss' Kappa
        judges, data_matrix = prepare_data_for_fleiss_kappa(judgements_by_judge)

        # Calculate Fleiss' Kappa
        kappa = calculate_fleiss_kappa(data_matrix)
        print(f"Model Fleiss' Kappa for {model_key}: {kappa:.4f}")
        model_kappa_results[model_key] = kappa

        # Calculate pairwise agreement
        pairwise_agreement = calculate_pairwise_agreement(judgements_by_judge)
        model_pairwise_results[model_key] = pairwise_agreement

        # Create agreement matrix and plot it
        if len(judges) > 2:
            # Create a DataFrame for the agreement matrix
            agreement_matrix = pd.DataFrame(0.0, index=judges, columns=judges)

            # Fill the diagonal with 1.0 (perfect agreement with self)
            for judge in judges:
                agreement_matrix.loc[judge, judge] = 1.0

            # Fill the matrix with pairwise agreement values
            for (judge1, judge2), agreement in pairwise_agreement.items():
                agreement_matrix.loc[judge1, judge2] = agreement
                agreement_matrix.loc[judge2, judge1] = agreement

            # Sanitize filename components
            safe_evaluator = sanitize_filename(evaluator)
            safe_candidate = sanitize_filename(candidate)

            # Plot the agreement matrix
            output_path = (
                model_output_dir / f"model_agreement_matrix_{safe_evaluator}_{safe_candidate}.png"
            )
            plot_agreement_matrix(
                agreement_matrix,
                output_path,
                title=f"Model Judge Agreement Matrix\nEvaluator: {evaluator}\nCandidate: {candidate}",
            )

            # Also save the agreement matrix as CSV
            agreement_matrix.to_csv(
                model_output_dir / f"model_agreement_matrix_{safe_evaluator}_{safe_candidate}.csv"
            )

    # Save model kappa results
    model_kappa_df = pd.DataFrame(
        [
            {"evaluator": evaluator, "candidate": candidate, "fleiss_kappa": kappa}
            for (evaluator, candidate), kappa in model_kappa_results.items()
        ]
    )

    if not model_kappa_df.empty:
        model_kappa_df.to_csv(model_output_dir / "model_fleiss_kappa_results.csv", index=False)

        # Create model-based visualizations if there's enough data
        if len(model_kappa_df) > 1:
            # Create a pivot table
            pivot_df = model_kappa_df.pivot_table(
                index="evaluator", columns="candidate", values="fleiss_kappa"
            )

            # Save the pivot table
            pivot_df.to_csv(model_output_dir / "model_fleiss_kappa_pivot.csv")

            # Create a heatmap for models
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                pivot_df,
                annot=True,
                cmap="YlGnBu",
                fmt=".2f",
                linewidths=0.5,
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Fleiss' Kappa"},
            )

            plt.title("Judge Agreement (Fleiss' Kappa) by Model")
            plt.tight_layout()
            plt.savefig(model_output_dir / "model_kappa_heatmap.png", dpi=300)
            plt.close()


def main():
    """
    Run correlation analysis for binary compliance judgements across different judge models.

    Reads judgement data, calculates Fleiss' Kappa for each evaluator-candidate-statement combination,
    generates visualizations, and saves CSV outputs.
    """
    parser = argparse.ArgumentParser(
        description="Analyze correlation between judge models for binary compliance judgements."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="analysis/binary_correlation_config.yaml",
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

    # Get judgements_base_dir from config (default to data/judgements)
    judgements_base_dir = project_dir / config.get("judgements_dir", "data/judgements")

    # Get output_dir from config (default to analysis/binary_correlation)
    output_dir = project_dir / config.get("output_dir", "analysis/binary_correlation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up plot style
    sns.set_theme(style="whitegrid")

    # Scan all judgement files
    print(f"Scanning judgement files in {judgements_base_dir}...")
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

        fleiss_results = {}
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

            # Calculate Fleiss' Kappa
            judges, data_matrix = prepare_data_for_fleiss_kappa(judgements_by_judge)
            kappa = calculate_fleiss_kappa(data_matrix)

            print(f"Fleiss' Kappa for {key}: {kappa:.4f}")
            fleiss_results[key] = kappa

            # Calculate pairwise agreement
            pairwise_agreement = calculate_pairwise_agreement(judgements_by_judge)
            pairwise_results[key] = pairwise_agreement

            # If this combination has more than 2 judges, create an agreement matrix and plot it
            if len(judges) > 2:
                # Create a DataFrame for the agreement matrix
                agreement_matrix = pd.DataFrame(0.0, index=judges, columns=judges)

                # Fill the diagonal with 1.0 (perfect agreement with self)
                for judge in judges:
                    agreement_matrix.loc[judge, judge] = 1.0

                # Fill the matrix with pairwise agreement values
                for (judge1, judge2), agreement in pairwise_agreement.items():
                    agreement_matrix.loc[judge1, judge2] = agreement
                    agreement_matrix.loc[judge2, judge1] = agreement

                # Sanitize filename components
                safe_evaluator = sanitize_filename(evaluator)
                safe_candidate = sanitize_filename(candidate)
                safe_statement = sanitize_filename(statement)

                # Plot the agreement matrix with sanitized filenames
                output_path = (
                    default_output_dir
                    / f"agreement_matrix_{safe_evaluator}_{safe_candidate}_{safe_statement}.png"
                )
                plot_agreement_matrix(
                    agreement_matrix,
                    output_path,
                    title=f"Judge Agreement Matrix\nEvaluator: {evaluator}\nCandidate: {candidate}\nStatement: {statement}",
                )

        # Save results for default mode
        save_results(fleiss_results, pairwise_results, judge_counts, default_output_dir)
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
