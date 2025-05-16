#!/usr/bin/env python3
"""Analysis script for SpecEval framework.

1. Reads judgement data from the data/judgements directory
2. Groups models by common evaluator and judge combinations
3. Compares compliance scores across candidate models
4. Generates visualizations for overall and per-statement compliance
5. Saves the results as PNG files in the analysis directory
"""

import os
import json
import re
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional


def parse_model_combo_dir(dir_name: str) -> Tuple[str, str, str]:
    """Extract evaluator, candidate, and judge models from directory name."""
    # Directory format is: evaluatorxcandidatexjudge
    parts = dir_name.split("x")
    if len(parts) != 3:
        raise ValueError(f"Directory name {dir_name} does not match expected format")
    return parts[0], parts[1], parts[2]


def parse_org_model_dir(dir_name: str) -> Tuple[str, str, Optional[str]]:
    """Extract evaluator, candidate, and optionally judge models from directory name.

    Handles both formats:
    - evaluatorxcandidate (returns evaluator, candidate, None)
    - evaluatorxcandidatexjudge (returns evaluator, candidate, judge)
    """
    parts = dir_name.split("x")
    if len(parts) == 2:
        # Two-part format: evaluatorxcandidate
        return parts[0], parts[1], None
    elif len(parts) == 3:
        # Three-part format: evaluatorxcandidatexjudge
        return parts[0], parts[1], parts[2]
    else:
        raise ValueError(f"Directory name {dir_name} does not match expected format")


def extract_model_info_from_json(
    file_path: Path,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract evaluator, candidate, and judge model information from a judgment JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

            # The model info is nested in the metadata
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

            return evaluator, candidate, judge
    except Exception as e:
        print(f"Error extracting model info from {file_path}: {e}")
        return None, None, None


def extract_judge_from_json(file_path: Path) -> Optional[str]:
    """Extract judge model information from a judgment JSON file."""
    evaluator, candidate, judge = extract_model_info_from_json(file_path)
    return judge


def get_model_groups_from_config(
    judgements_dir: Path, target_judgement_dirs: List[str], judgment_org: Optional[str] = None
) -> Dict[Tuple[str, str], List[str]]:
    """
    Group candidates by evaluator/judge based on target directories.

    Args:
        judgements_dir: The base directory where judgement subdirectories reside.
        target_judgement_dirs: List of specific directory names (evaluatorxcandidatexjudge) to process.
        judgment_org: Optional organization prefix (e.g., "JUDGE_GOOGLE").

    Returns:
        Dictionary mapping (evaluator, judge) pairs to lists of candidate models.
    """
    model_groups = defaultdict(list)

    if judgment_org:
        # Handle the JUDGE_{org} structure
        org_dir = judgements_dir / judgment_org
        if not org_dir.is_dir():
            print(f"Warning: Organization directory not found: {org_dir}")
            return model_groups

        # Auto-discover all directories under the org directory
        for dir_name in os.listdir(org_dir):
            dir_path = org_dir / dir_name
            if not dir_path.is_dir():
                continue

            try:
                # Try to parse the directory name, supporting both formats
                dir_evaluator, candidate, dir_judge = parse_org_model_dir(dir_name)

                # Get first JSON file to extract model information
                json_files = list(dir_path.glob("*.json"))
                if not json_files:
                    print(f"Warning: No JSON files found in {dir_path}, skipping.")
                    continue

                # Extract model information from JSON
                json_evaluator, json_candidate, json_judge = extract_model_info_from_json(
                    json_files[0]
                )

                # Use directory info if available, otherwise use JSON info
                evaluator = dir_evaluator or json_evaluator
                judge = dir_judge or json_judge

                if not evaluator:
                    print(f"Warning: Could not determine evaluator for {dir_path}, skipping.")
                    continue

                if not judge:
                    print(f"Warning: Could not determine judge for {dir_path}, skipping.")
                    continue

                model_groups[(evaluator, judge)].append(candidate)
                print(f"Added {judgment_org} dir: {dir_name} with judge: {judge}")
            except ValueError as e:
                print(f"Skipping directory {dir_name}: {e}")
    else:
        # Handle the traditional structure
        for dir_name in target_judgement_dirs:
            dir_path = judgements_dir / dir_name
            if not dir_path.is_dir():
                print(
                    f"Warning: Specified directory not found or not a directory: {dir_path}, skipping."
                )
                continue
            try:
                evaluator, candidate, judge = parse_model_combo_dir(dir_name)
                model_groups[(evaluator, judge)].append(candidate)
            except ValueError as e:
                print(f"Skipping directory {dir_name}: {e}")

    return model_groups


def load_judgements(
    judgements_dir: Path,
    evaluator: str,
    candidate: str,
    judge: str,
    judgment_org: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """
    Load all judgement data for a specific evaluator-candidate-judge combination.

    Returns:
        Dictionary mapping statement IDs to lists of judgement data
    """
    if judgment_org:
        # Use the JUDGE_{org} directory structure
        # Try both formats - first the two-part format, then the three-part format
        two_part_dir = judgements_dir / judgment_org / f"{evaluator}x{candidate}"
        three_part_dir = judgements_dir / judgment_org / f"{evaluator}x{candidate}x{judge}"

        if two_part_dir.exists():
            combo_dir = two_part_dir
        elif three_part_dir.exists():
            combo_dir = three_part_dir
        else:
            print(
                f"Warning: No data directory found for {evaluator}x{candidate} under {judgment_org}"
            )
            return {}
    else:
        # Use the traditional directory structure
        dir_name = f"{evaluator}x{candidate}x{judge}"
        combo_dir = judgements_dir / dir_name

        if not combo_dir.exists():
            print(f"Warning: No data directory found for {combo_dir}")
            return {}

    judgements_by_statement = {}

    # First, verify that we're using the correct judge/evaluator by checking a file
    json_files = list(combo_dir.glob("*.json"))
    if json_files:
        json_evaluator, json_candidate, json_judge = extract_model_info_from_json(json_files[0])
        if json_judge and json_judge != judge:
            print(
                f"Warning: Judge mismatch for {combo_dir}. Expected '{judge}', found '{json_judge}'"
            )
        if json_evaluator and json_evaluator != evaluator:
            print(
                f"Warning: Evaluator mismatch for {combo_dir}. Expected '{evaluator}', found '{json_evaluator}'"
            )

    # Load all judgement files
    for file_name in os.listdir(combo_dir):
        if file_name.endswith(".json"):
            file_path = combo_dir / file_name
            statement_id = file_name.replace(".json", "")

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                judgements_by_statement[statement_id] = data.get("judgements", [])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return judgements_by_statement


def calculate_compliance_scores(
    judgements_by_statement: Dict[str, List[Dict]]
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate overall compliance score and per-statement compliance scores.

    Returns:
        Tuple of (overall_score, {statement_id: score, ...})
    """
    if not judgements_by_statement:
        return 0.0, {}

    total_compliant = 0
    total_judgements = 0
    per_statement_scores = {}

    for statement_id, judgements in judgements_by_statement.items():
        statement_compliant = sum(1 for j in judgements if j.get("compliant", False))
        statement_total = len(judgements)

        if statement_total > 0:
            per_statement_scores[statement_id] = statement_compliant / statement_total
            total_compliant += statement_compliant
            total_judgements += statement_total

    overall_score = total_compliant / total_judgements if total_judgements > 0 else 0.0
    return overall_score, per_statement_scores


def get_all_statement_ids(
    judgements_dir: Path,
    judgment_org: Optional[str] = None,
    target_judgement_dirs: Optional[List[str]] = None,
) -> Set[str]:
    """Get a set of all statement IDs based on the directory structure."""
    all_statement_ids = set()

    if judgment_org:
        # Scan through the JUDGE_{org} directory
        org_dir = judgements_dir / judgment_org
        if org_dir.is_dir():
            for model_dir in os.listdir(org_dir):
                combo_dir = org_dir / model_dir
                if combo_dir.is_dir():
                    for file_name in os.listdir(combo_dir):
                        if file_name.endswith(".json"):
                            all_statement_ids.add(file_name.replace(".json", ""))
    elif target_judgement_dirs:
        # Scan through the traditional directories
        for dir_name in target_judgement_dirs:
            combo_dir = judgements_dir / dir_name
            if combo_dir.is_dir():
                for file_name in os.listdir(combo_dir):
                    if file_name.endswith(".json"):
                        all_statement_ids.add(file_name.replace(".json", ""))
            else:
                print(
                    f"Warning: Directory {combo_dir} specified in config not found during statement ID scan."
                )

    return all_statement_ids


def clean_model_name(model_name: str) -> str:
    """Clean up model name for display in plots."""
    # Remove provider prefixes
    name = re.sub(r"^[a-zA-Z]+-[a-zA-Z]+[-/]", "", model_name)

    # Shorten common model names
    name = name.replace("-Instruct", "")
    name = name.replace("-instruct", "")
    name = name.replace("-Turbo", "")

    # Limit length
    if len(name) > 25:
        name = name[:22] + "..."

    return name


def plot_overall_compliance(
    group_key: Tuple[str, str],
    scores: Dict[str, float],
    output_dir: Path,
    judgment_org: Optional[str] = None,
):
    """Generate a bar chart of overall compliance scores for all models in a group."""
    evaluator, judge = group_key

    # Get clean model names for the plot
    models = list(scores.keys())
    display_names = [clean_model_name(model) for model in models]
    values = list(scores.values())

    # Sort by score (descending)
    sorted_indices = np.argsort(values)[::-1]
    sorted_names = [display_names[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    # Create the plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=sorted_names, y=sorted_values)

    # Customize the plot
    plt.title(
        f"Overall Compliance Scores\nEvaluator: {clean_model_name(evaluator)} | Judge: {clean_model_name(judge)}"
    )
    plt.xlabel("Model")
    plt.ylabel("Compliance Score")
    plt.ylim(0, 1.0)

    # Add value labels on top of bars
    for i, v in enumerate(sorted_values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    # Rotate x-axis labels if there are many models
    if len(sorted_names) > 3:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save the plot
    evaluator_short = clean_model_name(evaluator)
    judge_short = clean_model_name(judge)

    # Create org-specific subdirectory if needed
    org_output_dir = output_dir
    if judgment_org:
        org_output_dir = output_dir / judgment_org
        org_output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"overall_compliance_{evaluator_short}_{judge_short}.png"
    plt.savefig(org_output_dir / filename, dpi=300)
    plt.close()

    print(f"Saved overall compliance plot to {org_output_dir / filename}")


def plot_per_statement_compliance(
    group_key: Tuple[str, str],
    per_statement_scores: Dict[str, Dict[str, float]],
    all_statement_ids: Set[str],
    output_dir: Path,
    judgment_org: Optional[str] = None,
):
    """Generate a heatmap of compliance scores per statement for all models in a group."""
    evaluator, judge = group_key

    # Prepare data for the heatmap
    models = list(per_statement_scores.keys())

    # Convert to DataFrame
    statement_ids = sorted(all_statement_ids)

    # Create DataFrame with values (fill missing values with NaN)
    data = {}
    for model in models:
        model_scores = per_statement_scores[model]
        model_display_name = clean_model_name(model)
        data[model_display_name] = {
            stmt_id: model_scores.get(stmt_id, np.nan) for stmt_id in statement_ids
        }

    # Create DataFrame with statements as rows and models as columns (flipped from before)
    df = pd.DataFrame(data)

    # Calculate mean compliance per statement to sort rows
    mean_compliance = df.mean(axis=1, skipna=True)
    df["mean_score"] = mean_compliance
    df = df.sort_values("mean_score")
    df = df.drop("mean_score", axis=1)

    # Sort columns (models) by overall compliance (descending)
    model_means = df.mean(skipna=True)
    sorted_columns = model_means.sort_values(ascending=False).index
    df = df[sorted_columns]

    # Create the heatmap
    plt.figure(figsize=(max(10, len(models) * 0.8), max(12, len(statement_ids) * 0.3)))
    _ = sns.heatmap(
        df, annot=False, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Compliance Score"}
    )

    # Customize the plot
    plt.title(
        f"Per-Statement Compliance Scores\nEvaluator: {clean_model_name(evaluator)} | Judge: {clean_model_name(judge)}"
    )
    plt.xlabel("Model")
    plt.ylabel("Statement ID")

    # Rotate x-axis labels if there are many models
    if len(models) > 3:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save the plot
    evaluator_short = clean_model_name(evaluator)
    judge_short = clean_model_name(judge)

    # Create org-specific subdirectory if needed
    org_output_dir = output_dir
    if judgment_org:
        org_output_dir = output_dir / judgment_org
        org_output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"per_statement_compliance_{evaluator_short}_{judge_short}.png"
    plt.savefig(org_output_dir / filename, dpi=300)
    plt.close()

    print(f"Saved per-statement compliance plot to {org_output_dir / filename}")


def plot_statement_variance(
    group_key: Tuple[str, str],
    per_statement_scores: Dict[str, Dict[str, float]],
    all_statement_ids: Set[str],
    output_dir: Path,
    judgment_org: Optional[str] = None,
):
    """Generate a heatmap showing scores for statements with highest variance across models."""
    evaluator, judge = group_key

    # Prepare data for analysis
    models = list(per_statement_scores.keys())

    # Convert to DataFrame
    statement_ids = sorted(all_statement_ids)
    data = {}
    for model in models:
        model_scores = per_statement_scores[model]
        model_display_name = clean_model_name(model)
        data[model_display_name] = {
            stmt_id: model_scores.get(stmt_id, np.nan) for stmt_id in statement_ids
        }

    df = pd.DataFrame(data)

    # Calculate variance for each statement
    statement_variance = df.var(axis=1, skipna=True)

    # Get top 10 statements by variance
    top_10_variance = statement_variance.nlargest(10)

    # Get the data for these statements
    high_variance_data = df.loc[top_10_variance.index]

    # Sort models by overall mean score
    model_means = high_variance_data.mean(axis=0)
    sorted_columns = model_means.sort_values(ascending=False).index
    high_variance_data = high_variance_data[sorted_columns]

    # Create the heatmap
    plt.figure(figsize=(max(10, len(models) * 0.8), 8))
    _ = sns.heatmap(
        high_variance_data,
        annot=True,  # Show the actual scores
        fmt=".2f",  # Format to 2 decimal places
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Compliance Score"},
    )

    # Customize the plot
    plt.title(
        f"Compliance Scores for Top 10 Highest Variance Statements\nEvaluator: {clean_model_name(evaluator)} | Judge: {clean_model_name(judge)}"
    )
    plt.xlabel("Model")
    plt.ylabel("Statement ID")

    # Rotate x-axis labels if there are many models
    if len(models) > 3:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save the plot
    evaluator_short = clean_model_name(evaluator)
    judge_short = clean_model_name(judge)

    # Create org-specific subdirectory if needed
    org_output_dir = output_dir
    if judgment_org:
        org_output_dir = output_dir / judgment_org
        org_output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"statement_variance_{evaluator_short}_{judge_short}.png"
    plt.savefig(org_output_dir / filename, dpi=300)
    plt.close()

    print(f"Saved statement variance plot to {org_output_dir / filename}")


def save_compliance_data(
    group_key: Tuple[str, str],
    overall_scores: Dict[str, float],
    per_statement_scores: Dict[str, Dict[str, float]],
    output_dir: Path,
    judgment_org: Optional[str] = None,
):
    """Save compliance data as CSV files for further analysis."""
    evaluator, judge = group_key
    evaluator_short = clean_model_name(evaluator)
    judge_short = clean_model_name(judge)

    # Create org-specific subdirectory if needed
    org_output_dir = output_dir
    if judgment_org:
        org_output_dir = output_dir / judgment_org
        org_output_dir.mkdir(parents=True, exist_ok=True)

    # Save overall scores
    overall_df = pd.DataFrame(
        {"model": list(overall_scores.keys()), "compliance_score": list(overall_scores.values())}
    )
    overall_df = overall_df.sort_values("compliance_score", ascending=False)
    overall_csv = org_output_dir / f"overall_compliance_{evaluator_short}_{judge_short}.csv"
    overall_df.to_csv(overall_csv, index=False)

    # Save per-statement scores as a matrix
    statement_df = pd.DataFrame(per_statement_scores).T
    statement_csv = org_output_dir / f"per_statement_compliance_{evaluator_short}_{judge_short}.csv"
    statement_df.to_csv(statement_csv)

    print(f"Saved compliance data to CSV files in {org_output_dir}")


def main():
    """Run the SpecEval compliance analysis.

    Set up paths and plotting configurations, read target directories from config,
    group candidate models, load and process judgement data, calculate compliance scores,
    generate visualizations, and save CSV outputs.
    """
    parser = argparse.ArgumentParser(
        description="Analyze SpecEval compliance judgements based on a config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="analysis/analysis_config.yaml",
        help="Path to the YAML configuration file specifying judgement directories.",
    )
    args = parser.parse_args()

    # Set paths
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    judgements_dir = project_dir / "data" / "judgements"
    output_dir = (
        current_dir / "binary_compliance_analysis"
    )  # Output plots/CSVs to the analysis directory
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

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

    # Check for judgment_org in config
    judgment_org = config.get("judgment_org")
    if judgment_org:
        print(f"Using organization directory structure: {judgment_org}")
        target_judgement_dirs = None
    else:
        # Traditional config with explicit directories
        target_judgement_dirs = config.get("judgement_directories")
        if not target_judgement_dirs or not isinstance(target_judgement_dirs, list):
            print(f"Error: 'judgement_directories' key not found or is not a list in {config_path}")
            return
        if not target_judgement_dirs:
            print(
                f"Warning: 'judgement_directories' list is empty in {config_path}. No analysis will be performed."
            )
            return
        print(f"Loaded {len(target_judgement_dirs)} target directories from {config_path}")

    # Set up plot style
    sns.set_theme(style="whitegrid")

    # Get model groups based on config
    model_groups = get_model_groups_from_config(judgements_dir, target_judgement_dirs, judgment_org)
    if not model_groups:
        print("No valid model groups could be formed from the specified directories. Exiting.")
        return
    print(f"Formed {len(model_groups)} evaluator-judge group(s) for analysis.")

    # Get all statement IDs
    all_statement_ids = get_all_statement_ids(judgements_dir, judgment_org, target_judgement_dirs)
    if not all_statement_ids:
        print("Warning: No statement (.json) files found in the specified directories.")
        # We might still proceed if only overall scores are desired, but plots might be empty/fail.
    print(f"Found {len(all_statement_ids)} unique statement IDs across specified directories.")

    # Process each group
    for group_key, candidates in model_groups.items():
        evaluator, judge = group_key
        print(f"\nAnalyzing group: Evaluator={evaluator}, Judge={judge}")
        print(f"Found {len(candidates)} candidate models")

        # Skip groups with only one candidate model
        if len(candidates) < 2:
            print(f"Skipping group with only {len(candidates)} candidate(s)")
            continue

        # Collect data for all models in this group
        overall_scores = {}
        per_statement_scores = {}

        for candidate in candidates:
            print(f"Processing candidate: {candidate}")
            judgements = load_judgements(judgements_dir, evaluator, candidate, judge, judgment_org)

            if not judgements:
                print(f"No judgement data found for {candidate}, skipping")
                continue

            overall_score, statement_scores = calculate_compliance_scores(judgements)
            overall_scores[candidate] = overall_score
            per_statement_scores[candidate] = statement_scores

            print(f"  Overall compliance: {overall_score:.2f}")
            print(f"  Statements with judgements: {len(statement_scores)}")

        # Skip if we don't have enough data
        if len(overall_scores) < 2:
            print("Not enough models with data in this group, skipping")
            continue

        # Generate and save plots
        plot_overall_compliance(group_key, overall_scores, output_dir, judgment_org)
        plot_per_statement_compliance(
            group_key, per_statement_scores, all_statement_ids, output_dir, judgment_org
        )
        plot_statement_variance(
            group_key, per_statement_scores, all_statement_ids, output_dir, judgment_org
        )

        # Save data as CSVs
        save_compliance_data(
            group_key, overall_scores, per_statement_scores, output_dir, judgment_org
        )


if __name__ == "__main__":
    main()
