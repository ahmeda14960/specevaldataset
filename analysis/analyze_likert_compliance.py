#!/usr/bin/env python3
"""Analysis script for SpecEval framework (Likert Scale Judgements).

1. Reads Likert scale judgement data from the data/likert_judgements directory
2. Groups models by common evaluator and judge combinations based on config
3. Compares average Likert scores across candidate models
4. Generates visualizations for overall and per-statement average scores
5. Saves the results as PNG/CSV files in the analysis/likert_compliance_analysis directory
"""

import os
import json
import re
import argparse
import glob
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional


def parse_model_combo_from_path(path_str: str) -> Tuple[str, str]:
    """Extract evaluator and candidate models from the last part of a path."""
    # Path format is expected to be like: date/evaluatorxcandidate
    combo_part = Path(path_str).name
    parts = combo_part.split("x")
    if len(parts) != 2:  # Expect exactly 2 parts now
        raise ValueError(
            f"Directory name part {combo_part} does not match expected format 'evaluatorxcandidate'"
        )
    return parts[0], parts[1]  # Return only evaluator and candidate


def get_model_groups_from_config(
    likert_judgements_base_dir: Path,
    target_judgement_paths: Optional[List[str]] = None,
    judgment_judge_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Group candidate models by common evaluator, based on either.

    1. A list of relative directory paths (traditional approach)
    2. A specified judgment directory to auto-discover (new approach)

    Args:
        likert_judgements_base_dir: The base directory (e.g., 'data/likert_judgements').
        target_judgement_paths: List of specific relative paths (e.g., 'date/evaluatorxcandidate').
        judgment_judge_dir: Optional judge directory string (e.g., "JUDGE_google_gemini-2.0-flash-001").

    Returns:
        Dictionary mapping evaluator (str) to dictionaries mapping candidate models to their full path.
    """
    model_groups = defaultdict(dict)

    if judgment_judge_dir:
        # Handle the JUDGE_provider_model structure with date-based subdirectories
        judge_dir = likert_judgements_base_dir / judgment_judge_dir
        if not judge_dir.is_dir():
            print(f"Warning: Judge directory not found: {judge_dir}")
            return model_groups

        # Find all date subdirectories and model directories beneath them
        date_dirs = [d for d in judge_dir.iterdir() if d.is_dir()]
        if not date_dirs:
            print(f"Warning: No date directories found under {judge_dir}")
            return model_groups

        print(f"Found {len(date_dirs)} date directories under {judge_dir}")

        # Process each date directory
        for date_dir in date_dirs:
            print(f"Processing date directory: {date_dir.name}")

            # Find all evaluatorxcandidate directories under this date
            model_dirs = [d for d in date_dir.iterdir() if d.is_dir()]

            for model_dir in model_dirs:
                try:
                    # Extract evaluator and candidate from directory name
                    evaluator, candidate = parse_model_combo_from_path(model_dir.name)

                    # Store the full path to this model's judgements
                    # If a model appears in multiple dates, use the most recent one
                    if candidate in model_groups[evaluator]:
                        existing_date = model_groups[evaluator][candidate].parent.name
                        if date_dir.name > existing_date:
                            # Replace with more recent data
                            model_groups[evaluator][candidate] = model_dir
                            print(
                                f"  Updated {evaluator}x{candidate} with more recent data from {date_dir.name}"
                            )
                    else:
                        model_groups[evaluator][candidate] = model_dir
                        print(f"  Added {evaluator}x{candidate} from {date_dir.name}")

                except ValueError as e:
                    print(f"  Skipping directory {model_dir}: {e}")

    elif target_judgement_paths:
        # Traditional approach using explicit paths
        for rel_path_str in target_judgement_paths:
            full_path = likert_judgements_base_dir / rel_path_str
            if not full_path.is_dir():
                print(
                    f"Warning: Specified path not found or not a directory: {full_path}, skipping."
                )
                continue
            try:
                evaluator, candidate = parse_model_combo_from_path(rel_path_str)  # Parse 2 parts
                model_groups[evaluator][candidate] = full_path  # Group by evaluator only
            except ValueError as e:
                print(f"Skipping path {rel_path_str}: {e}")

    return model_groups


def load_likert_judgements(
    combo_dir: Path,
) -> Dict[str, List[Dict]]:
    """
    Load all Likert scale judgement data for a specific evaluator-candidate-judge combination path.

    Args:
        combo_dir: The specific directory path containing the JSON judgement files.

    Returns:
        Dictionary mapping statement IDs to lists of judgement data (containing 'rating').
    """
    if not combo_dir.exists() or not combo_dir.is_dir():
        print(f"Warning: Data directory not found or not a directory: {combo_dir}")
        return {}

    judgements_by_statement = {}

    for file_name in os.listdir(combo_dir):
        if file_name.endswith(".json"):
            file_path = combo_dir / file_name
            statement_id = file_name.replace(".json", "")

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                # Assuming judgements are under a 'judgements' key, and each judgement has a 'rating' field.
                judgements_by_statement[statement_id] = data.get("judgements", [])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return judgements_by_statement


def get_all_statement_ids(
    likert_judgements_base_dir: Path,
    target_judgement_paths: Optional[List[str]] = None,
    judgment_judge_dir: Optional[str] = None,
) -> Set[str]:
    """
    Get a set of all statement IDs across specified target paths or from a judge directory.

    Args:
        likert_judgements_base_dir: The base directory (e.g., 'data/likert_judgements').
        target_judgement_paths: List of specific paths to scan for JSON files.
        judgment_judge_dir: Optional judge directory to auto-discover paths from.

    Returns:
        Set of statement IDs (filenames without .json extension).
    """
    all_statement_ids = set()

    if judgment_judge_dir:
        # Judge directory approach
        judge_dir = likert_judgements_base_dir / judgment_judge_dir
        if not judge_dir.is_dir():
            print(f"Warning: Judge directory not found: {judge_dir}")
            return all_statement_ids

        # Find all JSON files in all model directories beneath date directories
        json_files = glob.glob(str(judge_dir / "**" / "*.json"), recursive=True)
        for json_file in json_files:
            file_name = os.path.basename(json_file)
            all_statement_ids.add(file_name.replace(".json", ""))

    elif target_judgement_paths:
        # Traditional approach
        for rel_path_str in target_judgement_paths:
            combo_dir = likert_judgements_base_dir / rel_path_str
            if combo_dir.is_dir():
                for file_name in os.listdir(combo_dir):
                    if file_name.endswith(".json"):
                        all_statement_ids.add(file_name.replace(".json", ""))
            else:
                print(
                    f"Warning: Directory {combo_dir} specified in config not found during statement ID scan."
                )

    return all_statement_ids


def calculate_average_scores(
    judgements_by_statement: Dict[str, List[Dict]]
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate overall average Likert score and per-statement average scores.

    Assumes each judgement dict has a 'rating' key with a numerical score.

    Returns:
        Tuple of (overall_avg_score, {statement_id: avg_score, ...})
    """
    if not judgements_by_statement:
        return 0.0, {}

    total_score_sum = 0
    total_judgements_count = 0
    per_statement_avg_scores = {}

    for statement_id, judgements in judgements_by_statement.items():
        statement_scores = [j.get("rating") for j in judgements if j.get("rating") is not None]
        statement_total = len(statement_scores)

        if statement_total > 0:
            statement_avg = sum(statement_scores) / statement_total
            per_statement_avg_scores[statement_id] = statement_avg
            total_score_sum += sum(statement_scores)
            total_judgements_count += statement_total
        else:
            per_statement_avg_scores[statement_id] = np.nan  # Use NaN if no valid ratings

    overall_avg_score = (
        total_score_sum / total_judgements_count if total_judgements_count > 0 else 0.0
    )
    return overall_avg_score, per_statement_avg_scores


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


def plot_overall_average_score(
    evaluator: str,
    scores: Dict[str, float],
    output_dir: Path,
    judgment_judge_dir: Optional[str] = None,
):
    """Generate a bar chart of overall average Likert scores for all models evaluated by an evaluator."""
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
    plt.title(f"Overall Average Likert Scores Evaluator: {clean_model_name(evaluator)}")
    plt.xlabel("Candidate Model")
    plt.ylabel("Average Likert Score")
    plt.ylim(0, 5.5)  # Assuming Likert scale 1-5

    # Add value labels on top of bars
    for i, v in enumerate(sorted_values):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center")  # Adjusted offset for 1-5 scale

    # Rotate x-axis labels if there are many models
    if len(sorted_names) > 3:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Create judge-specific subdirectory if needed
    if judgment_judge_dir:
        judge_output_dir = output_dir / judgment_judge_dir.replace("JUDGE_", "")
        judge_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        judge_output_dir = output_dir

    # Save the plot
    evaluator_short = clean_model_name(evaluator)
    filename = f"overall_avg_score_{evaluator_short}.png"
    plt.savefig(judge_output_dir / filename, dpi=300)
    plt.close()

    print(f"Saved overall average score plot to {judge_output_dir / filename}")


def plot_per_statement_average_scores(
    evaluator: str,
    per_statement_scores: Dict[str, Dict[str, float]],
    all_statement_ids: Set[str],
    output_dir: Path,
    judgment_judge_dir: Optional[str] = None,
):
    """Generate a heatmap of average Likert scores per statement for all models evaluated by an evaluator."""
    # Prepare data for the heatmap
    models = list(per_statement_scores.keys())

    # Convert to DataFrame
    statement_ids = sorted(list(all_statement_ids))  # Ensure it's a list for indexing

    # Create DataFrame with values (fill missing values with NaN)
    data = {}
    for model in models:
        model_scores = per_statement_scores[model]
        model_display_name = clean_model_name(model)
        # Use .get(stmt_id, np.nan) to handle missing statements for a model
        data[model_display_name] = {
            stmt_id: model_scores.get(stmt_id, np.nan) for stmt_id in statement_ids
        }

    # Create DataFrame with statements as rows and models as columns
    df = pd.DataFrame(data, index=statement_ids)

    # Calculate mean score per statement to sort rows
    mean_score = df.mean(axis=1, skipna=True)
    df["mean_score"] = mean_score
    df = df.sort_values("mean_score", na_position="first")  # Put NaNs first if sorting
    df = df.drop("mean_score", axis=1)

    # Sort columns (models) by overall average score (descending)
    model_means = df.mean(skipna=True)
    sorted_columns = model_means.sort_values(ascending=False).index
    df = df[sorted_columns]

    # Create the heatmap
    plt.figure(figsize=(max(10, len(models) * 0.8), max(12, len(statement_ids) * 0.3)))
    _ = sns.heatmap(
        df,
        annot=False,
        cmap="viridis",
        vmin=1,
        vmax=5,
        cbar_kws={"label": "Average Likert Score"},  # Adjusted range and label
    )

    # Customize the plot
    plt.title(f"Per-Statement Average Likert Scores Evaluator: {clean_model_name(evaluator)}")
    plt.xlabel("Candidate Model")
    plt.ylabel("Statement ID")

    # Rotate x-axis labels if there are many models
    if len(models) > 3:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Create judge-specific subdirectory if needed
    if judgment_judge_dir:
        judge_output_dir = output_dir / judgment_judge_dir.replace("JUDGE_", "")
        judge_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        judge_output_dir = output_dir

    # Save the plot
    evaluator_short = clean_model_name(evaluator)
    filename = f"per_statement_avg_score_{evaluator_short}.png"
    plt.savefig(judge_output_dir / filename, dpi=300)
    plt.close()

    print(f"Saved per-statement average score plot to {judge_output_dir / filename}")


def plot_statement_variance(
    evaluator: str,
    per_statement_scores: Dict[str, Dict[str, float]],
    all_statement_ids: Set[str],
    output_dir: Path,
    judgment_judge_dir: Optional[str] = None,
):
    """Generate a heatmap showing scores for statements with highest variance across models for an evaluator."""
    # Prepare data for analysis
    models = list(per_statement_scores.keys())

    # Convert to DataFrame
    statement_ids = sorted(list(all_statement_ids))
    data = {}
    for model in models:
        model_scores = per_statement_scores[model]
        model_display_name = clean_model_name(model)
        data[model_display_name] = {
            stmt_id: model_scores.get(stmt_id, np.nan) for stmt_id in statement_ids
        }

    df = pd.DataFrame(data, index=statement_ids)

    # Calculate variance for each statement
    statement_variance = df.var(axis=1, skipna=True)

    # Filter out statements where variance calculation failed (e.g., only one model has data)
    statement_variance = statement_variance.dropna()

    if statement_variance.empty:
        print(
            "Could not calculate variance for any statement (perhaps only one model per statement?). Skipping variance plot."
        )
        return

    # Get top N statements by variance (e.g., top 15 or min(15, num_statements))
    n_top = min(15, len(statement_variance))
    if n_top == 0:
        print("No statements with calculable variance found. Skipping variance plot.")
        return

    top_n_variance_statements = statement_variance.nlargest(n_top)

    # Get the data for these statements
    high_variance_data = df.loc[top_n_variance_statements.index]

    # Sort models by overall mean score within this subset
    model_means = high_variance_data.mean(axis=0, skipna=True)
    sorted_columns = model_means.sort_values(ascending=False).index
    high_variance_data = high_variance_data[sorted_columns]

    # Create the heatmap
    plt.figure(
        figsize=(max(10, len(models) * 0.8), max(6, n_top * 0.4))
    )  # Adjust height based on N
    _ = sns.heatmap(
        high_variance_data,
        annot=True,  # Show the actual average scores
        fmt=".2f",  # Format to 2 decimal places
        cmap="viridis",  # Use a different colormap maybe
        vmin=1,
        vmax=5,  # Likert scale 1-5
        cbar_kws={"label": "Average Likert Score"},
    )

    # Customize the plot
    plt.title(
        f"Avg Likert Scores for Top {n_top} Highest Variance Statements Evaluator: {clean_model_name(evaluator)}"
    )
    plt.xlabel("Candidate Model")
    plt.ylabel("Statement ID")

    # Rotate x-axis labels if there are many models
    if len(models) > 3:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Create judge-specific subdirectory if needed
    if judgment_judge_dir:
        judge_output_dir = output_dir / judgment_judge_dir.replace("JUDGE_", "")
        judge_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        judge_output_dir = output_dir

    # Save the plot
    evaluator_short = clean_model_name(evaluator)
    filename = f"statement_variance_avg_score_{evaluator_short}.png"
    plt.savefig(judge_output_dir / filename, dpi=300)
    plt.close()

    print(f"Saved statement variance plot to {judge_output_dir / filename}")


def save_average_score_data(
    evaluator: str,
    overall_scores: Dict[str, float],
    per_statement_scores: Dict[str, Dict[str, float]],
    output_dir: Path,
    judgment_judge_dir: Optional[str] = None,
):
    """Save average Likert score data as CSV files for an evaluator."""
    evaluator_short = clean_model_name(evaluator)

    # Create judge-specific subdirectory if needed
    if judgment_judge_dir:
        judge_output_dir = output_dir / judgment_judge_dir.replace("JUDGE_", "")
        judge_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        judge_output_dir = output_dir

    # Save overall scores
    overall_df = pd.DataFrame(
        {
            "model": list(overall_scores.keys()),
            "average_score": list(overall_scores.values()),
        }
    )
    overall_df = overall_df.sort_values("average_score", ascending=False)
    overall_csv = judge_output_dir / f"overall_avg_score_{evaluator_short}.csv"
    overall_df.to_csv(overall_csv, index=False)

    # Save per-statement scores as a matrix
    # Convert dict of dicts to DataFrame, handling potential missing statements/models
    statement_df = pd.DataFrame.from_dict(per_statement_scores, orient="index")
    statement_df.index.name = "model"
    statement_df = statement_df.T  # Transpose so statements are rows, models are columns
    statement_df.index.name = "statement_id"

    statement_csv = judge_output_dir / f"per_statement_avg_score_{evaluator_short}.csv"
    statement_df.to_csv(statement_csv)

    print(f"Saved average score data to CSV files in {judge_output_dir}")


def main():
    """Run the SpecEval Likert score analysis.

    Set up paths and plotting configurations, read target paths from config,
    group candidate models, load and process judgement data, calculate average scores,
    generate visualizations, and save CSV outputs.
    """
    parser = argparse.ArgumentParser(
        description="Analyze SpecEval Likert scale judgements based on a config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="analysis/likert_analysis_config.yaml",
        help="Path to the YAML configuration file specifying judgement directory paths relative to data/likert_judgements.",
    )
    args = parser.parse_args()

    # Set paths
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    judgements_base_dir = project_dir / "data" / "likert_judgements"
    output_dir = current_dir / "likert_compliance_analysis"
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

    # Check for judgment_judge_dir in config
    judgment_judge_dir = config.get("judgment_judge_dir")

    if judgment_judge_dir:
        print(f"Using judge directory structure: {judgment_judge_dir}")
        target_judgement_paths = None
    else:
        # Traditional config with explicit paths
        target_judgement_paths = config.get("judgement_paths")
        if not target_judgement_paths or not isinstance(target_judgement_paths, list):
            print(f"Error: 'judgement_paths' key not found or is not a list in {config_path}")
            return
        if not target_judgement_paths:
            print(
                f"Warning: 'judgement_paths' list is empty in {config_path}. No analysis will be performed."
            )
            return
        print(f"Loaded {len(target_judgement_paths)} target paths from {config_path}")

    # Set up plot style
    sns.set_theme(style="whitegrid")

    # Get model groups based on config paths or judge directory
    model_groups = get_model_groups_from_config(
        judgements_base_dir, target_judgement_paths, judgment_judge_dir
    )
    if not model_groups:
        print("No valid model groups could be formed from the specified paths. Exiting.")
        return
    print(f"Formed {len(model_groups)} evaluator group(s) for analysis.")

    # Get all statement IDs
    all_statement_ids = get_all_statement_ids(
        judgements_base_dir, target_judgement_paths, judgment_judge_dir
    )
    if not all_statement_ids:
        print("Warning: No statement (.json) files found in the specified paths.")
        # Analysis might still proceed for overall scores if data exists.
    print(f"Found {len(all_statement_ids)} unique statement IDs across specified paths.")

    # Process each group
    for evaluator, candidates_paths in model_groups.items():
        print(f"Analyzing group: Evaluator={evaluator}")
        print(f"Found {len(candidates_paths)} candidate models")

        # Skip groups with only one candidate model (no comparison possible)
        if len(candidates_paths) < 2:
            print(f"Skipping group with only {len(candidates_paths)} candidate(s)")
            continue

        # Collect data for all models in this group
        overall_avg_scores = {}
        per_statement_avg_scores = {}

        for candidate, combo_path in candidates_paths.items():
            print(f"Processing candidate: {candidate} (from {combo_path.relative_to(project_dir)})")
            # Load judgements expects the full path to the specific combo dir
            judgements = load_likert_judgements(combo_path)

            if not judgements:
                print(
                    f"No judgement data found or loaded for {candidate} at {combo_path}, skipping"
                )
                continue

            # Use the new calculation function
            overall_avg, statement_avgs = calculate_average_scores(judgements)
            overall_avg_scores[candidate] = overall_avg
            per_statement_avg_scores[candidate] = statement_avgs

            print(f"  Overall average score: {overall_avg:.2f}")
            print(f"  Statements with ratings: {len(statement_avgs)}")

        # Skip if we don't have enough data across models after loading
        if len(overall_avg_scores) < 2:
            print("Not enough models with data in this group to compare, skipping plots and CSVs.")
            continue

        # Generate and save plots using updated functions
        plot_overall_average_score(evaluator, overall_avg_scores, output_dir, judgment_judge_dir)
        plot_per_statement_average_scores(
            evaluator, per_statement_avg_scores, all_statement_ids, output_dir, judgment_judge_dir
        )
        plot_statement_variance(
            evaluator, per_statement_avg_scores, all_statement_ids, output_dir, judgment_judge_dir
        )

        # Save data as CSVs using updated function
        save_average_score_data(
            evaluator, overall_avg_scores, per_statement_avg_scores, output_dir, judgment_judge_dir
        )


if __name__ == "__main__":
    main()
