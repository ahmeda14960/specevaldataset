#!/usr/bin/env python3
"""Combine binary compliance data from multiple judge files.

This script combines and analyzes binary compliance data from multiple judge files,
creating visualizations and summary statistics for model performance across judges.
"""

import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_judge_from_path(path):
    """Extract judge name from file path."""
    # Assuming path structure: analysis/binary_compliance_analysis/JUDGE_{JUDGE}/overall_compliance_*.csv
    parts = Path(path).parts
    for part in parts:
        if part.startswith("JUDGE_"):
            return part.replace("JUDGE_", "").lower()
    return "unknown"


def combine_compliance_data(config):
    """Combine compliance data from multiple judge CSV files into a single dataframe."""
    output_dir = config.get("output_dir", "analysis/multi_judge_binary")
    csv_paths = config.get("csv_paths", [])

    if not csv_paths:
        print("No CSV paths provided in config file")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store dataframes by judge
    judge_data = {}
    all_models = set()

    # Load each CSV file
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            continue

        # Extract judge name from path
        judge = extract_judge_from_path(csv_path)

        # Load data
        df = pd.read_csv(csv_path)
        df = df.sort_values("compliance_score", ascending=False)

        # Store in dictionary
        judge_data[judge] = df

        # Track all models
        all_models.update(df["model"].tolist())

    # Create combined dataframe with rankings
    combined_data = {}
    for judge, df in judge_data.items():
        # Create a mapping of model to rank
        model_ranks = {model: idx + 1 for idx, model in enumerate(df["model"])}

        # Create a mapping of model to score
        model_scores = dict(zip(df["model"], df["compliance_score"]))

        # Add to combined data
        combined_data[f"{judge}_rank"] = {
            model: model_ranks.get(model, float("nan")) for model in all_models
        }
        combined_data[f"{judge}_score"] = {
            model: model_scores.get(model, float("nan")) for model in all_models
        }

    # Convert to dataframe - convert set to list first
    combined_df = pd.DataFrame(index=list(all_models))
    for col, data in combined_data.items():
        combined_df[col] = pd.Series(data)

    # Reset index to make model a column
    combined_df = combined_df.reset_index().rename(columns={"index": "model"})

    # Sort by average rank (across all judges)
    rank_cols = [col for col in combined_df.columns if col.endswith("_rank")]
    combined_df["avg_rank"] = combined_df[rank_cols].mean(axis=1)
    combined_df = combined_df.sort_values("avg_rank")

    # Save to CSV
    output_csv = os.path.join(output_dir, "combined_compliance_rankings.csv")
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined rankings saved to {output_csv}")

    # Create visualizations
    create_visualizations(combined_df, output_dir, judge_data)

    return combined_df


def create_visualizations(combined_df, output_dir, judge_data):
    """Create visualizations of combined compliance data."""
    # Extract score columns
    score_cols = [col for col in combined_df.columns if col.endswith("_score")]

    # Only create heatmap and average score charts if there are multiple judges
    if len(score_cols) > 1:
        # 1. Create a heatmap of scores
        plt.figure(figsize=(12, 8))
        score_df = combined_df[["model"] + score_cols].set_index("model")

        # Rename columns to just judge names
        score_df.columns = [col.replace("_score", "") for col in score_df.columns]

        # Create heatmap
        sns.heatmap(score_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
        plt.title("Compliance Scores by Judge")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "compliance_scores_heatmap.png"))
        plt.close()

        # 2. Create a bar chart of average compliance scores
        plt.figure(figsize=(12, 8))
        avg_scores = score_df.mean(axis=1).sort_values(ascending=False)
        avg_scores.plot(kind="bar", color="skyblue")
        plt.axhline(y=avg_scores.mean(), color="r", linestyle="-", label="Average")
        plt.title("Average Compliance Score Across Judges")
        plt.ylabel("Compliance Score")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "average_compliance_scores.png"))
        plt.close()

        # 3. Create ranking matrix visualization
        plt.figure(figsize=(10, max(8, len(combined_df) * 0.4)))

        # Create a dataframe with just the model names and their rankings for each judge
        rank_cols = [col for col in combined_df.columns if col.endswith("_rank")]
        rank_df = combined_df[["model"] + rank_cols].set_index("model")

        # Rename columns to just judge names
        rank_df.columns = [col.replace("_rank", "") for col in rank_df.columns]

        # Sort models by average rank
        rank_df = rank_df.loc[combined_df.sort_values("avg_rank")["model"]]

        # Create a side-by-side ranking matrix
        sns.heatmap(
            rank_df,
            annot=True,
            cmap="YlGnBu_r",
            fmt=".0f",
            linewidths=0.5,
            cbar_kws={"label": "Rank (lower is better)"},
        )
        plt.title("Model Rankings by Judge (lower is better)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_rankings_matrix.png"))
        plt.close()

    # 4. Create individual judge bar charts
    for judge, df in judge_data.items():
        plt.figure(figsize=(12, 8))
        df = df.sort_values("compliance_score", ascending=False)
        df.plot(x="model", y="compliance_score", kind="bar", color="skyblue")
        plt.axhline(y=df["compliance_score"].mean(), color="r", linestyle="-", label="Average")
        plt.title(f"Compliance Scores: {judge.upper()}")
        plt.ylabel("Compliance Score")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{judge}_compliance_scores.png"))
        plt.close()


def main():
    """Run the main compliance data combination pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine binary compliance data from multiple judge files"
    )
    parser.add_argument(
        "--config",
        default="analysis/binary_compliance_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    combine_compliance_data(config)


if __name__ == "__main__":
    main()
