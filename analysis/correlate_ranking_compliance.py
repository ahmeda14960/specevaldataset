#!/usr/bin/env python3
"""Correlation script for ranking compliance judgments across ranker models.

This script:
1. Reads detailed ranking judgement data (JSON files) from the ranking output directory.
2. Groups judgements by evaluator and candidate pair.
3. Calculates agreement between different ranker models using Fleiss' Kappa,
   considering ranks (-1, 0, 1) as categories.
4. Generates a correlation matrix and summary statistics.
5. Saves the results as CSV files.

Modes:
- global: Compute agreement across all items (evaluator-candidate_pair-input combinations).
- model_pair: Compute agreement for each evaluator-candidate_pair combination.
"""

import json
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.inter_rater import fleiss_kappa
from typing import Dict, List, Tuple, Optional


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sanitize_filename(name: str) -> str:
    """Sanitize a string to make it safe for use in filenames."""
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


def parse_pair_dir(dir_name: str) -> Optional[Tuple[str, str]]:
    """Split a directory name of the form 'ModelAVSModelB' into (ModelA, ModelB)."""
    if "VS" not in dir_name:
        return None
    return tuple(dir_name.split("VS", 1))


def scan_ranking_files(
    rankings_base_dir: Path, rankers: List[str], evaluators: List[str], models: List[str]
) -> Dict[Tuple[str, str, str, str], Dict[str, int]]:
    """Scan ranking JSON files and organize by evaluator, model pair, and input.

    Args:
        rankings_base_dir (Path): Base directory containing ranking files
        rankers (List[str]): List of ranker model names
        evaluators (List[str]): List of evaluator model names
        models (List[str]): List of candidate model names

    Returns:
        Dict[Tuple[str, str, str, str], Dict[str, int]]: Rankings organized by key
    """
    rankings_by_key = defaultdict(dict)

    for ranker in rankers:
        for evaluator in evaluators:
            eval_dir = rankings_base_dir / ranker / evaluator
            if not eval_dir.is_dir():
                print(
                    f"Warning: Directory not found for ranker='{ranker}', evaluator='{evaluator}'. Skipping: {eval_dir}"
                )
                continue

            for pair_dir in eval_dir.iterdir():
                if not pair_dir.is_dir():
                    continue

                parsed_pair = parse_pair_dir(pair_dir.name)
                if not parsed_pair:
                    continue

                modelA, modelB = parsed_pair
                # Ensure the models are in the specified list (if provided)
                if models and (modelA not in models or modelB not in models):
                    continue

                # Ensure consistent ordering for the key (alphabetical)
                m1, m2 = sorted((modelA, modelB))

                for json_file in pair_dir.glob("*.json"):
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)

                        # Get rankings list
                        rankings = data.get("rankings", [])
                        if not rankings:
                            continue

                        # Process each ranking entry
                        for ranking in rankings:
                            input_str = ranking.get("input")
                            rank_val = ranking.get("rank")  # Expected: -1, 0, 1

                            if input_str is None or rank_val is None:
                                print(f"Warning: Missing 'input' or 'rank' in {json_file}")
                                continue

                            # Adjust rank based on alphabetical order if necessary
                            # If modelA != m1, it means the original order was reversed relative to alphabetical
                            # So we need to flip the rank (-1 becomes 1, 1 becomes -1, 0 stays 0)
                            adjusted_rank = rank_val if modelA == m1 else -rank_val

                            # Store the rank for this specific input, identified by the consistent key
                            key = (evaluator, m1, m2, input_str)
                            rankings_by_key[key][ranker] = adjusted_rank

                    except json.JSONDecodeError:
                        print(f"Error reading JSON file: {json_file}")
                    except Exception as e:
                        print(f"Error processing file {json_file}: {e}")

    return rankings_by_key


def prepare_data_for_fleiss_kappa(rankings_by_ranker: Dict[str, int]) -> np.ndarray:
    """Prepare data for Fleiss' Kappa calculation for a single item.

    Args:
        rankings_by_ranker (Dict[str, int]): Mapping of ranker to rank (-1, 0, or 1)

    Returns:
        np.ndarray: A 1x3 numpy array representing counts for categories [-1, 0, 1]
    """
    # Categories are ranks: -1, 0, 1. Map them to indices 0, 1, 2.
    category_map = {-1: 0, 0: 1, 1: 2}
    counts = np.zeros(3, dtype=int)

    for ranker, rank_val in rankings_by_ranker.items():
        if rank_val in category_map:
            counts[category_map[rank_val]] += 1
        else:
            print(
                f"Warning: Unexpected rank value '{rank_val}' encountered for ranker {ranker}. Skipping this rating."
            )

    return counts.reshape(1, 3)


def calculate_fleiss_kappa(data_matrix: np.ndarray) -> float:
    """Calculate Fleiss' Kappa from a data matrix.

    Args:
        data_matrix (np.ndarray): An N x 3 matrix where N is items, columns are category counts [-1, 0, 1]

    Returns:
        float: Fleiss' Kappa value, or NaN if calculation fails
    """
    # Ensure there are multiple raters (sum across categories > 1 for at least one row)
    # Ensure there are multiple items
    if data_matrix.shape[0] < 2 or not np.any(np.sum(data_matrix, axis=1) > 1):
        return float("nan")

    # Ensure variance in ratings
    if (
        np.all(data_matrix[:, 0] == np.sum(data_matrix, axis=1))
        or np.all(data_matrix[:, 1] == np.sum(data_matrix, axis=1))
        or np.all(data_matrix[:, 2] == np.sum(data_matrix, axis=1))
    ):
        # print("Skipping Fleiss Kappa: No variance in ratings (all raters agree on every item or only one category used).")
        return float("nan")  # Or potentially 1.0 if all agree, but statsmodels might raise error.

    try:
        kappa = fleiss_kappa(data_matrix, method="fleiss")  # Use 'fleiss' method explicitly
        # Handle potential edge cases where kappa might be slightly outside [-1, 1] due to floating point
        return max(-1.0, min(1.0, kappa))
    except ZeroDivisionError:
        print(
            "Error calculating Fleiss' Kappa: Division by zero. Often happens with perfect agreement or no variance."
        )
        # Check for perfect agreement
        num_raters = np.sum(data_matrix[0, :])  # Assume constant number of raters per item
        if np.all((data_matrix == num_raters) | (data_matrix == 0)):
            # This check is heuristic for perfect agreement across all items
            # print("Perfect agreement detected, returning Kappa = 1.0")
            return 1.0
        return float("nan")
    except Exception as e:
        print(f"Error calculating Fleiss' Kappa: {e}")
        # print("Data matrix causing error:")
        # print(data_matrix)
        return float("nan")


def calculate_pairwise_agreement(
    rankings_by_key: Dict[Tuple[str, str, str, str], Dict[str, int]], rankers: List[str]
) -> Dict[Tuple[str, str], float]:
    """Calculate pairwise agreement percentage between rankers across all items.

    Args:
        rankings_by_key (Dict): Dictionary mapping keys to ranker rankings
        rankers (List[str]): List of ranker names

    Returns:
        Dict[Tuple[str, str], float]: Dictionary mapping ranker pairs to agreement percentage
    """
    agreement_by_pair = {}
    num_rankers = len(rankers)

    for i in range(num_rankers):
        for j in range(i + 1, num_rankers):
            r1 = rankers[i]
            r2 = rankers[j]

            common_items = 0
            agreements = 0

            for key, ranker_ranks in rankings_by_key.items():
                rank1 = ranker_ranks.get(r1)
                rank2 = ranker_ranks.get(r2)

                # Only compare if both rankers rated this item
                if rank1 is not None and rank2 is not None:
                    common_items += 1
                    if rank1 == rank2:
                        agreements += 1

            if common_items > 0:
                agreement_percentage = agreements / common_items
            else:
                agreement_percentage = float("nan")  # No common items rated

            agreement_by_pair[(r1, r2)] = agreement_percentage

    return agreement_by_pair


def plot_agreement_matrix(
    agreement_matrix: pd.DataFrame, output_path: Path, title: str = "Ranker Agreement Matrix"
):
    """Generate a heatmap of agreement between ranker models.

    Args:
        agreement_matrix (pd.DataFrame): Matrix of agreement scores
        output_path (Path): Path to save the plot
        title (str): Title for the plot
    """
    if agreement_matrix.empty or agreement_matrix.shape[0] < 2:
        print(f"Skipping plot '{title}': Not enough data.")
        return

    plt.figure(
        figsize=(max(8, agreement_matrix.shape[1] * 0.8), max(6, agreement_matrix.shape[0] * 0.8))
    )
    mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)  # Mask upper triangle

    sns.heatmap(
        agreement_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.5,
        mask=mask,
        vmin=0,  # Agreement percentage is typically 0-1
        vmax=1,
        cbar_kws={"label": "Pairwise Agreement Score"},
    )

    plt.title(title, wrap=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def process_global_mode(
    rankings_by_key: Dict[Tuple[str, str, str, str], Dict[str, int]],
    rankers: List[str],
    output_dir: Path,
):
    """Process data in global mode - calculate agreement across all items.

    Args:
        rankings_by_key (Dict): Dictionary mapping keys to ranker rankings
        rankers (List[str]): List of ranker names
        output_dir (Path): Directory to save output files
    """
    global_output_dir = output_dir / "global"
    global_output_dir.mkdir(parents=True, exist_ok=True)
    print("\nProcessing in GLOBAL mode...")

    if len(rankers) < 2:
        print("Need at least two rankers for global analysis. Skipping.")
        return

    # Filter items rated by at least two rankers
    items_for_kappa = []
    valid_keys = []
    for key, ranker_ranks in rankings_by_key.items():
        valid_rankers = [r for r in rankers if r in ranker_ranks]
        if len(valid_rankers) >= 2:
            # Prepare data row for this item, considering only the valid rankers for this specific item
            item_data = prepare_data_for_fleiss_kappa({r: ranker_ranks[r] for r in valid_rankers})
            items_for_kappa.append(item_data)
            valid_keys.append(key)  # Keep track of which items were included

    if not items_for_kappa:
        print("No items found rated by at least two rankers. Skipping global analysis.")
        return

    # Stack the item rows into the final data matrix for Fleiss Kappa
    data_matrix = np.vstack(items_for_kappa)

    print(f"Calculating Global Fleiss' Kappa across {data_matrix.shape[0]} items...")
    global_kappa = calculate_fleiss_kappa(data_matrix)

    if pd.isna(global_kappa):
        print("Global Fleiss' Kappa could not be calculated.")
    else:
        print(f"Global Fleiss' Kappa: {global_kappa:.4f}")
        global_results = pd.DataFrame(
            [
                {
                    "mode": "global",
                    "fleiss_kappa": global_kappa,
                    "num_items": data_matrix.shape[0],
                    "num_rankers": len(rankers),
                }
            ]
        )
        global_results.to_csv(global_output_dir / "global_fleiss_kappa.csv", index=False)

    # Calculate and save pairwise agreement using ALL original items where pairs overlap
    print("Calculating Global Pairwise Agreement...")
    # Use the full rankings_by_key here, as pairwise agreement only needs pairs
    pairwise_agreement = calculate_pairwise_agreement(rankings_by_key, rankers)

    pairwise_data = []
    for (r1, r2), agreement in pairwise_agreement.items():
        # Count common items for this specific pair
        common_items_count = sum(1 for key, rr in rankings_by_key.items() if r1 in rr and r2 in rr)
        pairwise_data.append(
            {
                "ranker1": r1,
                "ranker2": r2,
                "agreement": agreement,
                "common_items": common_items_count,
            }
        )

    if pairwise_data:
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df.to_csv(global_output_dir / "global_pairwise_agreement.csv", index=False)

        # Create and save agreement matrix
        agreement_matrix = pairwise_df.pivot(index="ranker1", columns="ranker2", values="agreement")
        # Make it symmetric and add diagonal
        agreement_matrix = agreement_matrix.combine_first(agreement_matrix.T)
        for r in rankers:
            if r not in agreement_matrix.index:
                agreement_matrix.loc[r, r] = 1.0  # Add missing rankers if any
            else:
                agreement_matrix.loc[r, r] = 1.0
        agreement_matrix = agreement_matrix.reindex(index=rankers, columns=rankers)  # Ensure order

        output_path = global_output_dir / "global_agreement_matrix.png"
        plot_agreement_matrix(agreement_matrix, output_path, title="Global Ranker Agreement Matrix")
        agreement_matrix.to_csv(global_output_dir / "global_agreement_matrix.csv")
        print(f"Global agreement results saved to {global_output_dir}")
    else:
        print("No pairwise agreement data generated.")


def process_model_pair_mode(
    rankings_by_key: Dict[Tuple[str, str, str, str], Dict[str, int]],
    rankers: List[str],
    output_dir: Path,
):
    """Process data in model_pair mode - calculate agreement for each evaluator-candidate_pair.

    Args:
        rankings_by_key (Dict): Dictionary mapping keys to ranker rankings
        rankers (List[str]): List of ranker names
        output_dir (Path): Directory to save output files
    """
    model_pair_output_dir = output_dir / "model_pair"
    model_pair_output_dir.mkdir(parents=True, exist_ok=True)
    print("\nProcessing in MODEL_PAIR mode...")

    if len(rankers) < 2:
        print("Need at least two rankers for model-pair analysis. Skipping.")
        return

    # Group keys by (evaluator, model1, model2)
    grouped_keys = defaultdict(list)
    for key in rankings_by_key.keys():
        evaluator, m1, m2, input_str = key
        group_key = (evaluator, m1, m2)
        grouped_keys[group_key].append(key)

    model_pair_kappa_results = []
    all_pairwise_agreement_data = []

    for group_key, item_keys in grouped_keys.items():
        evaluator, m1, m2 = group_key

        # Filter rankings for this specific model pair
        pair_rankings = {k: rankings_by_key[k] for k in item_keys}

        items_for_kappa = []
        for key in item_keys:
            ranker_ranks = pair_rankings[key]
            valid_rankers = [r for r in rankers if r in ranker_ranks]
            if len(valid_rankers) >= 2:
                item_data = prepare_data_for_fleiss_kappa(
                    {r: ranker_ranks[r] for r in valid_rankers}
                )
                items_for_kappa.append(item_data)

        if not items_for_kappa:
            # print(f"Skipping Kappa for {group_key}: No items found rated by at least two rankers.")
            kappa = float("nan")
            num_items = 0
        else:
            data_matrix = np.vstack(items_for_kappa)
            num_items = data_matrix.shape[0]
            kappa = calculate_fleiss_kappa(data_matrix)

        # print(f"Fleiss' Kappa for {group_key}: {kappa:.4f} ({num_items} items)" if not pd.isna(kappa) else f"Fleiss' Kappa for {group_key}: NaN ({num_items} items)")
        model_pair_kappa_results.append(
            {
                "evaluator": evaluator,
                "model1": m1,
                "model2": m2,
                "fleiss_kappa": kappa,
                "num_items": num_items,
                "num_rankers": len(rankers),  # Total potential rankers considered
            }
        )

        # Calculate pairwise agreement for this model pair
        pairwise_agreement = calculate_pairwise_agreement(pair_rankings, rankers)
        pair_agreement_data = []
        for (r1, r2), agreement in pairwise_agreement.items():
            common_items_count = sum(
                1 for key, rr in pair_rankings.items() if r1 in rr and r2 in rr
            )
            all_pairwise_agreement_data.append(
                {
                    "evaluator": evaluator,
                    "model1": m1,
                    "model2": m2,
                    "ranker1": r1,
                    "ranker2": r2,
                    "agreement": agreement,
                    "common_items": common_items_count,
                }
            )
            pair_agreement_data.append(
                {"ranker1": r1, "ranker2": r2, "agreement": agreement}
            )  # For matrix

        # Create and save agreement matrix for this pair if enough data
        if pair_agreement_data:
            pairwise_df = pd.DataFrame(pair_agreement_data)
            if not pairwise_df.empty and len(rankers) >= 2:
                agreement_matrix = pairwise_df.pivot(
                    index="ranker1", columns="ranker2", values="agreement"
                )
                agreement_matrix = agreement_matrix.combine_first(agreement_matrix.T)
                for r in rankers:
                    if r not in agreement_matrix.index:
                        agreement_matrix.loc[r, r] = 1.0
                    else:
                        agreement_matrix.loc[r, r] = 1.0
                agreement_matrix = agreement_matrix.reindex(index=rankers, columns=rankers)

                safe_eval = sanitize_filename(evaluator)
                safe_m1 = sanitize_filename(m1)
                safe_m2 = sanitize_filename(m2)

                output_path = (
                    model_pair_output_dir
                    / f"agreement_matrix_{safe_eval}_{safe_m1}_VS_{safe_m2}.png"
                )
                plot_title = f"Ranker Agreement Matrix\nEvaluator: {evaluator}\nPair: {m1} vs {m2}"
                plot_agreement_matrix(agreement_matrix, output_path, title=plot_title)
                # agreement_matrix.to_csv(model_pair_output_dir / f"agreement_matrix_{safe_eval}_{safe_m1}_VS_{safe_m2}.csv")

    # Save aggregated results for model_pair mode
    if model_pair_kappa_results:
        kappa_df = pd.DataFrame(model_pair_kappa_results)
        kappa_df.to_csv(
            model_pair_output_dir / "model_pair_fleiss_kappa.csv", index=False, float_format="%.4f"
        )
        print(f"Model-pair Fleiss' Kappa results saved to {model_pair_output_dir}")

        # Create pivot table for kappa
        try:
            pivot_kappa = kappa_df.pivot_table(
                index=["evaluator", "model1"], columns="model2", values="fleiss_kappa"
            )
            pivot_kappa.to_csv(
                model_pair_output_dir / "model_pair_fleiss_kappa_pivot.csv", float_format="%.4f"
            )
        except Exception as e:
            print(f"Could not create pivot table for model pair kappa: {e}")

    else:
        print("No Fleiss' Kappa results generated for model-pair mode.")

    if all_pairwise_agreement_data:
        pairwise_df_all = pd.DataFrame(all_pairwise_agreement_data)
        pairwise_df_all.to_csv(
            model_pair_output_dir / "model_pair_pairwise_agreement.csv",
            index=False,
            float_format="%.4f",
        )
        print(f"Model-pair pairwise agreement results saved to {model_pair_output_dir}")
    else:
        print("No pairwise agreement results generated for model-pair mode.")


def main():
    """Run the main correlation analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Analyze correlation between ranker models for ranking compliance judgements."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="analysis/ranking_correlation_config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["global", "model_pair", "all"],
        default="all",
        help="Analysis mode: global (all items), model_pair (per evaluator-candidate pair), or all modes",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at {config_path}")
        # Create template config
        template_config = {
            "rankings_dir": "data/rankings",  # Base directory for ranking outputs
            "output_dir": "analysis/ranking_correlation",  # Where to save results
            "rankers": [
                "ranker_model_1",
                "ranker_model_2",
            ],  # List of ranker model names (must match directory names)
            "evaluators": ["evaluator_model"],  # List of evaluator models to analyze
            "models": [],  # Optional: List of candidate models to include (if empty, use all found)
            # 'filters': { # Optional: More specific filters if needed
            #     'statements': ['statement_id_1']
            # }
        }
        try:
            with open(config_path, "w") as f:
                yaml.dump(template_config, f, default_flow_style=False, sort_keys=False)
            print(f"Created template configuration file at {config_path}. Please edit it.")
        except Exception as e:
            print(f"Failed to create template config file: {e}")
        return

    try:
        config = load_config(args.config)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        return
    except Exception as e:
        print(f"Error reading configuration file {config_path}: {e}")
        return

    rankings_base_dir = Path(config.get("rankings_dir", "data/rankings"))
    output_dir = Path(config.get("output_dir", "analysis/ranking_correlation"))
    rankers = config.get("rankers", [])
    evaluators = config.get("evaluators", [])
    models = config.get("models", [])  # Optional filtering by candidate model

    if not rankers or len(rankers) < 2:
        print("Error: Configuration must specify at least two 'rankers' to compare.")
        return
    if not evaluators:
        print("Error: Configuration must specify at least one 'evaluator'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Scan all relevant ranking files
    print(
        f"Scanning ranking files in {rankings_base_dir} for {len(rankers)} rankers and {len(evaluators)} evaluators..."
    )
    # The key is (evaluator, modelA, modelB, input_str), value is {ranker: rank}
    rankings_by_key = scan_ranking_files(rankings_base_dir, rankers, evaluators, models)

    if not rankings_by_key:
        print("No valid ranking data found for the specified configuration. Exiting.")
        return

    print(
        f"Found {len(rankings_by_key)} unique (evaluator, model_pair, input) combinations with rankings."
    )

    # Process data according to the selected mode
    if args.mode in ["global", "all"]:
        process_global_mode(rankings_by_key, rankers, output_dir)

    if args.mode in ["model_pair", "all"]:
        process_model_pair_mode(rankings_by_key, rankers, output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
