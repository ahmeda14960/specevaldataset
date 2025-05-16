#!/usr/bin/env python3
"""Run analysis on SpecEval ranking results.

This script reads ranking output JSON files produced by the RankingPipeline,
aggregates win/loss/tie counts for each model, and handles an 'all_pairs' flag
in the configuration to correct for position bias by checking reversed pair folders.

The expected input directory structure is:
<rankings_dir>/<ranker_model>/<evaluator_model>/<ModelAVSModelB>/...

Usage:
  python analyze_rankings.py --config analysis/rankings_config.yaml
"""

import json
import argparse
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    """Load the configuration file specified by the given path.

    This function reads the YAML configuration file located at the specified path
    and returns its content as a dictionary.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The content of the configuration file as a dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_pair_dir(dir_name: str):
    """Split a directory name of the form 'ModelA<sep>ModelB' into (ModelA, ModelB). Supports both 'VS' and 'x'."""
    if "VS" in dir_name:
        return tuple(dir_name.split("VS", 1))
    if "x" in dir_name:
        return tuple(dir_name.split("x", 1))
    return None


def detect_separator(base_dir: Path) -> str:
    """Detect which separator ('VS' or 'x') is used in model-pair directory names under base_dir."""
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        if "VS" in entry.name:
            return "VS"
        if "x" in entry.name:
            return "x"
    # default to 'VS' if none found
    return "VS"


def analyze_all_pairs(
    rankings_base_dir: str,
    ranker: str,
    evaluator: str,
    models: list,
    spec: str = None,
    ranker_model: str = None,
) -> tuple:
    """Analyze all model pairs for ranking agreement.

    For each unordered model pair, load both A vs B and B vs A folders, compare per-input ranks,
    count only consistent wins (rank_AB == -rank_BA) and treat inconsistencies as ties.

    Returns:
        tuple: Stats in format (stats, statement_wins, avg_lengths)
    """
    root = Path(rankings_base_dir)
    # determine correct base directory: support layouts with optional spec/ranker_model nesting
    if spec and ranker_model:
        cand = root / ranker_model / spec / ranker / evaluator
        if cand.is_dir():
            base_dir = cand
        else:
            print(f"Error: Base directory not found under {cand}")
            return ({m: {"wins": 0, "losses": 0, "ties": 0} for m in models}, {}, {})
    elif (root / ranker / evaluator).is_dir():
        base_dir = root / ranker / evaluator
    elif (root / evaluator / ranker).is_dir():
        base_dir = root / evaluator / ranker
    elif root.is_dir() and any(
        ("VS" in e.name or "x" in e.name) and e.is_dir() for e in root.iterdir()
    ):
        base_dir = root
    else:
        print(
            f"Error: Base directory not found under {root}/{ranker}/{evaluator} or {root}/{evaluator}/{ranker}"
        )
        return ({m: {"wins": 0, "losses": 0, "ties": 0} for m in models}, {}, {})

    # detect which separator is used in model-pair dirs
    sep = detect_separator(base_dir)

    stats = {m: {"wins": 0, "losses": 0, "ties": 0} for m in models}
    lengths = {m: {"total": 0, "count": 0} for m in models}
    statement_wins = {m: {} for m in models}

    # Generate unique unordered pairs
    seen = set()
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                continue
            key = tuple(sorted((m1, m2)))
            if key in seen:
                continue
            seen.add(key)
            a, b = key
            # construct pair directories using detected separator
            dir_ab = base_dir / f"{a}{sep}{b}"
            dir_ba = base_dir / f"{b}{sep}{a}"
            # allow optional 'results' subfolder
            ab_ok = dir_ab.is_dir() or (dir_ab / "results").is_dir()
            ba_ok = dir_ba.is_dir() or (dir_ba / "results").is_dir()
            if not (ab_ok and ba_ok):
                print(f"Warning: Missing pair folders for {a} vs {b}, skipping.")
                continue

            # find JSON directories (support 'results' subfolder)
            json_ab = dir_ab / "results" if (dir_ab / "results").is_dir() else dir_ab
            json_ba = dir_ba / "results" if (dir_ba / "results").is_dir() else dir_ba
            # Find common statement files
            stmts_ab = {p.stem for p in json_ab.glob("*.json")}
            stmts_ba = {p.stem for p in json_ba.glob("*.json")}
            common_stmts = stmts_ab & stmts_ba

            for stmt in sorted(common_stmts):
                # Load ranking data for the statement
                try:
                    with open(json_ab / f"{stmt}.json", "r") as f:
                        data_ab = json.load(f).get("rankings", [])
                except json.JSONDecodeError:
                    print(
                        f"Warning: invalid or empty JSON in {json_ab / f'{stmt}.json'}, skipping statement {stmt}"
                    )
                    continue
                try:
                    with open(json_ba / f"{stmt}.json", "r") as f:
                        data_ba = json.load(f).get("rankings", [])
                except json.JSONDecodeError:
                    print(
                        f"Warning: invalid or empty JSON in {json_ba / f'{stmt}.json'}, skipping statement {stmt}"
                    )
                    continue

                # Build input->rank maps
                map_ab = {r["input"]: r["rank"] for r in data_ab}
                map_ba = {r["input"]: r["rank"] for r in data_ba}

                # Compare per-input
                for inp, rank_ab in map_ab.items():
                    rank_ba = map_ba.get(inp)
                    if rank_ba is None:
                        continue
                    # Check consistency
                    if rank_ab == -rank_ba:
                        final_rank = rank_ab
                    else:
                        # Inconsistent ranking across order => treat as tie
                        final_rank = 0

                    if final_rank == 1:
                        stats[a]["wins"] += 1
                        stats[b]["losses"] += 1
                        statement_wins[a][stmt] = statement_wins[a].get(stmt, 0) + 1
                    elif final_rank == -1:
                        stats[b]["wins"] += 1
                        stats[a]["losses"] += 1
                        statement_wins[b][stmt] = statement_wins[b].get(stmt, 0) + 1
                    else:
                        stats[a]["ties"] += 1
                        stats[b]["ties"] += 1

                # accumulate response lengths
                for r in data_ab:
                    lengths[a]["total"] += len(r.get("output_a", ""))
                    lengths[a]["count"] += 1
                    lengths[b]["total"] += len(r.get("output_b", ""))
                    lengths[b]["count"] += 1

    # compute average lengths per model
    avg_lengths = {
        m: (lengths[m]["total"] / lengths[m]["count"] if lengths[m]["count"] > 0 else 0)
        for m in models
    }
    return stats, statement_wins, avg_lengths


def analyze_unique_pairs(
    rankings_base_dir: str,
    ranker: str,
    evaluator: str,
    models: list,
    spec: str = None,
    ranker_model: str = None,
) -> tuple:
    """Analyze unique model pairs for ranking agreement.

    For each directory modelAVSmodelB, count rank==1 as a win for modelA, rank==-1 as a win for modelB,
    rank==0 as tie. Does not correct for position bias.

    Returns:
        tuple: Stats in format (stats, statement_wins, avg_lengths)
    """
    root = Path(rankings_base_dir)
    # determine correct base directory: support layouts with optional spec/ranker_model nesting
    if spec and ranker_model:
        cand = root / ranker_model / spec / ranker / evaluator
        if cand.is_dir():
            base_dir = cand
        else:
            print(f"Error: Base directory not found under {cand}")
            return ({m: {"wins": 0, "losses": 0, "ties": 0} for m in models}, {}, {})
    elif (root / ranker / evaluator).is_dir():
        base_dir = root / ranker / evaluator
    elif (root / evaluator / ranker).is_dir():
        base_dir = root / evaluator / ranker
    elif root.is_dir() and any(
        ("VS" in e.name or "x" in e.name) and e.is_dir() for e in root.iterdir()
    ):
        base_dir = root
    else:
        print(
            f"Error: Base directory not found under {root}/{ranker}/{evaluator} or {root}/{evaluator}/{ranker}"
        )
        return ({m: {"wins": 0, "losses": 0, "ties": 0} for m in models}, {}, {})

    # detect which separator is used in model-pair dirs
    sep = detect_separator(base_dir)

    stats = {m: {"wins": 0, "losses": 0, "ties": 0} for m in models}
    lengths = {m: {"total": 0, "count": 0} for m in models}
    statement_wins = {m: {} for m in models}

    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_pair_dir(entry.name)
        if not parsed:
            continue
        m1, m2 = parsed
        if m1 not in models or m2 not in models:
            continue

        # support optional 'results' subfolder
        data_dir = entry / "results" if (entry / "results").is_dir() else entry
        for file in data_dir.glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f).get("rankings", [])
            stmt = file.stem
            for r in data:
                # accumulate lengths for each model
                lengths[m1]["total"] += len(r.get("output_a", ""))
                lengths[m1]["count"] += 1
                lengths[m2]["total"] += len(r.get("output_b", ""))
                lengths[m2]["count"] += 1
                rank = r.get("rank", 0)
                if rank == 1:
                    stats[m1]["wins"] += 1
                    stats[m2]["losses"] += 1
                    statement_wins[m1][stmt] = statement_wins[m1].get(stmt, 0) + 1
                elif rank == -1:
                    stats[m2]["wins"] += 1
                    stats[m1]["losses"] += 1
                    statement_wins[m2][stmt] = statement_wins[m2].get(stmt, 0) + 1
                else:
                    stats[m1]["ties"] += 1
                    stats[m2]["ties"] += 1

    # compute average lengths per model
    avg_lengths = {
        m: (lengths[m]["total"] / lengths[m]["count"] if lengths[m]["count"] > 0 else 0)
        for m in models
    }
    return stats, statement_wins, avg_lengths


def save_summary(
    stats: dict,
    statement_wins: dict,
    avg_lengths: dict,
    output_dir: str,
    spec: str,
    ranker_model: str,
):
    """Save summary CSV and bar plots of wins per model and wins per statement.

    Outputs will be placed under output_dir / spec / ranker_model.

    Args:
        stats (dict): Statistics to save
        statement_wins (dict): Wins per statement
        avg_lengths (dict): Average response lengths
        output_dir (str): Base output directory
        spec (str): Name of the spec
        ranker_model (str): Name of the ranker model
    """
    df = pd.DataFrame(stats).T  # models x metrics
    out = Path(output_dir) / spec / ranker_model
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out / "ranking_summary.csv"
    df.to_csv(csv_path)
    print(f"Saved summary CSV to {csv_path}")

    # Bar plot of wins per model
    plt.figure(figsize=(10, 6))
    df["wins"].sort_values(ascending=False).plot(kind="bar", color="skyblue")
    plt.title(f"Total Wins per Model\nSpec: {spec} | Ranker Model: {ranker_model}")
    plt.ylabel("Win Count")
    plt.tight_layout()
    model_bar = out / "wins_per_model.png"
    plt.savefig(model_bar, dpi=300)
    plt.close()
    print(f"Saved wins per model plot to {model_bar}")

    # Bar plot of wins per statement
    stmt_df = pd.DataFrame.from_dict(statement_wins, orient="index").fillna(0).astype(int)
    wins_per_stmt = stmt_df.sum(axis=0)
    plt.figure(figsize=(12, 6))
    wins_per_stmt.sort_values(ascending=False).plot(kind="bar", color="coral")
    plt.title(f"Total Wins per Statement\nSpec: {spec} | Ranker Model: {ranker_model}")
    plt.ylabel("Win Count")
    plt.tight_layout()
    stmt_bar = out / "wins_per_statement.png"
    plt.savefig(stmt_bar, dpi=300)
    plt.close()
    print(f"Saved wins per statement plot to {stmt_bar}")

    # Scatter: win rate vs avg response length
    df2 = df.copy()
    df2["avg_length"] = pd.Series(avg_lengths)
    df2["total"] = df2[["wins", "losses", "ties"]].sum(axis=1)
    df2["win_rate"] = df2["wins"] / df2["total"]
    plt.figure(figsize=(6, 6))
    plt.scatter(df2["avg_length"], df2["win_rate"], color="green")
    for model, x, y in zip(df2.index, df2["avg_length"], df2["win_rate"]):
        plt.annotate(model, (x, y), textcoords="offset points", xytext=(5, -5), ha="left")
    plt.xlabel("Average Response Length (chars)")
    plt.ylabel("Win Rate")
    plt.title(f"Win Rate vs Avg Response Length\nSpec: {spec} | Ranker Model: {ranker_model}")
    plt.tight_layout()
    scatter2 = out / "win_rate_vs_avg_length.png"
    plt.savefig(scatter2, dpi=300)
    plt.close()
    print(f"Saved win rate vs avg length plot to {scatter2}")


def main():
    """Run the main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyze SpecEval ranking results")
    parser.add_argument(
        "--config",
        type=str,
        default="analysis/rankings_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ranker = config.get("ranker")
    evaluator = config.get("evaluator")
    models = config.get("models", [])
    all_pairs = config.get("all_pairs", False)
    rankings_dir = config.get("rankings_dir", "data/rankings")
    output_dir = config.get("output_dir", "analysis/rankings_analysis")
    spec = config.get("spec")
    ranker_model = config.get("ranker_model")

    if not ranker or not evaluator or not models:
        print("Error: ranker, evaluator, and models must be specified in the config file.")
        return

    if all_pairs:
        print(
            f"Running analysis with all_pairs=True for spec: {spec}, ranker_model: {ranker_model}, ranker: {ranker}, evaluator: {evaluator}"
        )
        stats, statement_wins, avg_lengths = analyze_all_pairs(
            rankings_dir, ranker, evaluator, models, spec, ranker_model
        )
    else:
        print(
            f"Running analysis with all_pairs=False for spec: {spec}, ranker_model: {ranker_model}, ranker: {ranker}, evaluator: {evaluator}"
        )
        stats, statement_wins, avg_lengths = analyze_unique_pairs(
            rankings_dir, ranker, evaluator, models, spec, ranker_model
        )
        print(
            "Suggestion: enable all_pairs in the config or rerun RankingPipeline with --all-pairs "
            + "to mitigate potential position bias."
        )
    print(stats)
    if stats:
        save_summary(stats, statement_wins, avg_lengths, output_dir, spec, ranker_model)
        print("Analysis complete.")
    else:
        print("Analysis skipped due to missing data or errors.")


if __name__ == "__main__":
    main()
