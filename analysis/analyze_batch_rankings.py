#!/usr/bin/env python3
"""
Analyze batched ranking results produced by the BatchedRankingPipeline.

Expected input directory structure:
  <base_dir>/spec_<spec>/judge_<judge>/<modelA>x<modelB>/results/*.json

Each JSON must contain a top-level "rankings" array with objects:
  {"input": ..., "output_a": ..., "output_b": ..., "rank": 1|0|-1, ...}

Usage:
  python analyze_batch_rankings.py \
    --spec openai \
    --judge claude-3-7-sonnet-20250219 \
    --models modelA modelB modelC \
    [--all_pairs] \
    [--base_dir data/batched_rankings] \
    [--output_dir analysis/batch_ranking_analysis]
"""
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml  # for YAML config support
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze batched ranking results and compute win/loss/tie counts per model"
    )
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file (optional).")
    parser.add_argument(
        "--spec", type=str, help="Specification name (used to form 'spec_<spec>' directory)"
    )
    parser.add_argument(
        "--judge", type=str, help="Ranking judge job name (used to form 'judge_<judge>' directory)"
    )
    parser.add_argument("--models", nargs="+", help="List of candidate model names to analyze")
    parser.add_argument(
        "--all_pairs",
        action="store_true",
        help="Enable position-bias correction (requires both A x B and B x A folders)",
    )
    parser.add_argument("--base_dir", type=str, help="Base directory for batch ranking outputs")
    parser.add_argument("--output_dir", type=str, help="Directory to write summary CSV and plots")
    return parser.parse_args()


def get_base_dir(base_dir: str, spec: str, judge: str) -> Path:
    base = Path(base_dir) / f"spec_{spec}" / f"judge_{judge}"
    if not base.is_dir():
        sys.exit(f"Error: expected base directory not found: {base}")
    return base


def validate_structure(base: Path, models):
    """
    Scan base directory for valid pair subdirectories of the form '<modelA>x<modelB>'.
    Only directories where both modelA and modelB are in the provided models list,
    and contain a 'results' folder with at least one .json file, are kept.
    Returns a list of Path objects for valid pair directories.
    Exits if no valid pairs found.
    """
    valid_dirs = []
    for d in base.iterdir():
        if not d.is_dir() or "x" not in d.name:
            continue
        a, b = d.name.split("x", 1)
        if a not in models or b not in models:
            print(f"Warning: skipping pair directory '{d.name}' not in specified models")
            continue
        results_dir = d / "results"
        if not results_dir.is_dir():
            print(f"Warning: skipping '{d.name}': missing 'results' subdirectory")
            continue
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            print(f"Warning: skipping '{d.name}': no .json files in results")
            continue
        valid_dirs.append(d)
    if not valid_dirs:
        sys.exit(f"Error: no valid pair directories found under {base} for models {models}")
    return valid_dirs


def analyze_unique_pairs(pair_dirs, models):
    # Enforce canonical ordering by lexicographic model names: only keep A×B where A < B
    sorted_models = sorted(models)
    dir_map = {d.name: d for d in pair_dirs}
    canonical_dirs = []
    for i in range(len(sorted_models)):
        for j in range(i + 1, len(sorted_models)):
            a, b = sorted_models[i], sorted_models[j]
            name = f"{a}x{b}"
            if name in dir_map:
                canonical_dirs.append(dir_map[name])
            else:
                print(f"Warning: missing canonical directory {name}, skipping")
    pair_dirs = canonical_dirs

    # Count each ordered pair separately, no bias correction
    lengths = {m: {"total": 0, "count": 0} for m in models}
    stats = {m: {"wins": 0, "losses": 0, "ties": 0} for m in models}
    # Track wins per statement
    statement_wins = {m: {} for m in models}
    for pair_dir in pair_dirs:
        a, b = pair_dir.name.split("x", 1)
        results_dir = pair_dir / "results"
        for file in sorted(results_dir.glob("*.json")):
            stmt = file.stem
            data = json.load(open(file, "r")).get("rankings", [])
            for r in data:
                # track response lengths
                output_a = r.get("output_a", "")
                output_b = r.get("output_b", "")
                lengths[a]["total"] += len(output_a)
                lengths[a]["count"] += 1
                lengths[b]["total"] += len(output_b)
                lengths[b]["count"] += 1

                rank = r.get("rank", 0)
                if rank == 1:
                    stats[a]["wins"] += 1
                    stats[b]["losses"] += 1
                    statement_wins[a][stmt] = statement_wins[a].get(stmt, 0) + 1
                elif rank == -1:
                    stats[b]["wins"] += 1
                    stats[a]["losses"] += 1
                    statement_wins[b][stmt] = statement_wins[b].get(stmt, 0) + 1
                else:
                    stats[a]["ties"] += 1
                    stats[b]["ties"] += 1
    # compute average response lengths
    avg_lengths = {
        m: (lengths[m]["total"] / lengths[m]["count"] if lengths[m]["count"] > 0 else 0.0)
        for m in models
    }
    return stats, statement_wins, avg_lengths


def analyze_all_pairs(pair_dirs, models):
    # Correct for position bias: only unordered pairs where both orders exist
    lengths = {m: {"total": 0, "count": 0} for m in models}
    stats = {m: {"wins": 0, "losses": 0, "ties": 0} for m in models}
    # Track wins per statement
    statement_wins = {m: {} for m in models}
    # Map each ordered pair to its directory
    available = {}
    for d in pair_dirs:
        a, b = d.name.split("x", 1)
        available[(a, b)] = d
    seen = set()
    for (a, b), dir_ab in available.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        dir_ba = available.get((b, a))
        if not dir_ba:
            print(f"Warning: missing reverse directory for pair {a}x{b}, skipping bias correction")
            continue
        res_ab = dir_ab / "results"
        res_ba = dir_ba / "results"
        stmts_ab = {p.stem for p in res_ab.glob("*.json")}
        stmts_ba = {p.stem for p in res_ba.glob("*.json")}
        common = stmts_ab & stmts_ba
        if not common:
            print(f"Warning: no common statement JSONs for pair {a} and {b}, skipping")
            continue
        for stmt in sorted(common):
            data_ab = json.load(open(res_ab / f"{stmt}.json", "r")).get("rankings", [])
            data_ba = json.load(open(res_ba / f"{stmt}.json", "r")).get("rankings", [])
            map_ab = {r["input"]: r["rank"] for r in data_ab}
            map_ba = {r["input"]: r["rank"] for r in data_ba}
            # build mapping to retrieve outputs for length tracking
            rec_ab_map = {r["input"]: r for r in data_ab}
            for inp, rank_ab in map_ab.items():
                if inp not in map_ba:
                    print(f"Warning: missing input {inp} in {res_ba / f'{stmt}.json'}!")
                    continue
                # track response lengths
                rec = rec_ab_map.get(inp, {})
                output_a = rec.get("output_a", "")
                output_b = rec.get("output_b", "")
                lengths[a]["total"] += len(output_a)
                lengths[a]["count"] += 1
                lengths[b]["total"] += len(output_b)
                lengths[b]["count"] += 1

                rank_ba = map_ba[inp]
                final = rank_ab if rank_ab == -rank_ba else 0
                if final == 1:
                    stats[a]["wins"] += 1
                    stats[b]["losses"] += 1
                    statement_wins[a][stmt] = statement_wins[a].get(stmt, 0) + 1
                elif final == -1:
                    stats[b]["wins"] += 1
                    stats[a]["losses"] += 1
                    statement_wins[b][stmt] = statement_wins[b].get(stmt, 0) + 1
                else:
                    stats[a]["ties"] += 1
                    stats[b]["ties"] += 1
    # compute average response lengths
    avg_lengths = {
        m: (lengths[m]["total"] / lengths[m]["count"] if lengths[m]["count"] > 0 else 0.0)
        for m in models
    }
    return stats, statement_wins, avg_lengths


def save_summary(stats, statement_wins, avg_lengths, output_dir: Path, spec: str, judge: str):
    # Write CSV and bar plot of total wins
    df = pd.DataFrame(stats).T  # index: model, columns: wins, losses, ties
    out = output_dir / f"spec_{spec}" / f"judge_{judge}"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ranking_summary.csv"
    df.to_csv(csv_path)
    print(f"Saved summary CSV to {csv_path}")
    # bar chart
    plt.figure(figsize=(8, 5))
    df["wins"].sort_values(ascending=False).plot(kind="bar", color="skyblue")
    plt.title(f"Total Wins per Model\nSpec: {spec} | Judge: {judge}")
    plt.ylabel("Win Count")
    plt.tight_layout()
    plot_path = out / "wins_per_model.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved bar chart to {plot_path}")

    # --- per-statement wins heatmap and CSV ---
    stmt_df = pd.DataFrame.from_dict(statement_wins, orient="index").fillna(0).astype(int)
    # Save per-statement wins CSV
    stmt_csv = out / "per_statement_wins.csv"
    stmt_df.to_csv(stmt_csv)
    print(f"Saved per-statement wins CSV to {stmt_csv}")
    # Plot heatmap with dynamic sizing and improved readability
    num_rows, num_cols = stmt_df.shape
    width = max(8, num_cols * 0.5)
    height = max(6, num_rows * 0.4)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(stmt_df.values, aspect="auto", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, label="Wins per Statement")
    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(stmt_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(stmt_df.index, fontsize=8)
    ax.set_title(f"Wins per Statement Heatmap\nSpec: {spec} | Judge: {judge}")
    plt.tight_layout()
    stmt_plot = out / "per_statement_wins_heatmap.png"
    fig.savefig(stmt_plot, dpi=300)
    plt.close(fig)
    print(f"Saved per-statement wins heatmap to {stmt_plot}")

    # --- Combined overall and per-statement summary figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 2]})
    # Top: overall wins bar chart
    df["wins"].sort_values(ascending=False).plot(kind="bar", color="skyblue", ax=ax1)
    ax1.set_title(f"Total Wins per Model\nSpec: {spec} | Judge: {judge}")
    ax1.set_ylabel("Wins")
    # Bottom: per-statement heatmap
    im = ax2.imshow(stmt_df.values, aspect="auto", cmap="Blues")
    fig.colorbar(im, ax=ax2, label="Wins per Statement")
    ax2.set_xticks(range(len(stmt_df.columns)))
    ax2.set_xticklabels(stmt_df.columns, rotation=90)
    ax2.set_yticks(range(len(stmt_df.index)))
    ax2.set_yticklabels(stmt_df.index)
    ax2.set_title("Wins per Statement Heatmap")
    plt.tight_layout()
    combined_path = out / "wins_overall_and_per_statement.png"
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined summary figure to {combined_path}")

    # --- wins vs avg response length scatter plot ---
    df_with_len = df.copy()
    df_with_len["avg_length"] = pd.Series(avg_lengths)
    plt.figure(figsize=(6, 6))
    plt.scatter(df_with_len["avg_length"], df_with_len["wins"])
    for model, x, y in zip(df_with_len.index, df_with_len["avg_length"], df_with_len["wins"]):
        plt.annotate(model, (x, y), textcoords="offset points", xytext=(5, -5), ha="left")
    plt.xlabel("Average Response Length (chars)")
    plt.ylabel("Total Wins")
    plt.title(f"Wins vs. Avg Response Length\nSpec: {spec} | Judge: {judge}")
    plt.tight_layout()
    scatter_path = out / "wins_vs_avg_length.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved wins vs avg response length plot to {scatter_path}")


def main():
    args = parse_args()

    # Load YAML config if provided
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            sys.exit(f"Error loading config file {args.config}: {e}")

    # Merge CLI args and config (CLI overrides)
    spec = args.spec or config.get("spec")
    judge = args.judge or config.get("judge")
    models = args.models or config.get("models")
    if models and not isinstance(models, list):
        sys.exit("Error: 'models' must be a list of model names.")

    # all_pairs: default False, enable if CLI or config says true
    all_pairs = bool(config.get("all_pairs", False))
    if args.all_pairs:
        all_pairs = True

    # Directories: CLI overrides config, config overrides defaults
    base_dir = args.base_dir or config.get("base_dir", "data/batched_rankings")
    output_dir = args.output_dir or config.get("output_dir", "analysis/batch_ranking_analysis")

    # Validate required parameters
    missing = []
    if not spec:
        missing.append("spec")
    if not judge:
        missing.append("judge")
    if not models:
        missing.append("models")
    if missing:
        sys.exit(
            f"Error: missing required parameters: {', '.join(missing)}. Provide via CLI or config file."
        )

    # Validate and get list of valid pair directories
    base = get_base_dir(base_dir, spec, judge)
    valid_dirs = validate_structure(base, models)
    # Run analysis
    if all_pairs:
        print(f"Running all_pairs analysis on {base}")
        stats, statement_wins, avg_lengths = analyze_all_pairs(valid_dirs, models)
    else:
        print(f"Running unique_pairs analysis on {base}")
        stats, statement_wins, avg_lengths = analyze_unique_pairs(valid_dirs, models)
    save_summary(stats, statement_wins, avg_lengths, Path(output_dir), spec, judge)


if __name__ == "__main__":
    main()
