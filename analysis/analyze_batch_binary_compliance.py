#!/usr/bin/env python3
"""Batch binary compliance analysis for SpecEval.

Reads JSON files under data/batched_compliance/spec_{spec}/judge_{judge}/{model}/results/*.json,
computes overall compliance fraction per model, and generates bar plots for each spec-judge combination.
"""

import os
import json
import argparse
import yaml
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

# Mapping for custom short model names.
# Edit these entries to map full model directory names to short display names in short mode.
SHORT_NAME_MAP = {
    "claude-3-5-haiku-20241022": "claude-3-5-haiku",
    "claude-3-5-sonnet-20240620": "claude-3-5-sonnet",
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    "gpt-4.1-2025-04-14": "gpt-4.1",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "gpt-4o-2024-11-20": "gpt-4o",
    "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-2.0-flash-001": "gemini-2.0-flash",
    "deepseek-ai-DeepSeek-V3": "DeepSeek-V3",
    "Qwen/Qwen3-235B-A22B-fp8-tput": "Qwen3-235B",
    "Qwen-Qwen3-235B-A22B-fp8-tput": "Qwen3-235B",
    "Qwen-Qwen2.5-72B-Instruct-Turbo": "Qwen2.5-72B",
    "Qwen-Qwen2-72B-Instruct": "Qwen2-72B",
    "meta-llama-Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick",
    "meta-llama-Meta-Llama-3.1-405B-Instruct-Turbo": "Llama-3.1-405B",
}

def clean_model_name(model_name: str) -> str:
    """Clean up model name for display."""
    # Strip provider prefixes if any
    name = re.sub(r"^[a-zA-Z]+-[a-zA-Z]+[-/]", "", model_name)
    # Remove common suffixes for readability
    name = name.replace("-Instruct", "").replace("-instruct", "").replace("-Turbo", "")
    # Truncate long names
    if len(name) > 25:
        name = name[:22] + "..."
    return name


def analyze_batch_compliance(config: dict):
    base_dir = Path(config["base_dir"])
    output_dir = Path(config["output_dir"])
    specs = config.get("specs", [])
    judges = config.get("judges", [])
    # Read short_mode flag and define helper for display names
    short_mode = config.get("short_mode", False)
    def get_display_name(model_name: str) -> str:
        if short_mode:
            # print(f"Shortening "{model_name}" "{SHORT_NAME_MAP.get(model_name, '<not in map>')}")
            return SHORT_NAME_MAP.get(model_name, clean_model_name(model_name))
        return clean_model_name(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        # Initialize accumulator for per-model compliance scores across judges
        spec_model_scores = {}
        # Initialize accumulator for raw per-judge compliance scores for bias analysis
        spec_scores = {}
        # Initialize accumulator for per-statement compliance counts across judges
        spec_per_stmt_counts = {}
        spec_dir = base_dir / f"spec_{spec}"
        if not spec_dir.is_dir():
            print(f"Warning: spec directory not found: {spec_dir}")
            continue

        for judge in judges:
            judge_dir = spec_dir / f"judge_{judge}"
            if not judge_dir.is_dir():
                print(f"Warning: judge directory not found: {judge_dir}")
                continue

            scores = {}
            avg_lengths = {}
            # Track per-statement counts per model
            per_stmt_counts = {}
            for model_dir in sorted(judge_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                results_dir = model_dir / "results"
                if not results_dir.is_dir():
                    print(f"Warning: results directory not found for model {model_dir.name}")
                    continue

                # Overall compliance counters
                total = 0
                compliant = 0
                # Per-statement compliance counters for this model
                per_model_counts = {}
                length_total = 0
                length_count = 0
                for json_file in results_dir.glob("*.json"):
                    # Statement identifier from filename
                    stmt = json_file.stem
                    per_model_counts.setdefault(stmt, {"compliant": 0, "total": 0})
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                        for entry in data.get("results", []):
                            total += 1
                            per_model_counts[stmt]["total"] += 1
                            entry_compliant = entry.get("compliant")
                            if entry_compliant is None:
                                print(f"Found none for {json_file}", flush=True)
                            if entry_compliant:
                                compliant += 1
                                per_model_counts[stmt]["compliant"] += 1
                            # track response length
                            output = entry.get("output", "")
                            length_total += len(output)
                            length_count += 1
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

                score = compliant / total if total > 0 else 0.0
                avg_len = length_total / length_count if length_count > 0 else 0.0
                scores[model_dir.name] = score
                avg_lengths[model_dir.name] = avg_len
                # Save this model's per-statement counts
                per_stmt_counts[model_dir.name] = per_model_counts
                print(
                    f"Spec={spec}, Judge={judge}, Model={model_dir.name}, Compliance={score:.2f}, AvgLen={avg_len:.1f}"
                )

            if not scores:
                print(f"No models found for spec={spec}, judge={judge}")
                continue
            # Record this judge's raw compliance scores for bias analysis
            spec_scores[judge] = scores
            # Accumulate this judge's model compliance scores for summary stats
            for m, s in scores.items():
                spec_model_scores.setdefault(m, []).append(s)
            # Accumulate per-statement compliance counts across judges
            for m, counts in per_stmt_counts.items():
                spec_per_stmt_counts.setdefault(m, {})
                for stmt, c in counts.items():
                    spec_per_stmt_counts[m].setdefault(stmt, {"compliant": 0, "total": 0})
                    spec_per_stmt_counts[m][stmt]["compliant"] += c.get("compliant", 0)
                    spec_per_stmt_counts[m][stmt]["total"] += c.get("total", 0)

            # Plotting
            models = list(scores.keys())
            values = [scores[m] for m in models]
            display_names = [get_display_name(m) for m in models]
            # Sort by descending compliance
            idx = np.argsort(values)[::-1]
            sorted_names = [display_names[i] for i in idx]
            sorted_values = [values[i] for i in idx]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=sorted_names, y=sorted_values)
            # Removed title and axis labels for minimalist plot
            plt.ylim(0, 1.0)
            for i, v in enumerate(sorted_values):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            if len(sorted_names) > 3:
                plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save plot
            spec_out_dir = output_dir / f"spec_{spec}"
            spec_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = spec_out_dir / f"batch_binary_compliance_{judge}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved plot to {out_file}")

            # Scatter plot: Compliance vs Avg Response Length
            plt.figure(figsize=(8, 6))
            for m, v in scores.items():
                x = avg_lengths[m]
                y = v
                plt.scatter(x, y)
                plt.annotate(
                    get_display_name(m),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                )
            plt.xlabel("Average Response Length (chars)")
            plt.ylabel("Compliance Fraction")
            plt.title(f"Compliance vs Avg Response Length: Spec={spec} | Judge={judge}")
            plt.tight_layout()
            out_file2 = spec_out_dir / f"batch_binary_compliance_vs_avg_length_{judge}.png"
            plt.savefig(out_file2, dpi=300)
            plt.close()
            print(f"Saved compliance vs avg length plot to {out_file2}")

            # Per-statement compliance heatmap
            # Build DataFrame of compliance rates per statement per model
            all_statements = sorted(
                {stmt for counts in per_stmt_counts.values() for stmt in counts}
            )
            df = pd.DataFrame(index=all_statements, columns=scores.keys(), dtype=float)
            for m, counts in per_stmt_counts.items():
                for stmt in all_statements:
                    tot = counts.get(stmt, {}).get("total", 0)
                    comp = counts.get(stmt, {}).get("compliant", 0)
                    df.at[stmt, m] = comp / tot if tot > 0 else np.nan
            # Rename columns for display
            df.columns = [get_display_name(m) for m in df.columns]
            plt.figure(figsize=(max(10, len(df.columns) * 0.8), max(6, len(df.index) * 0.3)))
            sns.heatmap(
                df, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={}
            )
            if len(df.columns) > 3:
                plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out_stmt = spec_out_dir / f"batch_binary_compliance_per_statement_{judge}.png"
            plt.savefig(out_stmt, dpi=300)
            plt.close()
            print(f"Saved per-statement compliance heatmap to {out_stmt}")

        # Per-statement compliance heatmap aggregated over all judges
        all_statements_avg = sorted(
            {stmt for counts in spec_per_stmt_counts.values() for stmt in counts}
        )
        df_avg = pd.DataFrame(index=all_statements_avg, columns=spec_per_stmt_counts.keys(), dtype=float)
        for m, counts in spec_per_stmt_counts.items():
            for stmt in all_statements_avg:
                tot = counts.get(stmt, {}).get("total", 0)
                comp = counts.get(stmt, {}).get("compliant", 0)
                df_avg.at[stmt, m] = comp / tot if tot > 0 else np.nan
        # Rename columns for display
        df_avg.columns = [get_display_name(m) for m in df_avg.columns]
        plt.figure(figsize=(max(10, len(df_avg.columns) * 0.8), max(6, len(df_avg.index) * 0.3)))
        sns.heatmap(df_avg, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={})
        # Minimalist heatmap: no title, no y-axis label, no colorbar label
        if len(df_avg.columns) > 3:
            plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_avg_stmt = spec_out_dir / "batch_binary_compliance_per_statement_avg.png"
        plt.savefig(out_avg_stmt, dpi=300)
        plt.close()
        print(f"Saved averaged per-statement compliance heatmap to {out_avg_stmt}")

        # After all judges for this spec, generate summary per model across judges
        spec_out_dir = output_dir / f"spec_{spec}"
        spec_out_dir.mkdir(parents=True, exist_ok=True)
        # Calculate mean, standard error, and 95% CI for each model
        models_list = sorted(spec_model_scores.keys())
        means = []
        ses = []
        cis = []
        alpha = 0.05
        for m in models_list:
            vals = np.array(spec_model_scores[m])
            m_mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ci = t.ppf(1 - alpha / 2, df=len(vals) - 1) * se if len(vals) > 1 else 0.0
            means.append(m_mean)
            ses.append(se)
            cis.append(ci)

        # Sort models by mean compliance descending (rank 1 first)
        desc_idx = np.argsort(means)[::-1]
        models_list = [models_list[i] for i in desc_idx]
        means = [means[i] for i in desc_idx]
        ses = [ses[i] for i in desc_idx]
        cis = [cis[i] for i in desc_idx]

        # Plot average model compliance across judges with SE and 95% CI (sorted by rank)
        x = np.arange(len(models_list))
        plt.figure(figsize=(10, 6))
        plt.bar(x, means, color="skyblue", label="Mean")
        plt.errorbar(x, means, yerr=ses, fmt="none", ecolor="black", capsize=5, label="SE")
        plt.errorbar(
            x,
            means,
            yerr=cis,
            fmt="none",
            ecolor="red",
            capsize=5,
            label=f"{100*(1-alpha):.0f}% CI",
        )
        plt.xticks(x, [get_display_name(m) for m in models_list], rotation=45, ha="right")
        plt.ylim(0, 1.0)
        plt.legend()
        # Annotate each bar with its rank (1 = highest mean)
        for i in range(len(models_list)):
            plt.text(x[i], means[i] / 2, str(i + 1), ha="center", va="center", fontweight="bold")
        plt.tight_layout()
        out_file_spec = spec_out_dir / "avg_model_compliance_across_judges.png"
        plt.savefig(out_file_spec, dpi=300)
        plt.close()
        print(f"Saved avg model compliance across judges plot to {out_file_spec}")

        # Build judge-model bias table and compute deviations Δ
        df = pd.DataFrame(spec_scores)
        mean_per_model = df.mean(axis=1)
        # Compute per-judge deviation Δ_{J,M} = score(J,M) minus the mean score of model M across all judges
        delta = df.sub(mean_per_model, axis=0)
        # Shorten both model (rows) and judge (columns) labels in bias table
        if short_mode:
            delta.index = [get_display_name(m) for m in delta.index]
            delta.columns = [get_display_name(j) for j in delta.columns]

        # Heatmap of judge-model bias (Δ)
        # Compute the maximum absolute deviation across all (judge, model) cells:
        #   1. delta.abs()           -> DataFrame of absolute Δ values
        #   2. .max() (first call)  -> Series of max abs value per column (judge)
        #   3. .max() (second call) -> single largest abs Δ across all judges and models
        # This makes sure the heatmap is symmetrically scaled to the largest absolute deviation
        vlim = delta.abs().max().max()
        plt.figure(figsize=(max(10, len(delta.columns) * 1.2), max(6, len(delta.index) * 0.3)))
        # Use a more distinctive diverging palette and annotate each cell with Δ value
        cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
        sns.heatmap(
            delta,
            cmap=cmap,
            center=0,
            vmin=-vlim,
            vmax=vlim,
            annot=True,
            fmt=".3f",
            annot_kws={"fontsize": 10, "weight": "bold"},
            linewidths=0.5,
            cbar_kws={},
        )
        # Make x-axis labels horizontal and bold
        plt.xticks(rotation=0, ha="center", fontsize=10, weight="bold")
        plt.yticks(fontsize=10, weight="bold")
        plt.tight_layout()
        out_bias_heatmap = spec_out_dir / "judge_model_bias_heatmap.png"
        plt.savefig(out_bias_heatmap, dpi=300)
        plt.close()
        print(f"Saved judge-model bias heatmap to {out_bias_heatmap}")

        # Per-judge bias bar charts
        for j in delta.columns:
            sorted_vals = delta[j].sort_values()
            plt.figure(figsize=(8, max(4, len(sorted_vals) * 0.3)))
            plt.barh(
                [get_display_name(m) for m in sorted_vals.index],
                sorted_vals.values,
                color="skyblue",
            )
            plt.axvline(0, color="k")
            plt.title(f"{j} bias vs other judges: Spec={spec}")
            plt.xlabel("Δ compliance")
            plt.tight_layout()
            out_bar = spec_out_dir / f"bias_bar_{j}.png"
            plt.savefig(out_bar, dpi=300)
            plt.close()
            print(f"Saved per-judge bias bar chart to {out_bar}")

        # Summary of bias variability per judge
        stds = delta.std(axis=0)
        plt.figure(figsize=(6, 4))
        plt.bar(stds.index, stds.values, color="skyblue")
        plt.title(f"Judge bias variability (Std Δ): Spec={spec}")
        plt.xlabel("Judge")
        plt.ylabel("Std Δ compliance")
        # Rotate and shrink x-axis labels for readability
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        out_std = spec_out_dir / "bias_std_delta.png"
        plt.savefig(out_std, dpi=300)
        plt.close()
        print(f"Saved judge bias variability plot to {out_std}")


def create_aggregate_bias_heatmap(config: dict, number_size_scale=2.0):
    """Create an aggregate heatmap combining bias data from all specs.
    
    Args:
        config: Configuration dictionary
        number_size_scale: Scale factor for the size of numbers in cells (default=2.0)
    """
    base_dir = Path(config["base_dir"])
    output_dir = Path(config["output_dir"])
    specs = ["openai", "google", "anthropic"]  # Fixed order for specs
    
    # Create total directory
    total_dir = output_dir / "total"
    total_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all delta DataFrames
    all_deltas = {}
    for spec in specs:
        spec_dir = base_dir / f"spec_{spec}"
        if not spec_dir.is_dir():
            print(f"Warning: spec directory not found: {spec_dir}")
            continue
            
        judges = config.get("judges", [])
        spec_scores = {}
        
        for judge in judges:
            judge_dir = spec_dir / f"judge_{judge}"
            if not judge_dir.is_dir():
                continue
                
            scores = {}
            for model_dir in sorted(judge_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                results_dir = model_dir / "results"
                if not results_dir.is_dir():
                    continue
                    
                total, compliant = 0, 0
                for json_file in results_dir.glob("*.json"):
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                        for entry in data.get("results", []):
                            total += 1
                            if entry.get("compliant"):
                                compliant += 1
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")
                
                score = compliant / total if total > 0 else 0.0
                scores[model_dir.name] = score
            
            if scores:
                spec_scores[judge] = scores
        
        if spec_scores:
            df = pd.DataFrame(spec_scores)
            mean_per_model = df.mean(axis=1)
            delta = df.sub(mean_per_model, axis=0)
            
            # Shorten both model (rows) and judge (columns) labels
            if config.get("short_mode", False):
                delta.index = [SHORT_NAME_MAP.get(m, clean_model_name(m)) for m in delta.index]
                delta.columns = [SHORT_NAME_MAP.get(j, clean_model_name(j)) for j in delta.columns]
            
            all_deltas[spec] = delta
    
    if not all_deltas:
        print("No data found for aggregate heatmap")
        return
        
    # Create combined figure with horizontal layout
    fig = plt.figure(figsize=(20, 6))  # Adjusted figure size for horizontal layout
    
    # Calculate the maximum absolute value across all specs for consistent color scaling
    max_abs_val = max(df.abs().max().max() for df in all_deltas.values())
    
    # Create a subplot for each spec horizontally
    for idx, (spec, delta) in enumerate(all_deltas.items(), 1):
        ax = plt.subplot(1, 3, idx)  # Changed from 3,1 to 1,3 for horizontal layout
        
        # Create heatmap
        sns.heatmap(
            delta,
            cmap=sns.diverging_palette(240, 10, n=9, as_cmap=True),
            center=0,
            vmin=-max_abs_val,
            vmax=max_abs_val,
            annot=True,
            fmt=".3f",
            annot_kws={"fontsize": 10, "weight": "bold"},
            linewidths=0.5,
            cbar=True if idx == len(all_deltas) else False,  # Only show colorbar for last spec
            cbar_kws={} if idx == len(all_deltas) else {"spacing": "proportional"},
            ax=ax
        )
        
        # Customize appearance
        ax.set_title(f"{spec.upper()} Specification", pad=10, fontsize=12, weight="bold")
        ax.set_xlabel("Judges" if idx == 2 else "")  # Only show xlabel for middle plot
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10, weight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, weight='bold')
    
    plt.tight_layout()
    out_file = total_dir / "aggregate_judge_model_bias_heatmap.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregate bias heatmap to {out_file}")

    # Create organization-spec bias matrix
    # Map judges to their organizations
    judge_to_org = {
        "gpt-4.1": "openai",
        "claude-3-7-sonnet": "anthropic",
        "gemini-2.0-flash": "google"
    }
    orgs = ["openai", "anthropic", "google"]
    specs = ["openai", "anthropic", "google"]
    
    # Initialize the 3x3 matrix for organization-spec biases
    org_spec_bias = np.zeros((3, 3))
    
    # Calculate biases for each organization-spec pair
    for i, judge_org in enumerate(orgs):
        for j, spec in enumerate(specs):
            if spec in all_deltas:
                delta = all_deltas[spec]
                # Get the judge for this organization
                judge = next(j for j, o in judge_to_org.items() if o == judge_org)
                if judge in delta.columns:
                    # Calculate mean bias for this judge across all models for this spec
                    org_spec_bias[i, j] = delta[judge].mean()
    
    # Create figure for organization-spec bias matrix
    plt.figure(figsize=(12, 10))
    
    # Calculate max absolute value for symmetric color scaling
    max_bias = np.abs(org_spec_bias).max()
    
    # Create heatmap with larger numbers
    sns.heatmap(
        pd.DataFrame(
            org_spec_bias,
            index=[org.upper() for org in orgs],
            columns=[f"{spec.upper()} Spec" for spec in specs]
        ),
        cmap=sns.diverging_palette(240, 10, n=9, as_cmap=True),
        center=0,
        vmin=-max_bias,
        vmax=max_bias,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": int(12 * number_size_scale), "weight": "bold"},  # Scaled font size
        linewidths=1,
        square=True,
        # cbar_kws={"label": "Mean Judge Bias"}
    )
    
    # Make tick labels horizontal, bold and larger
    plt.xticks(rotation=0, ha='center', fontsize=int(14 * number_size_scale), weight='bold')
    plt.yticks(rotation=0, fontsize=int(14 * number_size_scale), weight='bold')
    
    plt.tight_layout()
    out_file_total = total_dir / "aggregate_total_judge_model_bias_heatmap.png"
    plt.savefig(out_file_total, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved organization-spec bias matrix to {out_file_total}")


def main():
    parser = argparse.ArgumentParser(description="Analyze batch binary compliance data.")
    parser.add_argument(
        "--config",
        type=str,
        default="analysis/configs/batch_binary_compliance_configs/batch_binary_compliance_config.yaml",
        help="Path to YAML config file for batch binary compliance analysis",
    )
    parser.add_argument(
        "--number-size-scale",
        type=float,
        default=2.0,
        help="Scale factor for size of numbers in aggregate heatmap (default: 2.0)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config file not found: {config_path}")
        return
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sns.set_theme(style="whitegrid")
    analyze_batch_compliance(config)
    # Add number size scale parameter to create_aggregate_bias_heatmap call
    create_aggregate_bias_heatmap(config, args.number_size_scale)


if __name__ == "__main__":
    main()
