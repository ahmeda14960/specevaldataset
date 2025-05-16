#!/usr/bin/env python3
"""Batch binary compliance analysis per organization (Anthropic, Google, OpenAI)."""

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
import importlib
from scipy.stats import t

from speceval.utils.parsing import extract_model_name_from_path

# Global configuration
NUMBER_SIZE_SCALE = 2.0  # Adjust this value to change the size of numbers in all plots

CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"


GPT_4O = "gpt-4o-2024-11-20"
GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
GPT_4_1 = "gpt-4.1-2025-04-14"
GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"


GEMINI_2_0_FLASH = "gemini-2.0-flash-001"
GEMINI_1_5_FLASH = "gemini-1.5-pro"


def clean_model_name(model_name: str) -> str:
    """Clean up model name for display."""
    name = re.sub(r"^[a-zA-Z]+-[a-zA-Z]+[-/]", "", model_name)
    name = name.replace("-Instruct", "").replace("-instruct", "").replace("-Turbo", "")
    if len(name) > 25:
        name = name[:22] + "..."
    return name


def analyze_per_org(config: dict):
    """Analyze batch binary compliance per organization."""
    base_dir = Path(config["base_dir"])
    output_dir = Path(config["output_dir"])
    judges = config.get("judges", [])

    output_dir.mkdir(parents=True, exist_ok=True)
    specs = ["anthropic", "google", "openai"]

    sns.set_theme(style="whitegrid")

    for spec in specs:
        # Prepare to collect raw compliance per judge for average summary
        spec_scores = {}
        # Import the module to gather this org's model constants
        try:
            mod = importlib.import_module(f"speceval.models.{spec}")
        except ImportError:
            print(f"Warning: could not import speceval.models.{spec}")
            continue
        org_models = [v for k, v in mod.__dict__.items() if k.isupper() and isinstance(v, str)]

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
            per_stmt_counts = {}

            for model_dir in sorted(judge_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                try:
                    # Only consider the model directory name to avoid ambiguity with judge folder
                    name = extract_model_name_from_path(Path(model_dir.name))
                except ValueError:
                    continue
                if name not in org_models:
                    continue

                results_dir = model_dir / "results"
                if not results_dir.is_dir():
                    print(f"Warning: results directory not found for model {model_dir.name}")
                    continue

                total = 0
                compliant = 0
                length_total = 0
                length_count = 0
                per_model_counts = {}

                for json_file in results_dir.glob("*.json"):
                    stmt = json_file.stem
                    per_model_counts.setdefault(stmt, {"compliant": 0, "total": 0})
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                        for entry in data.get("results", []):
                            total += 1
                            per_model_counts[stmt]["total"] += 1
                            if entry.get("compliant"):
                                compliant += 1
                                per_model_counts[stmt]["compliant"] += 1
                            output = entry.get("output", "")
                            length_total += len(output)
                            length_count += 1
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

                score = compliant / total if total > 0 else 0.0
                avg_len = length_total / length_count if length_count > 0 else 0.0
                scores[name] = score
                avg_lengths[name] = avg_len
                per_stmt_counts[name] = per_model_counts
                print(
                    f"Spec={spec}, Judge={judge}, Model={name}, Compliance={score:.2f}, AvgLen={avg_len:.1f}"
                )

            if not scores:
                print(f"No models found for spec={spec}, judge={judge}")
                continue

            # Record raw compliance fractions for this judge
            spec_scores[judge] = scores

            # Bar plot: compliance per model
            models = list(scores.keys())
            values = [scores[m] for m in models]
            display_names = [clean_model_name(m) for m in models]
            idx = np.argsort(values)[::-1]
            sorted_names = [display_names[i] for i in idx]
            sorted_values = [values[i] for i in idx]

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=sorted_names, y=sorted_values)
            # Annotate bars with compliance values centered inside
            for i, v in enumerate(sorted_values):
                ax.text(i, v/2, f"{v:.2f}", ha="center", va="center", 
                       fontsize=int(12 * NUMBER_SIZE_SCALE), fontweight="bold")
            # Formatting
            plt.xlabel("Model", fontsize=int(10 * NUMBER_SIZE_SCALE))
            plt.ylabel("")  # remove y-axis label
            plt.xticks(range(len(sorted_names)), sorted_names, rotation=0, 
                      ha="center", fontsize=int(10 * NUMBER_SIZE_SCALE))
            plt.ylim(0, 1.0)
            plt.tight_layout()
            spec_out = output_dir / f"spec_{spec}"
            spec_out.mkdir(parents=True, exist_ok=True)
            out_file = spec_out / f"batch_binary_compliance_{judge}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved plot to {out_file}")

            # Scatter plot: compliance vs avg length
            plt.figure(figsize=(8, 6))
            for m, v in scores.items():
                plt.scatter(avg_lengths[m], v)
                plt.annotate(
                    clean_model_name(m),
                    (avg_lengths[m], v),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                )
            plt.xlabel("Average Response Length (chars)")
            plt.ylabel("Compliance Fraction")
            plt.title(f"Compliance vs Avg Length: Spec={spec} | Judge={judge}")
            plt.tight_layout()
            out2 = spec_out / f"batch_binary_compliance_vs_avg_length_{judge}.png"
            plt.savefig(out2, dpi=300)
            plt.close()
            print(f"Saved compliance vs avg length to {out2}")

            # Per-statement compliance heatmap
            all_statements = sorted(
                {stmt for counts in per_stmt_counts.values() for stmt in counts}
            )
            df = pd.DataFrame(index=all_statements, columns=scores.keys(), dtype=float)
            for m, counts in per_stmt_counts.items():
                for stmt in all_statements:
                    tot = counts.get(stmt, {}).get("total", 0)
                    comp = counts.get(stmt, {}).get("compliant", 0)
                    df.at[stmt, m] = comp / tot if tot > 0 else np.nan
            df.columns = [clean_model_name(m) for m in df.columns]
            plt.figure(figsize=(max(10, len(df.columns) * 0.8), max(6, len(df.index) * 0.3)))
            sns.heatmap(
                df, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Compliance Fraction"}
            )
            plt.title(f"Per-Statement Compliance: Spec={spec} | Judge={judge}")
            plt.xlabel("Model")
            plt.ylabel("Statement ID")
            if len(df.columns) > 3:
                plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out3 = spec_out / f"batch_binary_compliance_per_statement_{judge}.png"
            plt.savefig(out3, dpi=300)
            plt.close()
            print(f"Saved per-statement heatmap to {out3}")

        # After all judges for this spec, compute and plot average compliance per model across judges
        if spec_scores:
            spec_out = output_dir / f"spec_{spec}"
            spec_out.mkdir(parents=True, exist_ok=True)
            # Build DataFrame: rows = models, columns = judges
            df_scores = pd.DataFrame(spec_scores)
            if spec == "openai":
                # Define GPT tiers
                tiers = {
                    "gpt": [GPT_4_1, GPT_4O],
                    "gpt_mini": [GPT_4O_MINI, GPT_4_1_MINI],
                }
                for tier_name, models_in_tier in tiers.items():
                    df_tier = df_scores.loc[df_scores.index.intersection(models_in_tier)]
                    if df_tier.empty:
                        print(f"No models found for spec={spec}, tier={tier_name}")
                        continue
                    # Compute mean, SE, CI
                    mean_model = df_tier.mean(axis=1)
                    sd_model = df_tier.std(axis=1, ddof=1)
                    se_model = sd_model / np.sqrt(df_tier.shape[1])
                    alpha = 0.05
                    ci_model = se_model * t.ppf(1 - alpha / 2, df=df_tier.shape[1] - 1)
                    order = mean_model.sort_values(ascending=False).index.tolist()
                    means = mean_model[order].values
                    ses = se_model[order].values
                    cis = ci_model[order].values
                    x = np.arange(len(order))
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(x, means, color="skyblue", width=0.6, label="Mean")
                    plt.errorbar(x, means, yerr=ses, fmt="none", ecolor="black", capsize=5, label="SE")
                    plt.errorbar(x, means, yerr=cis, fmt="none", ecolor="red", capsize=5, label="95% CI")
                    # Annotate bars with compliance values centered inside
                    for i, v in enumerate(means):
                        plt.text(i, v/2, f"{v:.2f}", ha="center", va="center", 
                               fontsize=int(12 * NUMBER_SIZE_SCALE), fontweight="bold")
                    plt.xticks(x, [clean_model_name(m) for m in order], rotation=0, 
                             ha="center", fontsize=int(10 * NUMBER_SIZE_SCALE))
                    plt.xlabel("Model", fontsize=int(10 * NUMBER_SIZE_SCALE))
                    plt.ylabel("")  # remove y-axis label
                    plt.ylim(0, 1.0)
                    plt.legend(fontsize=int(10 * NUMBER_SIZE_SCALE))
                    plt.tight_layout()
                    out_file = spec_out / f"avg_model_compliance_{tier_name}.png"
                    plt.savefig(out_file, dpi=300)
                    plt.close()
                    print(f"Saved avg model compliance across judges for tier {tier_name} to {out_file}")
            elif spec == "anthropic":
                # Only compare Sonnet 3.7 and 3.5
                anth_models = [CLAUDE_3_7_SONNET, CLAUDE_3_5_SONNET]
                df_anth = df_scores.loc[df_scores.index.intersection(anth_models)]
                if df_anth.empty:
                    print(f"No Sonnet models found for spec={spec}")
                else:
                    mean_model = df_anth.mean(axis=1)
                    sd_model = df_anth.std(axis=1, ddof=1)
                    se_model = sd_model / np.sqrt(df_anth.shape[1])
                    alpha = 0.05
                    ci_model = se_model * t.ppf(1 - alpha / 2, df=df_anth.shape[1] - 1)
                    order = mean_model.sort_values(ascending=False).index.tolist()
                    means = mean_model[order].values
                    ses = se_model[order].values
                    cis = ci_model[order].values
                    x = np.arange(len(order))
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(x, means, color="skyblue", width=0.6, label="Mean")
                    plt.errorbar(x, means, yerr=ses, fmt="none", ecolor="black", capsize=5, label="SE")
                    plt.errorbar(x, means, yerr=cis, fmt="none", ecolor="red", capsize=5, label="95% CI")
                    # Annotate bars with compliance values centered inside
                    for i, v in enumerate(means):
                        plt.text(i, v/2, f"{v:.2f}", ha="center", va="center", 
                               fontsize=int(12 * NUMBER_SIZE_SCALE), fontweight="bold")
                    plt.xticks(x, [clean_model_name(m) for m in order], rotation=0, 
                             ha="center", fontsize=int(10 * NUMBER_SIZE_SCALE))
                    plt.xlabel("Model", fontsize=int(10 * NUMBER_SIZE_SCALE))
                    plt.ylabel("")  # remove y-axis label
                    plt.ylim(0, 1.0)
                    plt.legend(fontsize=int(10 * NUMBER_SIZE_SCALE))
                    plt.tight_layout()
                    out_file = spec_out / "avg_model_compliance_sonnet_comparison.png"
                    plt.savefig(out_file, dpi=300)
                    plt.close()
                    print(f"Saved avg Sonnet compliance across judges to {out_file}")
            else:
                # Default for Google and other specs
                mean_model = df_scores.mean(axis=1)
                sd_model = df_scores.std(axis=1, ddof=1)
                se_model = sd_model / np.sqrt(df_scores.shape[1])
                alpha = 0.05
                ci_model = se_model * t.ppf(1 - alpha / 2, df=df_scores.shape[1] - 1)
                order = mean_model.sort_values(ascending=False).index.tolist()
                means = mean_model[order].values
                ses = se_model[order].values
                cis = ci_model[order].values
                x = np.arange(len(order))
                plt.figure(figsize=(10, 6))
                bars = plt.bar(x, means, color="skyblue", width=0.6, label="Mean")
                plt.errorbar(x, means, yerr=ses, fmt="none", ecolor="black", capsize=5, label="SE")
                plt.errorbar(x, means, yerr=cis, fmt="none", ecolor="red", capsize=5, label="95% CI")
                # Annotate bars with compliance values centered inside
                for i, v in enumerate(means):
                    plt.text(i, v/2, f"{v:.2f}", ha="center", va="center", 
                           fontsize=int(12 * NUMBER_SIZE_SCALE), fontweight="bold")
                plt.xticks(x, [clean_model_name(m) for m in order], rotation=0, 
                         ha="center", fontsize=int(10 * NUMBER_SIZE_SCALE))
                plt.xlabel("Model", fontsize=int(10 * NUMBER_SIZE_SCALE))
                plt.ylabel("")  # remove y-axis label
                plt.ylim(0, 1.0)
                plt.legend(fontsize=int(10 * NUMBER_SIZE_SCALE))
                plt.tight_layout()
                out_avg = spec_out / "avg_model_compliance_across_judges.png"
                plt.savefig(out_avg, dpi=300)
                plt.close()
                print(f"Saved avg model compliance across judges plot to {out_avg}")


def main():
    parser = argparse.ArgumentParser(description="Analyze batch binary compliance per organization")
    parser.add_argument(
        "--config",
        type=str,
        default="analysis/configs/batch_binary_compliance_configs/batch_binary_compliance_per_org.yaml",
        help="Path to YAML config file for per-org batch binary compliance analysis",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config file not found: {config_path}")
        return
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    analyze_per_org(config)


if __name__ == "__main__":
    main()
