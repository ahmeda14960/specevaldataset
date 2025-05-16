#!/usr/bin/env python3
"""Analyze binary compliance for top models of each org across multiple specs and judges.

Reads JSON results under base_dir/spec_{spec}/judge_{judge}/{model_folder}/results/*.json,
computes compliance fractions per (model, spec, judge), then averages across judges and
plots a 6×3 heatmap of average compliance (models × specs). Also saves per-judge and
average compliance CSVs.
"""

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from speceval.utils.parsing import (
    ALL_KNOWN_MODEL_NAMES,
    GPT_4O, GPT_4O_MINI, GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO,
    CLAUDE_3_7_SONNET, CLAUDE_3_5_SONNET, CLAUDE_3_5_HAIKU,
    GEMINI_2_0_FLASH, GEMINI_1_5_FLASH,
    DEEPSEEK_V3,
    QWEN_235B_FP8, QWEN_2_5_72B_TURBO, QWEN_2_72B_INSTRUCT,
    LLAMA_4_MAVERICK_17B, LLAMA_3_1_405B_TURBO,
)

# Available color palettes (uncomment one to use):
# CMAP = 'cividis'       # Colorblind-friendly (default)
# CMAP = 'viridis'       # Perceptually uniform
# CMAP = 'plasma'
# CMAP = 'magma'
# CMAP = 'inferno'
# CMAP = 'rocket'        # Seaborn
# CMAP = 'mako'
# CMAP = 'flare'
# CMAP = 'crest'

# Default colormap
CMAP = 'viridis'

# Mapping of org keys to their known model sets
ORG_MODELS = {
    "openai": {GPT_4O, GPT_4O_MINI, GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO},
    "anthropic": {CLAUDE_3_7_SONNET, CLAUDE_3_5_SONNET, CLAUDE_3_5_HAIKU},
    "google": {GEMINI_2_0_FLASH, GEMINI_1_5_FLASH},
    "deepseek": {DEEPSEEK_V3},
    "qwen": {QWEN_235B_FP8, QWEN_2_5_72B_TURBO, QWEN_2_72B_INSTRUCT},
    "meta_llama": {LLAMA_4_MAVERICK_17B, LLAMA_3_1_405B_TURBO},
}
# Fixed org order for rows
ORG_KEYS = ["openai", "anthropic", "google", "deepseek", "qwen", "meta_llama"]


def validate_top_models(config: dict):
    """Ensure each top_<org>_model is a known model and belongs to its org."""
    top_models = []
    for org in ORG_KEYS:
        key = f"top_{org}_model"
        if key not in config:
            raise ValueError(f"Missing config entry: {key}")
        model = config[key]
        if model not in ALL_KNOWN_MODEL_NAMES:
            raise ValueError(f"Model '{model}' is not a known model name.")
        if model not in ORG_MODELS[org]:
            raise ValueError(f"Model '{model}' is not valid for org '{org}'.")
        top_models.append(model)
    return top_models


def load_and_compute(base_dir: Path, specs: list, judges: list, top_models: list):
    """Load JSON results and compute per-judge compliance fractions."""
    records = []

    for spec in specs:
        for judge in judges:
            for model in top_models:
                # sanitize folder name (replace '/' with '-')
                folder = model.replace("/", "-")
                results_dir = base_dir / f"spec_{spec}" / f"judge_{judge}" / folder / "results"
                if not results_dir.is_dir():
                    print(f"Warning: missing results directory for {model} (spec={spec}, judge={judge}): {results_dir}")
                    continue

                total, compliant = 0, 0
                for jf in results_dir.glob("*.json"):
                    try:
                        with open(jf, 'r') as f:
                            data = json.load(f)
                    except Exception as e:
                        print(f"Error reading {jf}: {e}")
                        continue
                    for entry in data.get("results", []):
                        total += 1
                        if entry.get("compliant"):
                            compliant += 1
                comp_frac = compliant / total if total > 0 else np.nan
                records.append({
                    "model": model,
                    "spec": spec,
                    "judge": judge,
                    "compliance": comp_frac,
                })
    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(description="Analyze binary compliance in diagonal (top-models × specs).")
    parser.add_argument(
        "--config", type=str,
        default="analysis/configs/batch_binary_compliance_configs/batch_binary_compliance_diagonal_config.yaml",
        help="Path to diagonal compliance YAML config"
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    config = yaml.safe_load(cfg_path.read_text())

    base_dir = Path(config["base_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = config.get("specs", [])
    judges = config.get("judges", [])

    # Validate top models and get them in fixed order
    top_models = validate_top_models(config)

    # Compute per-judge compliance
    per_judge_df = load_and_compute(base_dir, specs, judges, top_models)
    per_judge_csv = output_dir / "compliance_per_judge.csv"
    per_judge_df.to_csv(per_judge_csv, index=False)
    print(f"Saved per-judge compliance to {per_judge_csv}")

    # Compute average compliance per model × spec
    avg_df = (
        per_judge_df
        .groupby(["model", "spec"])["compliance"]
        .mean()
        .unstack(level="spec")
    )
    # Ensure row and column order
    avg_df = avg_df.reindex(index=top_models, columns=specs)
    avg_csv = output_dir / "average_compliance.csv"
    avg_df.to_csv(avg_csv)
    print(f"Saved average compliance to {avg_csv}")

    # Plot heatmap with spaced layout and shortened model names
    # Mapping for display names
    DISPLAY_NAMES = {
        "gpt-4.1-2025-04-14": "gpt-4.1",
        "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
        "gemini-2.0-flash-001": "gemini-2.0-flash",
        "deepseek-ai/DeepSeek-V3": "DeepSeek-V3",
        "Qwen/Qwen3-235B-A22B-fp8-tput": "Qwen3-235B",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick",
    }
    display_df = avg_df.copy()
    display_df.index = [DISPLAY_NAMES.get(m, m) for m in display_df.index]

    # Plot the heatmap transposed (specs on y-axis, models on x-axis) with a colorblind-friendly colormap
    fig, ax = plt.subplots(figsize=(len(top_models) * 2.0, len(specs) * 1.2))
    sns.heatmap(
        display_df.T,
        annot=True,
        fmt=".2f",
        cmap=CMAP,
        # cbar_kws={"label": "Avg Compliance"},
        linewidths=1,
        linecolor="white",
        annot_kws={"fontsize": 12},
        ax=ax,
    )
    # Remove axis labels and title
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # Adjust tick label sizes and ensure horizontal orientation
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=14, rotation=0)
    
    # Make tick labels bold
    plt.setp(ax.get_xticklabels(), weight='bold')
    plt.setp(ax.get_yticklabels(), weight='bold')
    
    # Add a bit more figure size to accommodate larger text
    fig.set_size_inches(len(top_models) * 2.2, len(specs) * 1.3)
    
    # Tighten layout
    plt.tight_layout(pad=1.5)
    heatmap_file = output_dir / "average_compliance_heatmap.png"
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Saved heatmap to {heatmap_file}")


if __name__ == "__main__":
    main() 