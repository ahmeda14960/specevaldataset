# Batch Binary Compliance Analysis

This document describes the `analyze_batch_binary_compliance.py` script and explains how to interpret each of the generated plots.

---

## Overview

The `analyze_batch_binary_compliance.py` script processes raw JSON compliance judgments for multiple models, specifications, and judges. For each combination of spec and judge, it:

1. Reads all JSON result files under:
   ```
   {base_dir}/spec_{spec}/judge_{judge}/{model}/results/*.json
   ```
2. Computes per‐model metrics:
   - **Compliance Fraction**: (# compliant responses) / (total responses)
   - **Average Response Length**: mean output length in characters
   - **Per‐Statement Counts** to build a per‐statement compliance heatmap.
3. Generates three base plots per (spec, judge):
   - A bar plot of compliance fraction per model.
   - A scatter of compliance vs. average response length.
   - A heatmap of per‐statement compliance rates per model.
4. Aggregates across judges to produce:
   - A bar plot of each model's **average compliance across judges**, with both standard‐error and 95% confidence‐interval error bars, plus rank labels.
   - **Judge bias diagnostics** showing how each judge deviates from the mean on each model.

All plots are saved under:
```
{output_dir}/spec_{spec}/
```

---

## Plots and Interpretation

### 1. Batch Binary Compliance Bar Plot
- **Filename:** `batch_binary_compliance_{judge}.png`
- **What it shows:** The compliance fraction of each model under a single judge for a given spec. Models are sorted by descending compliance.
- **How to read:** Taller bars indicate models that follow the spec better under that judge. Annotated percentages show the exact compliance fraction.

### 2. Compliance vs. Average Response Length Scatter
- **Filename:** `batch_binary_compliance_vs_avg_length_{judge}.png`
- **What it shows:** Each model plotted by its average output length (x‐axis) vs. compliance fraction (y‐axis).
- **How to read:** Identify trade‐offs—models that are both concise and highly compliant appear in the upper‐left quadrant. Annotations show model names.

### 3. Per‐Statement Compliance Heatmap
- **Filename:** `batch_binary_compliance_per_statement_{judge}.png`
- **What it shows:** Rows are individual problem statements, columns are models; each cell is the fraction of compliant responses for that statement.
- **How to read:** Dark blue rows (low values) flag statements that all models struggle with. Models with consistently lighter columns perform better across statements.

### 4. Average Model Compliance Across Judges
- **Filename:** `avg_model_compliance_across_judges.png`
- **What it shows:** Each model's mean compliance across all judges for the current spec, with black error bars for standard error (SE) and red error bars for the 95% t‐interval. Bold numbers inside bars show each model's rank (1 = best).
- **How to read:** Compare models on a level playing field (all judges). Narrow error bars indicate consistent performance across judges. Rank labels help quickly identify the top models.

### 5. Judge–Model Bias Heatmap
- **Filename:** `judge_model_bias_heatmap.png`
- **What it shows:** A blue–white–red heatmap of Δ<sub>J,M</sub> = (Judge J's score for Model M) – (mean score of Model M across judges).
- **How to read:**
  - **Red cells (Δ≫0):** Judge J is more lenient on Model M than peers.
  - **Blue cells (Δ≪0):** Judge J is harsher on Model M.
  - **White cells (Δ≈0):** Judge J's evaluation matches the average.

### 6. Per‐Judge Bias Bar Charts
- **Filename:** `bias_bar_{judge}.png`
- **What it shows:** For each judge, a horizontal bar chart of Δ<sub>J,M</sub> sorted from most negative to most positive.
- **How to read:** The longest right‐hand bars show which models this judge scores higher than average; the longest left‐hand bars show which models are scored lower.

### 7. Judge Bias Variability
- **Filename:** `bias_std_delta.png`
- **What it shows:** A bar chart of the standard deviation of Δ<sub>J,M</sub> for each judge.
- **How to read:** Judges with larger Std(Δ) have more model‐specific leniency or strictness—they "favor" or "penalize" some models more than others.

---

## How to Run

```bash
python analysis/analyze_batch_binary_compliance.py --config path/to/config.yaml
```

Ensure your config YAML lists the correct `base_dir`, `output_dir`, `specs`, and `judges`. The script will create all directories and plots automatically.

---

By following these visualizations, you can answer questions like:
- Which model is most compliant under each judge?
- Are some models concise yet compliant?
- Which statements cause universal failures?
- Do certain judges consistently favor or penalize specific models?
- Which judge is the strictest or most variable in their scoring?
