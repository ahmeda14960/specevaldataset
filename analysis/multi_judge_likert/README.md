# Multi-Judge Likert Compliance Analysis

This directory contains combined analysis of Likert scale compliance scores (1-5) across multiple judges.

## Overview

The `combine_likert_compliance.py` script combines Likert scale compliance scores from different judge models (Google, Anthropic, OpenAI, etc.) into a single dataset. It allows you to see how different models perform across various judge evaluations on a 1-5 Likert scale and identify patterns in compliance scores.

## Usage

```bash
# Run with default config file
python analysis/combine_likert_compliance.py

# Or specify a custom config file
python analysis/combine_likert_compliance.py --config path/to/your_config.yaml
```

## Configuration

Edit the `analysis/likert_compliance_config.yaml` file to specify:

1. CSV paths - List of paths to overall average Likert score CSV files from different judges
2. Output directory - Where to save combined results (default: `analysis/multi_judge_likert`)

Example configuration:
```yaml
csv_paths:
  - "analysis/likert_compliance_analysis/google_gemini-2.0-flash-001/overall_avg_score_gpt-4.1-2025-04-14.csv"
  - "analysis/likert_compliance_analysis/anthropic_claude-3-7-sonnet-20250219/overall_avg_score_gpt-4.1-2025-04-14.csv"
  - "analysis/likert_compliance_analysis/openai_gpt-4.1-2025-04-14/overall_avg_score_gpt-4.1-2025-04-14.csv"

output_dir: "analysis/multi_judge_likert"
```

## Output Files

The script generates the following output:

1. `combined_likert_rankings.csv` - Combined data with:
   - Model names
   - Per-judge rankings and scores
   - Average rankings across judges

2. Visualizations:
   - `likert_scores_heatmap.png` - Heatmap of Likert scores by model and judge
   - `average_likert_scores.png` - Bar chart of average Likert scores across judges
   - `model_likert_rankings_matrix.png` - Matrix showing how models rank for each judge
   - Individual judge bar charts (e.g., `google_gemini-2.0-flash-001_likert_scores.png`)

## Interpretation

- Models are sorted by average rank across all judges
- Higher Likert scores (closer to 5) indicate better adherence to specifications
- The Likert scale (1-5) provides more granular information than binary compliance
- Visualizations help identify which models perform consistently across judges versus those with variable performance
