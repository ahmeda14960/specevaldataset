# Multi-Judge Binary Compliance Analysis

This directory contains combined analysis of binary compliance scores across multiple judges.

## Overview

The `combine_binary_compliance.py` script combines binary compliance scores from different judge models (Anthropic, Google, OpenAI, etc.) into a single dataset. It allows you to see how different models perform across various judge evaluations and identify patterns in compliance scores.

## Usage

```bash
# Run with default config file
python analysis/combine_binary_compliance.py

# Or specify a custom config file
python analysis/combine_binary_compliance.py --config path/to/your_config.yaml
```

## Configuration

Edit the `analysis/binary_compliance_config.yaml` file to specify:

1. CSV paths - List of paths to overall compliance CSV files from different judges
2. Output directory - Where to save combined results (default: `analysis/multi_judge_binary`)

Example configuration:
```yaml
csv_paths:
  - "analysis/binary_compliance_analysis/JUDGE_ANTHROPIC/overall_compliance_gpt-4.1-2025-04-14_3-7-sonnet-20250219.csv"
  - "analysis/binary_compliance_analysis/JUDGE_GOOGLE/overall_compliance_gpt-4.1-2025-04-14_gemini-2.0-flash-001.csv"
  - "analysis/binary_compliance_analysis/JUDGE_OPENAI/overall_compliance_gpt-4.1-2025-04-14_gpt-4.1-2025-04-14.csv"

output_dir: "analysis/multi_judge_binary"
```

## Output Files

The script generates the following output:

1. `combined_compliance_rankings.csv` - Combined data with:
   - Model names
   - Per-judge rankings and scores
   - Average rankings across judges

2. Visualizations:
   - `compliance_scores_heatmap.png` - Heatmap of scores by model and judge
   - `average_compliance_scores.png` - Bar chart of average scores across judges
   - `model_rankings_matrix.png` - Matrix showing how models rank for each judge
   - Individual judge bar charts (e.g., `anthropic_compliance_scores.png`)

## Interpretation

- Models are sorted by average rank across all judges
- Higher compliance scores indicate better adherence to specifications
- Visualizations help identify which models perform consistently across judges versus those with variable performance
