# SpecEval Compliance Analysis

This directory contains tools to analyze compliance data from the SpecEval framework.

## Binary Compliance Analysis

### Running the Binary Compliance Analysis

To run the compliance analysis script:

```bash
# Make sure you have the required dependencies
pip install numpy pandas matplotlib seaborn pyyaml

# Activate your environment (if using micromamba)
micromamba activate aispec

# 1. Edit analysis_config.yaml to specify either:
#    - The organization directory to analyze (e.g., JUDGE_GOOGLE)
#    - Or specific judgment directories within data/judgements/

# 2. Run the analysis script
#    It will use analysis_config.yaml by default
./analyze_compliance.py

#    Or specify a different config file
# ./analyze_compliance.py --config path/to/your_config.yaml
```

### Configuration File (`analysis_config.yaml`)

The analysis script provides two ways to specify which judgment data to analyze:

#### Option 1: Organization-based Structure

When using the organization structure, specify the `judgment_org` field in the config file:

```yaml
# Option 1: Specify an organization directory to analyze
judgment_org: "JUDGE_GOOGLE"
```

The script will:
1. Automatically discover all subdirectories within `data/judgements/JUDGE_GOOGLE/`
2. Support both directory formats:
   - `data/judgements/JUDGE_GOOGLE/evaluatorxcandidate/`
   - `data/judgements/JUDGE_GOOGLE/evaluatorxcandidatexjudge/`
3. Extract evaluator, candidate, and judge model information from the metadata in JSON files
4. Output results to `analysis/binary_compliance_analysis/JUDGE_GOOGLE/`

#### Option 2: Traditional Structure

If the `judgment_org` field is not specified, the script uses a list of specific judgment directories:

```yaml
# Option 2: List specific judgment directories to analyze
judgement_directories:
  - gpt-4.1-2025-04-14xgpt-4.1-2025-04-14xgpt-4.1-2025-04-14
  - gpt-4.1-2025-04-14xclaude-3-7-sonnet-20250219xgpt-4.1-2025-04-14
  # - other_evalxother_candidatexother_judge
```

With this option:
- The script expects directories to follow the format `evaluatorxcandidatexjudge`
- Each string must be the exact name of a subdirectory within `data/judgements/`
- Results are output to `analysis/binary_compliance_analysis/`

### What the Binary Analysis Script Does

The script performs the following steps:

1. Reads the configuration file (`analysis_config.yaml` by default).
2. Groups models by common evaluator and judge combinations.
3. Calculates overall and per-statement compliance scores for each model within each group.
4. Generates visualizations:
   - Bar charts showing overall compliance scores for different models in a group.
   - Heatmaps showing per-statement compliance scores across statements and models.
   - Heatmaps showing scores for statements with the highest variance within a group.
5. Saves the calculated scores as CSV files.

### Binary Analysis Output Files

The script generates the following files:

For organization-based structure:
- `analysis/binary_compliance_analysis/JUDGE_GOOGLE/overall_compliance_[evaluator]_[judge].png`
- `analysis/binary_compliance_analysis/JUDGE_GOOGLE/per_statement_compliance_[evaluator]_[judge].png`
- `analysis/binary_compliance_analysis/JUDGE_GOOGLE/overall_compliance_[evaluator]_[judge].csv`
- `analysis/binary_compliance_analysis/JUDGE_GOOGLE/per_statement_compliance_[evaluator]_[judge].csv`

For traditional structure:
- `analysis/binary_compliance_analysis/overall_compliance_[evaluator]_[judge].png`
- `analysis/binary_compliance_analysis/per_statement_compliance_[evaluator]_[judge].png`
- `analysis/binary_compliance_analysis/overall_compliance_[evaluator]_[judge].csv`
- `analysis/binary_compliance_analysis/per_statement_compliance_[evaluator]_[judge].csv`

## Likert Compliance Analysis

### Running the Likert Compliance Analysis

To run the Likert compliance analysis script:

```bash
# Make sure you have the required dependencies
pip install numpy pandas matplotlib seaborn pyyaml

# Activate your environment (if using micromamba)
micromamba activate aispec

# 1. Edit likert_analysis_config.yaml to specify either:
#    - The full judge directory to analyze (e.g., JUDGE_google_gemini-2.0-flash-001)
#    - Or specific date/model directory paths within data/likert_judgements/

# 2. Run the analysis script
./analyze_likert_compliance.py

#    Or specify a different config file
# ./analyze_likert_compliance.py --config path/to/your_config.yaml
```

### Configuration File (`likert_analysis_config.yaml`)

The Likert analysis script provides two ways to specify which judgement data to analyze:

#### Option 1: Judge Directory Structure

When using the judge directory structure, specify the `judgment_judge_dir` field in the config file:

```yaml
# Option 1: Specify the full judge directory name
judgment_judge_dir: "JUDGE_google_gemini-2.0-flash-001"
```

The script will:
1. Automatically discover all date subdirectories within `data/likert_judgements/JUDGE_google_gemini-2.0-flash-001/`
2. Find all model directories in the format `evaluatorxcandidate` under each date directory
3. Group all models by evaluator and calculate average Likert scores
4. Output results to `analysis/likert_compliance_analysis/google_gemini-2.0-flash-001/`

#### Option 2: Manual Path Specification

If the `judgment_judge_dir` field is not specified, the script uses a list of specific paths:

```yaml
# Option 2: List specific relative paths to analyze
judgement_paths:
  - 04_28_2025/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14
  - 04_28_2025/gpt-4.1-2025-04-14xclaude-3-7-sonnet-20250219
  # - other_date/evaluatorxcandidate
```

With this option:
- Each string must be a relative path under `data/likert_judgements/`
- The format should be `[date]/[evaluator]x[candidate]`
- Results are output to `analysis/likert_compliance_analysis/`

### Likert Data Directory Structure

The Likert analysis supports a specific nested directory structure:

```
data/likert_judgements/JUDGE_provider_model/
└── MM_DD_YYYY/
    └── [evaluator]x[candidate]/
        └── [statement_id].json
```

For example:
```
data/likert_judgements/JUDGE_google_gemini-2.0-flash-001/
└── 04_28_2025/
    ├── gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/
    │   ├── be_creative.json
    │   └── be_efficient.json
    └── gpt-4.1-2025-04-14xclaude-3-7-sonnet-20250219/
        ├── be_creative.json
        └── be_efficient.json
```

### What the Likert Analysis Script Does

The script performs the following steps:

1. Reads the configuration file (`likert_analysis_config.yaml` by default).
2. Groups models by common evaluator.
3. Calculates average Likert scores (overall and per-statement) for each model.
4. Generates visualizations:
   - Bar charts showing overall average scores for different models.
   - Heatmaps showing per-statement average scores across statements and models.
   - Heatmaps showing scores for statements with the highest variance.
5. Saves the calculated scores as CSV files.

### Likert Analysis Output Files

The script generates the following files:

For judge directory structure:
- `analysis/likert_compliance_analysis/google_gemini-2.0-flash-001/overall_avg_score_[evaluator].png`
- `analysis/likert_compliance_analysis/google_gemini-2.0-flash-001/per_statement_avg_score_[evaluator].png`
- `analysis/likert_compliance_analysis/google_gemini-2.0-flash-001/statement_variance_avg_score_[evaluator].png`
- `analysis/likert_compliance_analysis/google_gemini-2.0-flash-001/overall_avg_score_[evaluator].csv`
- `analysis/likert_compliance_analysis/google_gemini-2.0-flash-001/per_statement_avg_score_[evaluator].csv`

For the traditional structure:
- `analysis/likert_compliance_analysis/overall_avg_score_[evaluator].png`
- `analysis/likert_compliance_analysis/per_statement_avg_score_[evaluator].png`
- `analysis/likert_compliance_analysis/statement_variance_avg_score_[evaluator].png`
- `analysis/likert_compliance_analysis/overall_avg_score_[evaluator].csv`
- `analysis/likert_compliance_analysis/per_statement_avg_score_[evaluator].csv`

## Example Visualizations

The overall compliance/score plots show how different models compare:

```
Overall Compliance Scores
Evaluator: gpt-4o | Judge: gpt-4o-mini
┌────────────────────────────────────────────────────┐
│                                                    │
│                                                    │
│  █████  █████  █████  █████  █████  █████  █████  │
│  █████  █████  █████  █████  █████  █████  █████  │
│  █████  █████  █████  █████  █████  █████  █████  │
│  █████  █████  █████  █████  █████  █████  █████  │
│  Model1  Model2 Model3 Model4 Model5 Model6 Model7 │
└────────────────────────────────────────────────────┘
```

The per-statement heatmaps show scores for each model across different statement IDs:

```
Per-Statement Compliance Scores
Evaluator: gpt-4o | Judge: gpt-4o-mini
┌────────────────────────────────────────────────────┐
│  Model1 ██████████████████████████████████████████ │
│  Model2 ██████████████████████████████████████████ │
│  Model3 ██████████████████████████████████████████ │
│  Model4 ██████████████████████████████████████████ │
│         stmt1 stmt2 stmt3 stmt4 stmt5 stmt6 stmt7  │
└────────────────────────────────────────────────────┘
```

## Ranking Results Analysis

This directory also includes tools to analyze ranking outputs generated by the SpecEval `RankingPipeline`.

### Script: `analyze_rankings.py`

- Reads per-statement JSON ranking results written under `data/rankings/<ranker_model>/<evaluator_model>/<ModelAVSModelB>/*.json`.
- Aggregates win/loss/tie counts for each model.
- If `all_pairs: true` in the config, it will correct for position bias by matching each pair with its reversed ordering and treating inconsistent comparisons as ties.
- If `all_pairs: false`, it tallies results directly (position bias may be present), and suggests enabling `all_pairs` for bias correction.

### Configuration File (`rankings_config.yaml`)

The script uses a YAML config (default `analysis/rankings_config.yaml`) with:

```yaml
# Name of the ranker model folder under data/rankings
ranker: <ranker_model_name>

# Name of the evaluator folder under data/rankings/<ranker>
# The evaluator is the model that generated the prompts/inputs for ranking
evaluator: <evaluator_model_name>

# List of candidate model names matching folders like <ModelAVSModelB>
models:
  - <candidate_model_1>
  - <candidate_model_2>
  # ...

# Boolean flag to correct for position bias
all_pairs: true  # or false

# Paths (relative to project root)
rankings_dir: data/rankings
output_dir: analysis/rankings_analysis
```

Field descriptions:
- **`ranker`**: The ranker model directory under `data/rankings`. The ranker is the model performing the A/B comparison.
- **`evaluator`**: The evaluator model directory under `data/rankings/<ranker>`. The evaluator generated the inputs/prompts used for the ranking comparisons.
- **`models`**: Candidate models to include in the analysis (these correspond to the `ModelA` and `ModelB` in the directory names).
- **`all_pairs`**: When `true`, applies consistency checks across reversed pairs (`A vs B` and `B vs A`) and treats mismatches as ties.
- **`rankings_dir`**: Base directory containing ranking outputs.
- **`output_dir`**: Base directory where analysis results will be saved.

### Usage

```bash
# Install dependencies
pip install pandas matplotlib pyyaml numpy

# Run the ranking analysis
python analyze_rankings.py --config analysis/configs/ranking_configs/rankings_config.yaml
```

After running, check `analysis/rankings_analysis/<ranker_model_name>/<evaluator_model_name>/` for:
- `ranking_summary.csv` — aggregated wins/losses/ties per model
- `wins_per_model.png` — bar chart of total wins per model

**Note:** If your ranking outputs use the standard (non-batched) directory layout under `data/rankings_google/...`, the `analyze_batch_rankings.py` tool will not find them. Instead, use the single-run analyzer:
```bash
python analysis/analyze_rankings.py --config analysis/configs/ranking_configs/gemini_ranker_openai_spec.yaml
```

### Batched Ranking Analysis: `analyze_batch_rankings.py`

This script processes outputs produced by the `BatchedRankingPipeline`.  It expects a directory tree of the form:

```
<base_dir>/spec_<spec>/judge_<judge>/<modelA>x<modelB>/results/*.json
```

Each JSON under `results/` must contain a top-level `"rankings"` array of objects with fields:
- `input` (unique input ID)
- `output_a`, `output_b` (the two model outputs)
- `rank` (1 if A wins, -1 if B wins, 0 for tie)

The script enforces this structure, then aggregates per-model win/loss/tie counts in one of two modes:

- **unique-pairs** (default): counts each ordered `modelA x modelB` run separately
- **all-pairs** (`--all_pairs`): requires both `A x B` and `B x A` directories, treats inconsistent pairs as ties

#### CLI Usage

```bash
# Unique-pairs analysis
python analysis/analyze_batch_rankings.py \
  --spec openai \
  --judge claude-3-7-sonnet-20250219 \
  --models gpt-4.1-2025-04-14 claude-3-5-haiku-20241022 claude-3-5-sonnet-20240620 \
  --base_dir data/batched_rankings \
  --output_dir analysis/batch_ranking_analysis

# All-pairs analysis (position-bias correction)
python analysis/analyze_batch_rankings.py \
  --spec openai \
  --judge claude-3-7-sonnet-20250219 \
  --models gpt-4.1-2025-04-14 claude-3-5-haiku-20241022 claude-3-5-sonnet-20240620 \
  --all_pairs \
  --base_dir data/batched_rankings \
  --output_dir analysis/batch_ranking_analysis
```

A sample YAML config is provided at:

```
analysis/configs/batch_ranking_configs/sample_batch_ranking_config.yaml
```

#### YAML-driven usage

```bash
python analysis/analyze_batch_rankings.py --config analysis/configs/batch_ranking_configs/sample_batch_ranking_config.yaml
```

Or use one of the specialized config files:
```bash
# Google spec (Gemini judge)
python analysis/analyze_batch_rankings.py --config analysis/configs/batch_ranking_configs/google_spec_gemini_ranker.yaml

# Anthropic spec (Gemini judge)
python analysis/analyze_batch_rankings.py --config analysis/configs/batch_ranking_configs/anthropic_spec_gemini_ranker.yaml

# OpenAI spec (Gemini judge)
python analysis/analyze_batch_rankings.py --config analysis/configs/batch_ranking_configs/openai_spec_gemini_ranker.yaml
```

### Position-Bias-Corrected Batched Ranking Analysis: `correct_position_bias_rankings.py`

This script mirrors your entire `data/batched_rankings` hierarchy under the given `--output_dir` and writes only the corrected JSON files there—**it does not overwrite your original files**.

```bash
python analysis/correct_position_bias_rankings.py \
  --base_dir data/batched_rankings \
  --output_dir data/batched_rankings_corrected
```

## Judge Agreement Analysis

This directory also includes tools to analyze agreement between different judge models for both binary compliance and Likert scale judgments.

### Binary Compliance Agreement Analysis: `correlate_binary_compliance.py`

This script analyzes agreement between different judge models for binary compliance judgments:

1. Groups judgments by the same evaluator, candidate, and statement combinations
2. Calculates Fleiss' Kappa to measure agreement across multiple judges
3. Computes pairwise agreement between judge models
4. Generates agreement matrices as heatmaps
5. Saves detailed CSV outputs for further analysis

The script offers three different analysis modes:

#### Default Mode
Calculates agreement for each unique evaluator-candidate-statement combination:
- Generates separate agreement metrics for each statement-model pair
- Best for understanding judge agreement on specific statements and models
- Results stored in `analysis/binary_correlation/default/`

#### Global Mode
Calculates a single agreement metric across all judgments:
- Combines all judgments across all evaluator-candidate-statement combinations
- Provides a single Fleiss' Kappa value representing overall judge agreement
- Creates a comprehensive agreement matrix for all judges
- Results stored in `analysis/binary_correlation/global/`

#### Model Mode
Calculates agreement for each evaluator-candidate combination by combining statements:
- Shows how consistently judges evaluate specific models across all statements
- Helpful for identifying models where judges have high or low agreement
- Results stored in `analysis/binary_correlation/model/`

#### Usage:

```bash
# Install required dependencies
pip install numpy pandas matplotlib seaborn pyyaml statsmodels

# Run in default mode (per-statement analysis)
./correlate_binary_compliance.py --mode default

# Run in global mode (single overall analysis)
./correlate_binary_compliance.py --mode global

# Run in model mode (per-model analysis)
./correlate_binary_compliance.py --mode model

# Run all modes at once
./correlate_binary_compliance.py --mode all

# Specify a custom config file with any mode
./correlate_binary_compliance.py --config path/to/your_config.yaml --mode model
```

For each mode, the script generates:
- CSV files with Fleiss' Kappa values
- Pairwise agreement matrices
- Visualizations of agreement as heatmaps
- Summary statistics

### Ranking Agreement Analysis: `correlate_ranking_compliance.py`

This script analyzes agreement between different **ranker models** based on their pairwise comparison results (A vs B rankings):

1. Reads detailed ranking results (JSON files) from `data/rankings`.
2. Groups comparisons by evaluator and the specific pair of candidate models being compared.
3. Calculates Fleiss' Kappa to measure agreement across multiple ranker models, using the ranking outcomes (-1 for B wins, 0 for tie, 1 for A wins) as the categories.
4. Computes pairwise agreement percentages between ranker models.
5. Generates agreement matrices as heatmaps.
6. Saves detailed CSV outputs for further analysis.

The script offers different analysis modes:

#### Global Mode
Calculates a single agreement metric across all comparisons made by the specified rankers for the chosen evaluator(s) and models:
- Combines all judgments across all evaluator-candidate_pair-input combinations.
- Provides a single Fleiss' Kappa value representing overall ranker agreement.
- Creates a comprehensive agreement matrix for all rankers.
- Results stored in `analysis/ranking_correlation/global/`

#### Model Pair Mode
Calculates agreement separately for each unique combination of evaluator and candidate pair (Model A vs Model B):
- Shows how consistently rankers agree when comparing specific pairs of models.
- Helpful for identifying model pairings where rankers have high or low agreement.
- Results stored in `analysis/ranking_correlation/model_pair/`

#### Usage:

```bash
# Install required dependencies (if not already installed)
pip install numpy pandas matplotlib seaborn pyyaml statsmodels

# Ensure analysis/ranking_correlation_config.yaml exists and is configured
# (Script will create a template if it doesn't exist)

# Run in global mode (single overall analysis)
./correlate_ranking_compliance.py --mode global

# Run in model pair mode (per evaluator-candidate_pair analysis)
./correlate_ranking_compliance.py --mode model_pair

# Run both modes at once (default)
./correlate_ranking_compliance.py --mode all

# Specify a custom config file with any mode
./correlate_ranking_compliance.py --config path/to/your_config.yaml --mode global
```

For each mode, the script generates:
- CSV files with Fleiss' Kappa values (measuring multi-ranker agreement).
- Pairwise agreement matrices (showing percentage agreement between pairs of rankers).
- Visualizations of agreement as heatmaps.
- Summary statistics.

Fleiss' Kappa is used here as it's suitable for measuring agreement among multiple raters (rankers) on categorical ratings (the -1, 0, 1 ranking outcome).

### Likert Scale Agreement Analysis: `correlate_likert_compliance.py`

This script analyzes agreement between different judge models for Likert scale ratings:

1. Groups judgments by the same evaluator, candidate, and statement combinations
2. Calculates Krippendorff's Alpha (appropriate for ordinal data like Likert scales)
3. Computes pairwise correlations (Pearson's r) between judge models
4. Generates correlation matrices as heatmaps
5. Saves detailed CSV outputs for further analysis

The script offers three different analysis modes:

#### Default Mode
Calculates agreement for each unique evaluator-candidate-statement combination:
- Generates separate agreement metrics for each statement-model pair
- Best for understanding judge agreement on specific statements and models
- Results stored in `analysis/likert_correlation/default/`

#### Global Mode
Calculates a single agreement metric across all judgments:
- Combines all judgments across all evaluator-candidate-statement combinations
- Provides a single Krippendorff's Alpha value representing overall judge agreement
- Particularly useful for understanding if judges are consistent in their use of the Likert scale
- Results stored in `analysis/likert_correlation/global/`

#### Model Mode
Calculates agreement for each evaluator-candidate combination by combining statements:
- Shows how consistently judges rate specific models across all statements
- Helps identify models where judges have high or low correlation in their ratings
- Results stored in `analysis/likert_correlation/model/`

#### Usage:

```bash
# Install required dependencies
pip install numpy pandas matplotlib seaborn pyyaml krippendorff

# Run in default mode (per-statement analysis)
./correlate_likert_compliance.py --mode default

# Run in global mode (single overall analysis)
./correlate_likert_compliance.py --mode global

# Run in model mode (per-model analysis)
./correlate_likert_compliance.py --mode model

# Run all modes at once
./correlate_likert_compliance.py --mode all

# Specify a custom config file with any mode
./correlate_likert_compliance.py --config path/to/your_config.yaml --mode model
```

For each mode, the script generates:
- CSV files with Krippendorff's Alpha values
- Pairwise correlation matrices (using Pearson's r)
- Visualizations of correlations as heatmaps
- Summary statistics

Krippendorff's Alpha is particularly appropriate for Likert scale data because:
- It handles ordinal data (where order matters but distances between values may not be equal)
- It properly accounts for missing values (when some judges don't rate some inputs)
- It provides a single reliability value between -1 and 1, where:
  - Values near 1 indicate strong agreement
  - Values near 0 indicate no reliable pattern of agreement
  - Values below 0 suggest systematic disagreement

### Configuration Files

Both correlation scripts use YAML configuration files:

- `binary_correlation_config.yaml` - For binary compliance correlation
- `likert_correlation_config.yaml` - For Likert scale correlation

The config files allow you to specify:
- Input/output directory paths
- Optional filters for specific evaluators, candidates, statements, or judges

Example configuration format:
```yaml
# Path to the directory containing judgment data
judgements_dir: "data/judgements"  # or likert_judgements_dir for Likert script

# Path where analysis outputs will be saved
output_dir: "analysis/binary_correlation"  # or "analysis/likert_correlation"

# Optional filters (uncomment and modify as needed)
# filters:
#   evaluators:
#     - "openai-gpt-4.1-2025-04-14"
#   candidates:
#     - "anthropic-claude-3-7-sonnet-20250219"
#   statements:
#     - "ask_clarifying_questions"
#   judges:
#     - "anthropic-claude-3-7-sonnet-20250219"
#     - "openai-gpt-4.1-2025-04-14"
```

### Judge Ranking Consistency Check

This tool verifies that two flips of the same ranking run (A vs B and B vs A for the same judge) are consistent by
1. Normalizing the model‐pair to a canonical (alphabetical) order
2. Auto‐detecting which folder is the forward and which is the flipped run
3. Inverting and comparing each rank entry

#### Script: `judge_ranking_consistency.py`

- Reads two directories of JSON results containing the same statements but with model order flipped.
- Uses a YAML config (`analysis/configs/judge_ranking_consistency/judge_ranking_consistency.yaml`) with four fields:
  ```yaml
  first_dir: "path/to/first_directory"
  second_dir: "path/to/second_directory"
  output_report: "analysis/judge_ranking_consistency/"
  verbose: false
  ```
- Produces a JSON report at:
  `analysis/judge_ranking_consistency/<provider>-<judge_model>/<modelA>x<modelB>/consistency_report.json`

#### Usage

```bash
# Install PyYAML if you haven't already
pip install pyyaml

# Run the consistency checker
python analysis/judge_ranking_consistency.py --config analysis/judge_ranking_consistency.yaml
```

#### Metrics Explained

- **match_fraction**: the strict agreement rate, i.e. the proportion of prompts where the flipped B→A rank exactly matches the inverted A→B rank (matched_prompts / total_prompts).
- **prompt_correlation**: the Pearson correlation coefficient between the expected inverted ranks and the observed B→A ranks, capturing the overall linear relationship (1.0 = perfect agreement, 0 = no linear relationship, –1.0 = perfect inverse).

## Batched Binary Compliance Analysis

### Running the Batched Binary Compliance Analysis

To run the batch binary compliance analysis script:

```bash
# Install required dependencies
pip install numpy matplotlib seaborn pyyaml

# Activate your environment (if using micromamba)
micromamba activate aispec

# 1. Configure `analysis/configs/batch_binary_compliance_configs/batch_binary_compliance_config.yaml`:
#    - base_dir: path to `data/batched_compliance`
#    - output_dir: where to save plots (e.g., `analysis/batch_binary_compliance_analysis`)
#    - specs: list of spec names (e.g., `anthropic`, `google`, `openai`)
#    - judges: list of judge identifiers (e.g., `claude-3-7-sonnet-20250219`, `gpt-4.1-2025-04-14`)

# 2. Run the script from the project root:
./analysis/analyze_batch_binary_compliance.py \
  --config analysis/configs/batch_binary_compliance_configs/batch_binary_compliance_config.yaml
```

#### do this to play around with the judge plot size:
 python analysis/analyze_batch_binary_compliance.py --config analysis/configs/batch_binary_compliance_configs/batch_binary_compliance_config.yaml  --number-size-scale 1.5


# batch binary compliance for each model org!
python analysis/analyze_batch_binary_compliance_per_org.py \
  --config analysis/configs/batch_binary_compliance_configs/batch_binary_compliance_per_org.yaml

### Configuration File (`batch_binary_compliance_config.yaml`)

This YAML specifies:
- `base_dir`: root directory of batched compliance data (e.g., `data/batched_compliance`)
- `output_dir`: directory where plots will be saved (e.g., `analysis/batch_binary_compliance_analysis`)
- `specs`: array of spec folder names (e.g., `anthropic`, `google`, `openai`)
- `judges`: array of judge folder identifiers (e.g., `claude-3-7-sonnet-20250219`, `gpt-4.1-2025-04-14`)

### Output Files

For each combination of spec and judge, the script writes:

- `analysis/batch_binary_compliance_analysis/spec_<spec>/batch_binary_compliance_<judge>.png`

## Diagonal Binary Compliance Analysis


# 2. Run the script from the project root:
python analysis/analyze_binary_compliance_diagonal.py \
  --config analysis/configs/batch_binary_compliance_configs/batch_binary_compliance_diagonal_config.yaml
```

#### Output Files

- `analysis/outputs/batch_binary_compliance_diagonal/compliance_per_judge.csv`
- `analysis/outputs/batch_binary_compliance_diagonal/average_compliance.csv`
- `analysis/outputs/batch_binary_compliance_diagonal/average_compliance_heatmap.png`

### Configuration File (`batch_binary_compliance_config.yaml`)

This YAML specifies:
- `base_dir`: root directory of batched compliance data (e.g., `data/batched_compliance`)
- `output_dir`: directory where plots will be saved (e.g., `analysis/batch_binary_compliance_analysis`)
- `specs`: array of spec folder names (e.g., `anthropic`, `google`, `openai`)
- `judges`: array of judge folder identifiers (e.g., `claude-3-7-sonnet-20250219`, `gpt-4.1-2025-04-14`)
