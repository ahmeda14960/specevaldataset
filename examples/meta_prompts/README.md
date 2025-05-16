# Meta Prompt Generation Scripts

This directory contains scripts that use a "meta-prompting" approach. These scripts leverage a powerful language model (the "generator") to create specialized prompts tailored for specific tasks within the SpecEval framework, such as generating evaluation prompts for LLM-as-a-judge.

## `likert_meta_prompts.py`

**Purpose:** This script generates specific LLM-as-a-judge prompts designed to evaluate compliance with individual policy statements on a 1-5 Likert scale.

**How it works:**
1.  Reads a policy specification file (in JSONL format).
2.  For each statement in the specification:
    *   It takes the statement details (text, type, authority, examples, etc.).
    *   It formats these details into a pre-defined "meta-prompt" (either `META_PROMPT_LIKERT_RESPONSE_JUDGE_GENERATOR` or `META_PROMPT_LIKERT_INPUT_JUDGE_GENERATOR` from `speceval.utils.prompts`).
    *   It sends this meta-prompt to a specified generator LLM (e.g., GPT-4o, Claude 3.7 Sonnet, Gemini 1.5 Flash).
    *   The generator LLM, following the instructions in the meta-prompt, creates a new, detailed prompt specifically designed for an LLM judge to evaluate compliance with *that particular statement* using a tailored 1-5 Likert scale.
3.  Saves each generated judge prompt as a separate text file in the specified output directory.

**Usage:**

```bash
python examples/meta_prompts/likert_meta_prompts.py \
    --spec-path <path_to_spec.jsonl> \
    --org <generator_organization> \
    --generator-model <model_name> \
    --mode <response_judge|input_judge> \
    [--output-dir <directory_to_save_prompts>] \
    [--api-key <your_api_key>] \
    [--verbose]
```

**Arguments:**

*   `--spec-path` (Required): Path to the input specification file (e.g., `data/specs/openai/jsonl/openai.jsonl`).
*   `--org` (Required): The provider of the generator model. Choices: `openai`, `anthropic`, `google`.
*   `--generator-model` (Required): The specific model name to use for generation (must be valid for the chosen `--org`). Examples: `gpt-4o-2024-08-06`, `claude-3-7-sonnet-20250219`, `gemini-1.5-flash-latest`.
*   `--mode` (Required): Mode of judge-prompt generation. Choices: `response_judge` (default) to evaluate assistant *responses*, or `input_judge` to evaluate *user inputs* for distinguishability.
*   `--output-dir` (Optional): Directory where the generated judge prompt files will be saved. Defaults to `data/generated_judge_prompts/likert_<mode>` (e.g., `likert_response_judge` or `likert_input_judge`).
*   `--api-key` (Optional): API key for the generator model. If not provided, the script will look for the corresponding environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Note: For Google, authentication might be handled via `gcloud auth login`.
*   `--verbose` (Optional): Enable detailed logging output.

**Example:**

Generate Likert judge prompts (response mode) for the OpenAI spec using GPT-4.1, saving them to a custom directory:

```bash
python examples/meta_prompts/likert_meta_prompts.py \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --org openai \
    --generator-model gpt-4.1-2025-04-14 \
    --mode response_judge \
    --output-dir data/judge_prompts/openai_likert_response_judge \
    --verbose
```

```bash
python examples/meta_prompts/likert_meta_prompts.py \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --org openai \
    --generator-model gpt-4.1-2025-04-14 \
    --mode response_judge \
    --verbose
```

Generate judge-input prompts (input mode) for the same spec:

```bash
python examples/meta_prompts/likert_meta_prompts.py \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --org openai \
    --generator-model gpt-4.1-2025-04-14 \
    --mode input_judge \
    --output-dir data/judge_prompts/openai_likert_input_judge \
    --verbose
```

for anthropic spec / claude constitution

```bash
python examples/meta_prompts/likert_meta_prompts.py \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --org openai \
    --generator-model gpt-4.1-2025-04-14 \
    --mode input_judge \
    --output-dir data/judge_prompts/anthropic_likert_input_judge \
    --verbose
```

for sparrow rules / google org
```bash
python examples/meta_prompts/likert_meta_prompts.py \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --org openai \
    --generator-model gpt-4.1-2025-04-14 \
    --mode input_judge \
    --output-dir data/judge_prompts/google_likert_input_judge \
    --verbose
```
**Output:**

The script will create files in the `--output-dir`, such as:
`data/judge_prompts/openai_likert_gpt4.1/policy_content_intellectual_property_likert_judge.txt`
`data/judge_prompts/openai_likert_gpt4.1/policy_represent_openai_likert_judge.txt`
...

Each `.txt` file will contain the generated prompt ready to be used by an LLM judge for evaluating responses against that specific statement on a 1-5 scale.
