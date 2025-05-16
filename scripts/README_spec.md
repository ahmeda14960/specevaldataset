# Specification Parsing and Consolidation Workflow

This document describes the process for parsing a specification defined in a JSONL file into individual YAML files per statement, and then consolidating those YAML files back into a single JSONL file. This workflow is useful for editing or reviewing individual specification statements in a more human-readable format (YAML) before potentially re-assembling them.

Two scripts facilitate this workflow:

1.  `parse_oai_spec.py`: Extracts statements from a JSONL spec file into individual YAML files.
2.  `consolidate_yaml_spec.py`: Consolidates individual statement YAML files back into a single JSONL file.

## Dependencies

Both scripts require the `PyYAML` library. Install it if you haven't already:

```bash
pip install PyYAML
```

## Workflow

### 1. Parsing JSONL to Individual YAML Files

The `parse_oai_spec.py` script reads a source JSONL file (e.g., `data/specs/openai_model_spec.jsonl`) and creates a directory structure under `scr/`. For each statement in the JSONL file, it generates a separate YAML file named after the statement's ID.

**Usage:**

Run the script from the project root directory:

```bash
python scripts/parse_oai_spec.py
```

*   **Input:** By default, it reads `data/specs/openai_model_spec.jsonl`. (You can modify the `data_path` variable in the script if needed).
*   **Output:** Creates individual YAML files (e.g., `follow_all_applicable_instructions.yaml`, `letter_and_spirit.yaml`, etc.) inside the `scr/openai_model_spec/` directory. Each YAML file contains the structured data for one statement.

### 2. Consolidating YAML Files back to JSONL

The `consolidate_yaml_spec.py` script performs the reverse operation. It reads all `.yaml` files from a specified directory (where each file represents a single statement) and writes them as lines into a new JSONL file.

**Usage:**

Run the script from the project root directory, providing the directory containing the individual YAML files as an argument:

```bash
python scripts/consolidate_yaml_spec.py <path_to_yaml_directory>
```

*   **Input:** Reads all `.yaml` files within the specified directory (e.g., `scr/openai_model_spec/`).
*   **Output:** Creates a single JSONL file named after the input directory (e.g., `openai_model_spec.jsonl`) inside a `jsonl` subdirectory within the input directory (i.e., `scr/openai_model_spec/jsonl/openai_model_spec.jsonl`).

**Optional Reference Check:**

You can optionally provide a path to a reference JSONL file using the `--reference` argument. If provided, the script will compare the newly generated JSONL file with the reference file. The comparison ignores the order of lines (statements) within the files.

```bash
python scripts/consolidate_yaml_spec.py scr/openai_model_spec/ --reference data/specs/openai_model_spec.jsonl
```

*   If the files match (contain the same set of statements, regardless of order), it will print a success message.
*   If the files do not match, it will print a mismatch message to standard error and exit with a non-zero status code.
*   The comparison uses canonical JSON representations (keys sorted) for each line.

**Verbose Output for Differences:**

If the comparison fails and you want to see the specific differences, add the `--verbose` (or `-v`) flag:

```bash
python scripts/consolidate_yaml_spec.py scr/openai_model_spec/ --reference data/specs/openai_model_spec.jsonl --verbose
```

*   In case of a mismatch, this will print the statements (as canonical JSON strings) that are present in the output file but not the reference, and vice-versa.

This workflow allows for easier management and editing of individual specification statements while maintaining the ability to reconstruct the original JSONL format and verify its integrity against a reference.

### 3. Testing Individual YAML Parsing (Utility)

The `test_extract_yaml.py` script is a utility for verifying how a single YAML statement file is parsed into a `Statement` object (defined in `speceval/base/statement.py`). This can be helpful for debugging issues with a specific YAML file or understanding the object representation.

**Usage:**

Run the script from the project root directory, providing the path to the specific YAML file you want to test:

```bash
python scripts/test_extract_yaml.py <path_to_single_yaml_file>
```

*   **Input:** Reads the specified `.yaml` file (e.g., `scr/openai_model_spec/ask_clarifying_questions.yaml`).
*   **Output:** Prints the `Statement` object representation of the parsed YAML content to the standard output. It also includes a printout of the statement as a dictionary for more detailed inspection.

Example:

```bash
python scripts/test_extract_yaml.py scr/openai_model_spec/ask_clarifying_questions.yaml
```
