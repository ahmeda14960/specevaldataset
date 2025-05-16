# speceval/pipelines/ranking_pipeline.py

"""Implements a pipeline for ranking pre-generated model outputs."""

import json
import logging
import itertools
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm

from ..base import Specification, RankingModel

# Attempt to import the utility for model name extraction
try:
    from ..utils.parsing import extract_model_name_from_path
except ImportError:
    extract_model_name_from_path = None  # Fallback if not found
    logger.warning(
        "Could not import 'extract_model_name_from_path' from 'speceval.utils.parsing'. "
        "Model name extraction from path for batched mode might be affected."
    )

# flake8: noqa: F401
from ..models.openai import (
    GPT_4_1_MINI,
)  # Import the specific model as a default

# Set up logging
logger = logging.getLogger(__name__)

# --- Helper Functions ---


def _parse_generation_dir(generations_base_dir: Path) -> Tuple[Optional[str], List[str]]:
    """
    Parse the generations directory to find the common evaluator and candidate models.

    Expects subdirectories named like '{evaluator}x{candidate}'.

    Returns:
        A tuple containing:
        - The common evaluator model name (str) or None if inconsistent/not found.
        - A list of candidate model names (List[str]).
    """
    evaluator_model = None
    candidate_models = set()
    found_dirs = False

    if not generations_base_dir.is_dir():
        logger.error(f"Generations base directory not found: {generations_base_dir}")
        return None, []

    for item in generations_base_dir.iterdir():
        if item.is_dir() and "x" in item.name:
            found_dirs = True
            parts = item.name.split("x", 1)
            if len(parts) == 2:
                current_evaluator, current_candidate = parts
                if evaluator_model is None:
                    evaluator_model = current_evaluator
                elif evaluator_model != current_evaluator:
                    logger.error(
                        f"Inconsistent evaluator models found: '{evaluator_model}' and '{current_evaluator}'. "
                        "Ranking pipeline requires a single evaluator for the input set."
                    )
                    return None, []
                candidate_models.add(current_candidate)
            else:
                logger.warning(f"Skipping directory with unexpected name format: {item.name}")

    if not found_dirs:
        logger.warning(
            f"No subdirectories matching '{{evaluator}}x{{candidate}}' format found in {generations_base_dir}"
        )
        return None, []
    if evaluator_model is None:
        logger.warning(f"Could not determine a common evaluator model in {generations_base_dir}")
        return None, list(candidate_models)

    return evaluator_model, sorted(list(candidate_models))


def _load_generations(generation_file_path: Path) -> Dict[str, str]:
    """
    Load generations from a single JSON file.

    Args:
        generation_file_path: Path to the JSON file (e.g., .../{statement_id}.json).

    Returns:
        A dictionary mapping input_text to output_text.
        Returns an empty dictionary if the file doesn't exist or is invalid.
    """
    if not generation_file_path.exists():
        logger.warning(f"Generation file not found: {generation_file_path}")
        return {}
    try:
        with open(generation_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "generations" in data and isinstance(data["generations"], list):
            # Assuming unique inputs per file for simplicity here.
            # A more robust implementation might handle duplicate inputs if possible.
            return {
                gen["input"]: gen["output"]
                for gen in data["generations"]
                if "input" in gen and "output" in gen
            }
        else:
            logger.warning(
                f"Invalid format in {generation_file_path}: 'generations' key missing or not a list."
            )
            return {}
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in file: {generation_file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading generations from {generation_file_path}: {e}")
        return {}


def _validate_input_consistency(statement_id: str, gen_files: List[Path]) -> Optional[List[str]]:
    """
    Validate that all generation files for a given statement have the same inputs.

    Checks that inputs are identical and in the same order across files.

    Args:
        statement_id: The ID of the statement being checked.
        gen_files: A list of paths to the generation JSON files for this statement
                   across different candidate models.

    Returns:
        The list of common input strings if consistent, otherwise None.
    """
    if not gen_files:
        logger.warning(f"No generation files provided for validation for statement {statement_id}")
        return None

    first_inputs = None
    try:
        # Load inputs from the first file
        with open(gen_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        if "generations" not in data or not isinstance(data["generations"], list):
            logger.warning(
                f"Invalid format in {gen_files[0]}: 'generations' key missing or not a list."
            )
            return None
        first_inputs = [gen["input"] for gen in data["generations"] if "input" in gen]

        if not first_inputs:
            logger.warning(f"No inputs found in the first generation file: {gen_files[0]}")
            return None  # Or handle cases with zero inputs if valid

        # Compare inputs from subsequent files
        for file_path in gen_files[1:]:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "generations" not in data or not isinstance(data["generations"], list):
                logger.warning(
                    f"Invalid format in {file_path}: 'generations' key missing or not a list."
                )
                return None
            current_inputs = [gen["input"] for gen in data["generations"] if "input" in gen]
            if current_inputs != first_inputs:
                logger.error(
                    f"Input inconsistency found for statement '{statement_id}'. "
                    f"File {gen_files[0].name} and {file_path.name} have different inputs."
                )
                # Optionally log the differing inputs for debugging
                # logger.debug(f"Expected Inputs ({gen_files[0].name}): {first_inputs}")
                # logger.debug(f"Found Inputs ({file_path.name}): {current_inputs}")
                return None

        return first_inputs

    except FileNotFoundError as e:
        logger.error(f"Validation failed for statement '{statement_id}': File not found - {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Validation failed for statement '{statement_id}': Invalid JSON in {e.filename} - {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error during input validation for statement '{statement_id}': {e}"
        )
        return None


# --- Helper Functions for Batched Mode ---


def _load_batched_generations(generation_file_path: Path) -> Dict[str, str]:
    """
    Load generations from a single JSON file in batched format.
    The JSON file is expected to be a list of objects, each with
    'input_text' and 'output_text'.

    Args:
        generation_file_path: Path to the JSON file (e.g., .../results/{statement_id}.json).

    Returns:
        A dictionary mapping input_text to output_text.
        Returns an empty dictionary if the file doesn't exist or is invalid.
    """
    if not generation_file_path.exists():
        logger.warning(f"Batched generation file not found: {generation_file_path}")
        return {}
    try:
        with open(generation_file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        if isinstance(data_list, list):
            # Ensure no duplicate inputs, take the first occurrence if any (though not expected)
            generations_map = {}
            for item in data_list:
                if "input_text" in item and "output_text" in item:
                    if item["input_text"] not in generations_map:
                        generations_map[item["input_text"]] = item["output_text"]
                    else:
                        logger.warning(
                            f"Duplicate input_text found in {generation_file_path}: '{item['input_text'][:50]}...'. Using first occurrence."
                        )
                else:
                    logger.warning(
                        f"Missing 'input_text' or 'output_text' in item in {generation_file_path}. Item: {item}"
                    )
            return generations_map
        else:
            logger.warning(
                f"Invalid format in {generation_file_path}: Expected a list of generation objects."
            )
            return {}
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in batched generation file: {generation_file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading batched generations from {generation_file_path}: {e}")
        return {}


def _validate_batched_input_consistency(
    statement_id: str, gen_files: List[Path]
) -> Optional[List[str]]:
    """
    Validate that all batched generation files for a given statement have the same inputs.
    Checks that 'input_text' are identical and in the same order across files.

    Args:
        statement_id: The ID of the statement being checked.
        gen_files: A list of paths to the batched generation JSON files for this statement
                   across different candidate models.

    Returns:
        The list of common input strings if consistent, otherwise None.
    """
    if not gen_files:
        logger.warning(
            f"No batched generation files provided for validation for statement {statement_id}"
        )
        return None

    first_inputs = None
    try:
        # Load inputs from the first file
        with open(gen_files[0], "r", encoding="utf-8") as f:
            data_list = json.load(f)
        if not isinstance(data_list, list):
            logger.warning(
                f"Invalid format in {gen_files[0]}: Expected a list for batched generations."
            )
            return None
        first_inputs = [item["input_text"] for item in data_list if "input_text" in item]

        if not first_inputs:
            logger.warning(f"No inputs found in the first batched generation file: {gen_files[0]}")
            return None

        # Compare inputs from subsequent files
        for file_path in gen_files[1:]:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning(
                    f"Invalid format in {file_path}: Expected a list for batched generations."
                )
                return None
            current_inputs = [item["input_text"] for item in data if "input_text" in item]
            if current_inputs != first_inputs:
                logger.error(
                    f"Batched input inconsistency found for statement '{statement_id}'. "
                    f"File {gen_files[0].name} and {file_path.name} have different inputs."
                )
                return None
        return first_inputs
    except FileNotFoundError as e:
        logger.error(
            f"Batched validation failed for statement '{statement_id}': File not found - {e}"
        )
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Batched validation failed for statement '{statement_id}': Invalid JSON in {getattr(e, 'filename', 'unknown file')} - {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error during batched input validation for statement '{statement_id}': {e}"
        )
        return None


def _parse_batched_generation_dir(
    provider_dir: Path,
) -> Tuple[Optional[str], List[str], Optional[Dict[str, Path]]]:
    """
    Parse the batched generations directory structure.
    Example: data/batched_generations/{provider}/{model_name}/{run_id}/results/
    Selects the best run for each model (most statements, then most recent).

    Args:
        provider_dir: Path to the provider directory (e.g., data/batched_generations/openai).

    Returns:
        A tuple containing:
        - Provider name (str) or None if not determinable.
        - List of candidate model names (List[str]).
        - Dictionary mapping candidate model name to its best 'results' directory path (Dict[str, Path]).
    """
    provider_name = provider_dir.name
    candidate_model_paths_map: Dict[str, Path] = {}
    candidate_model_names_list: List[str] = []

    if not provider_dir.is_dir():
        logger.error(f"Provider directory not found: {provider_dir}")
        return None, [], None

    for model_dir in provider_dir.iterdir():
        if model_dir.is_dir():
            canonical_model_name = (
                model_dir.name
            )  # Default if extract_model_name_from_path is unavailable or fails
            if extract_model_name_from_path:
                try:
                    # Use the full path of the model_dir for extraction context
                    canonical_model_name = extract_model_name_from_path(model_dir)
                except ValueError as e:
                    logger.warning(
                        f"Could not extract unique model name from path {model_dir}: {e}. Using directory name '{model_dir.name}'."
                    )

            best_run_results_path: Optional[Path] = None
            relevant_runs: List[
                Tuple[int, str, Path]
            ] = []  # (statement_count, run_dir_name, run_results_path)

            for run_dir in model_dir.iterdir():
                if run_dir.is_dir():
                    results_dir = run_dir / "results"
                    if results_dir.is_dir():
                        statement_files = [
                            f for f in results_dir.iterdir() if f.is_file() and f.suffix == ".json"
                        ]
                        statement_count = len(statement_files)
                        if statement_count > 0:
                            relevant_runs.append((statement_count, run_dir.name, results_dir))

            if relevant_runs:
                # Sort by statement_count (desc), then run_dir_name (desc for recency)
                # Primary sort: statement_count (descending)
                # Secondary sort: run_dir_name (descending, for most recent, assuming YYYYMMDD_HHMMSS like format)
                relevant_runs.sort(key=lambda x: (x[0], x[1]), reverse=True)

                best_run_results_path = relevant_runs[0][2]
                if canonical_model_name in candidate_model_paths_map:
                    logger.warning(
                        f"Duplicate canonical model name '{canonical_model_name}' found. Overwriting with path from {model_dir.name}."
                    )
                candidate_model_paths_map[canonical_model_name] = best_run_results_path
                if (
                    canonical_model_name not in candidate_model_names_list
                ):  # Ensure uniqueness if multiple folders map to same canonical name
                    candidate_model_names_list.append(canonical_model_name)

    if not candidate_model_paths_map:
        logger.warning(f"No valid model runs found in {provider_dir}")
        return provider_name, [], None

    # Sort candidate_model_names_list for consistent ordering if needed, though map iteration order is usually preserved in Python 3.7+
    return provider_name, sorted(candidate_model_names_list), candidate_model_paths_map


# --- Ranking Pipeline ---


class RankingPipeline:
    """Pipeline for ranking pre-generated model outputs using a provided RankingModel."""

    def __init__(
        self,
        specification: Specification,
        generations_dir: str = None,
        candidate_generation_dirs: Optional[List[str]] = None,
        evaluator_model_name: Optional[str] = None,
        ranking_model: RankingModel = None,
        output_dir_base: str = "data/rankings",
        verbose: bool = False,
        all_pairs: bool = False,
        skip_existing: bool = False,
        batched_generations_mode: bool = False,
    ):
        """
        Initialize the ranking pipeline.

        Args:
            specification: The specification object.
            generations_dir: Path to the base directory containing generation subdirs
                             (e.g., 'data/generations').
            ranking_model: An instance of a RankingModel implementation.
            output_dir_base: Base directory to store ranking results.
            verbose: Enable verbose logging.
            all_pairs: When True, runs all possible pairs including both directions (A vs B and B vs A).
                      When False (default), only runs unique pairs (A vs B only).
            skip_existing: When True, skip ranking for statement/model pairs that already have results files.
            batched_generations_mode: When True, uses the batched generations directory structure and loading logic.
                                      The 'generations_dir' should point to the provider level,
                                      e.g., 'data/batched_generations/openai'.
        """
        self.specification = specification
        self.generations_base_dir = Path(generations_dir) if generations_dir else None
        self.ranking_model = ranking_model
        self.output_dir_base = Path(output_dir_base)
        self.verbose = verbose
        self.all_pairs = all_pairs
        self.skip_existing = skip_existing
        self.batched_generations_mode = batched_generations_mode
        # new: list of explicit candidate dirs
        self.candidate_generation_dirs = (
            [Path(p) for p in candidate_generation_dirs] if candidate_generation_dirs else None
        )
        # prepare mapping for candidate results when using explicit dirs or batched mode
        self.batched_candidate_results_paths: Optional[Dict[str, Path]] = None

        # mode: explicit candidate dirs
        if self.candidate_generation_dirs:
            # require evaluator_model_name
            if not evaluator_model_name:
                logger.error(
                    "evaluator_model_name is required when using candidate_generation_dirs"
                )
                raise ValueError(
                    "Missing evaluator_model_name for explicit candidate_generation_dirs mode"
                )
            self.evaluator_model_name = evaluator_model_name
            # extract model names
            if extract_model_name_from_path:
                self.candidate_model_names = [
                    extract_model_name_from_path(p) for p in self.candidate_generation_dirs
                ]
            else:
                self.candidate_model_names = [p.name for p in self.candidate_generation_dirs]
            # build mapping name -> path
            self.batched_candidate_results_paths = {
                name: path
                for name, path in zip(self.candidate_model_names, self.candidate_generation_dirs)
            }
        elif self.batched_generations_mode:
            if extract_model_name_from_path is None:
                logger.error(
                    "Batched generation mode requires 'extract_model_name_from_path' utility, but it could not be imported."
                )
                raise ImportError(
                    "Missing 'extract_model_name_from_path' for batched_generations_mode."
                )

            (
                self.evaluator_model_name,  # This will be the provider name, e.g., "openai"
                self.candidate_model_names,
                self.batched_candidate_results_paths,
            ) = _parse_batched_generation_dir(self.generations_base_dir)
            if (
                not self.batched_candidate_results_paths
            ):  # Check if parsing failed to find models/runs
                raise ValueError(
                    f"Could not parse batched generations from {self.generations_base_dir}. "
                    "Ensure the directory structure is correct and contains valid runs."
                )
        else:
            # standard generations_dir with subfolders evaluatorxcandidate
            self.evaluator_model_name, self.candidate_model_names = _parse_generation_dir(
                self.generations_base_dir
            )

        if (
            not self.evaluator_model_name
            or not self.candidate_model_names
            or len(self.candidate_model_names) < 2
        ):
            raise ValueError(
                "Could not determine a common evaluator and at least two candidate models "
                f"(Evaluator: {self.evaluator_model_name}, Candidates: {self.candidate_model_names})"
            )

        # Extract the ranking model name for the directory structure
        ranking_model_info = self.ranking_model.get_info()
        ranking_model_name = ranking_model_info.get("model_name", "unknown_ranker")

        # Create a directory structure with ranking model at the top level
        self.rankings_output_dir = (
            self.output_dir_base / f"ranker_{ranking_model_name}" / self.evaluator_model_name
        )
        self.rankings_output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            logger.info(f"Ranking output directory set to: {self.rankings_output_dir}")
            logger.info(f"Evaluator model identified: {self.evaluator_model_name}")
            logger.info(f"Candidate models identified: {self.candidate_model_names}")
            logger.info(f"Using ranking model: {ranking_model_name}")

    def run(self):
        """Run the ranking pipeline."""
        logger.info("Starting ranking pipeline...")

        statements = self.specification.statements
        if not statements:
            logger.warning("No statements found in the specification. Exiting.")
            return

        validated_statements: Dict[str, List[str]] = {}  # Store statement_id -> common_inputs

        # --- Validation Phase ---
        logger.info("Validating input consistency across candidate models...")
        for statement in tqdm(statements, desc="Validating Statements"):
            generation_files_for_statement = []
            all_files_exist = True

            if self.batched_generations_mode:
                if (
                    not self.batched_candidate_results_paths
                ):  # Should be caught in init, but double check
                    logger.error("Batched mode is active but candidate result paths are not set.")
                    return  # or raise error
                for candidate_name in self.candidate_model_names:
                    candidate_results_dir = self.batched_candidate_results_paths.get(candidate_name)
                    if not candidate_results_dir:
                        logger.warning(
                            f"Missing results path for batched candidate '{candidate_name}'. Skipping for statement '{statement.id}'."
                        )
                        all_files_exist = False
                        break
                    gen_file = candidate_results_dir / f"{statement.id}.json"
                    if not gen_file.exists():
                        logger.warning(
                            f"Missing batched generation file for validation: {gen_file}"
                        )
                        all_files_exist = False
                        break
                    generation_files_for_statement.append(gen_file)
            else:  # Original mode
                for candidate_name in self.candidate_model_names:
                    gen_dir = (
                        self.generations_base_dir / f"{self.evaluator_model_name}x{candidate_name}"
                    )
                    gen_file = gen_dir / f"{statement.id}.json"
                    if not gen_file.exists():
                        logger.warning(f"Missing generation file for validation: {gen_file}")
                        all_files_exist = False
                        break
                    generation_files_for_statement.append(gen_file)

            if not all_files_exist:
                logger.warning(
                    f"Skipping statement '{statement.id}' due to missing generation files."
                )
                continue

            if self.batched_generations_mode:
                common_inputs = _validate_batched_input_consistency(
                    statement.id, generation_files_for_statement
                )
            else:
                common_inputs = _validate_input_consistency(
                    statement.id, generation_files_for_statement
                )

            if common_inputs is not None:
                validated_statements[statement.id] = common_inputs
                if self.verbose:
                    logger.debug(
                        f"Statement '{statement.id}' validated with {len(common_inputs)} common inputs."
                    )
            else:
                logger.warning(f"Skipping statement '{statement.id}' due to input inconsistency.")
        # --- End Validation Phase ---

        if not validated_statements:
            logger.error("No statements passed validation. Cannot proceed with ranking.")
            return

        # --- Ranking Phase ---
        logger.info("Starting ranking comparisons...")

        # Get all model pairs based on all_pairs flag
        if self.all_pairs:
            # Generate all ordered pairs (A,B) and (B,A)
            model_pairs = []
            for model_a_name in self.candidate_model_names:
                for model_b_name in self.candidate_model_names:
                    if model_a_name != model_b_name:
                        model_pairs.append((model_a_name, model_b_name))
        else:
            # Generate unique pairs (A,B) only
            model_pairs = list(itertools.combinations(self.candidate_model_names, 2))

        for model_a_name, model_b_name in tqdm(model_pairs, desc="Ranking Model Pairs"):
            pair_output_dir = self.rankings_output_dir / f"{model_a_name}VS{model_b_name}"
            pair_output_dir.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                logger.info(f"Processing pair: {model_a_name} vs {model_b_name}")

            # Construct paths to the generation directories/files for this pair
            # This part is significantly different for batched mode vs original

            for statement_id, common_inputs in tqdm(
                validated_statements.items(),
                desc=f"Ranking {model_a_name} vs {model_b_name}",
                leave=False,
            ):
                statement = self.specification.get_statement(statement_id)
                if not statement:  # Should not happen if validation passed, but check anyway
                    logger.error(
                        f"Could not find statement object for validated ID: {statement_id}"
                    )
                    continue

                # Check if output file already exists and skip_existing is enabled
                output_file_path = pair_output_dir / f"{statement.id}.json"
                if self.skip_existing and output_file_path.exists():
                    if self.verbose:
                        logger.info(
                            f"Skipping existing ranking for statement '{statement.id}' pair {model_a_name} vs {model_b_name}"
                        )
                    continue

                # Load generations for this statement for both models in the pair
                generations_a: Dict[str, str] = {}
                generations_b: Dict[str, str] = {}

                if self.batched_generations_mode:
                    if not self.batched_candidate_results_paths:  # Should not happen
                        logger.error("Batched mode error: results paths not found during ranking.")
                        continue

                    path_a_results_dir = self.batched_candidate_results_paths.get(model_a_name)
                    path_b_results_dir = self.batched_candidate_results_paths.get(model_b_name)

                    if not path_a_results_dir or not path_b_results_dir:
                        logger.warning(
                            f"Could not find results directory for one or both models in pair ({model_a_name}, {model_b_name}) for batched mode. Skipping statement '{statement.id}'."
                        )
                        continue

                    gen_file_a = path_a_results_dir / f"{statement.id}.json"
                    gen_file_b = path_b_results_dir / f"{statement.id}.json"

                    generations_a = _load_batched_generations(gen_file_a)
                    generations_b = _load_batched_generations(gen_file_b)
                else:  # Original mode
                    gen_dir_a = (
                        self.generations_base_dir / f"{self.evaluator_model_name}x{model_a_name}"
                    )
                    gen_dir_b = (
                        self.generations_base_dir / f"{self.evaluator_model_name}x{model_b_name}"
                    )
                    generations_a = _load_generations(gen_dir_a / f"{statement.id}.json")
                    generations_b = _load_generations(gen_dir_b / f"{statement.id}.json")

                if not generations_a or not generations_b:
                    logger.warning(
                        f"Could not load generations for statement '{statement.id}' for pair {model_a_name} vs {model_b_name}. Skipping."
                    )
                    continue

                ranking_results_for_statement = []
                for input_text in tqdm(
                    common_inputs, desc=f" Inputs for {statement_id}", leave=False
                ):
                    output_a = generations_a.get(input_text)
                    output_b = generations_b.get(input_text)

                    if output_a is None or output_b is None:
                        logger.warning(
                            f"Missing output for input in statement '{statement.id}' for pair {model_a_name} vs {model_b_name}. "
                            f"Input: '{input_text[:50]}...'. Skipping this input."
                        )
                        continue

                    # Call the ranking model
                    try:
                        rank_score = self.ranking_model.rank_responses(
                            statement=statement,
                            input_text=input_text,
                            output_a=output_a,
                            output_b=output_b,
                        )
                    except Exception as e:
                        logger.error(
                            f"Ranking model failed for statement '{statement.id}', pair {model_a_name} vs {model_b_name}: {e}"
                        )
                        # Decide how to handle errors: skip input, assign default score, etc.
                        # Skipping for now:
                        continue

                    ranking_results_for_statement.append(
                        {
                            "input": input_text,
                            "output_a": output_a,
                            "output_b": output_b,
                            "rank": rank_score,  # 1 for A, -1 for B, 0 for tie
                        }
                    )

                # Prepare final JSON output for the statement/pair
                output_data = {
                    "metadata": {
                        "evaluator_model": self.evaluator_model_name,
                        "model_a": model_a_name,
                        "model_b": model_b_name,
                        "ranking_model": self.ranking_model.get_info(),
                        "statement_id": statement.id,
                        "timestamp": datetime.datetime.now().isoformat(),
                    },
                    "rankings": ranking_results_for_statement,
                }

                # Save the JSON file
                try:
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=2)
                    if self.verbose:
                        logger.debug(f"Saved ranking results to {output_file_path}")
                except Exception as e:
                    logger.error(f"Failed to save ranking results to {output_file_path}: {e}")

        logger.info("Ranking pipeline finished.")
