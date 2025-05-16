"""Standard evaluation pipeline implementation for the SpecEval framework."""
import datetime
import json
import logging
import random
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from ..base import Pipeline, Specification, Organization, Statement, EvaluationResults, TestCase
from ..autobencher import AutoBencher, ModelInputGenerator
from ..models.openai import OpenAIEvaluatorModel


# Set up logging
logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Enum defining different caching levels for the evaluation pipeline."""

    NONE = 0  # No caching
    INPUTS = 1  # Cache only inputs
    GENERATIONS = 2  # Cache inputs and generations
    ALL = 3  # Cache inputs, generations, and judgments


class StandardPipeline(Pipeline):
    """Standard implementation of an evaluation pipeline."""

    def __init__(
        self,
        specification: Specification,
        organization: Organization,
        num_test_cases: int = 100,
        statements_to_test: Optional[Union[List[str], List[Statement]]] = None,
        num_inputs_per_statement: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
        cache_level: CacheLevel = CacheLevel.ALL,
        cache_dir: str = "data",
        verbose: bool = False,
        pregenerated_inputs_dir: Optional[str] = None,
    ):
        """
        Initialize the standard pipeline.

        Args:
            specification: The specification to evaluate against.
            organization: The organization providing the models.
            num_test_cases: Total number of test cases to generate.
            statements_to_test: Optional list of statements to test. If provided as strings, interpreted as statement IDs.
                                If not provided, randomly selects statements to reach num_test_cases.
            num_inputs_per_statement: Number of inputs to generate for each statement.
            metadata: Optional metadata about the evaluation.
            cache_level: The level of caching to use (none, inputs, generations, all).
            cache_dir: Base directory to store cached data. Subdirectories will be created for different types of cached data.
            verbose: Whether to enable verbose logging.
            pregenerated_inputs_dir: Optional directory containing pre-generated inputs.
        """
        super().__init__(specification, organization, num_test_cases, metadata)

        # Convert statement IDs to Statement objects if needed
        if statements_to_test is not None:
            if all(isinstance(s, str) for s in statements_to_test):
                # Convert IDs to Statement objects
                self.statements_to_test = [
                    self.specification.get_statement(s)
                    for s in statements_to_test
                    if self.specification.get_statement(s) is not None
                ]
            else:
                # Already Statement objects
                self.statements_to_test = statements_to_test
        else:
            # Randomly select statements to reach num_test_cases
            # If not enough statements, each statement gets multiple inputs
            all_statements = self.specification.statements

            if len(all_statements) * num_inputs_per_statement <= num_test_cases:
                # Use all statements
                self.statements_to_test = all_statements
            else:
                # Randomly select statements
                num_statements = num_test_cases // num_inputs_per_statement
                self.statements_to_test = random.sample(all_statements, num_statements)

        self.num_inputs_per_statement = num_inputs_per_statement
        self.cache_level = cache_level
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose
        self.pregenerated_inputs_dir = (
            Path(pregenerated_inputs_dir) if pregenerated_inputs_dir else None
        )
        self.pregenerated_inputs: Dict[str, List[str]] = {}

        # Set up cache directories
        self.autobencher_cache_dir = self.cache_dir / "autobencher_outputs"
        self.generations_cache_dir = self.cache_dir / "generations"
        self.judgements_cache_dir = self.cache_dir / "judgements"

        # Create cache directories if they don't exist
        if self.cache_level != CacheLevel.NONE:
            self._ensure_cache_dirs_exist()

        # Load pre-generated inputs if directory is provided
        if self.pregenerated_inputs_dir:
            self._load_pregenerated_inputs()

    def _ensure_cache_dirs_exist(self):
        """Create all necessary cache directories if they don't exist."""
        if self.cache_level != CacheLevel.NONE:
            self.autobencher_cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logger.info(
                    f"Ensured autobencher cache directory exists at {self.autobencher_cache_dir}"
                )

        if self.cache_level in [CacheLevel.GENERATIONS, CacheLevel.ALL]:
            self.generations_cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logger.info(
                    f"Ensured generations cache directory exists at {self.generations_cache_dir}"
                )

        if self.cache_level == CacheLevel.ALL:
            self.judgements_cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logger.info(
                    f"Ensured judgements cache directory exists at {self.judgements_cache_dir}"
                )

    def _load_pregenerated_inputs(self):
        """Load inputs from JSON files in the pregenerated_inputs_dir."""
        if not self.pregenerated_inputs_dir or not self.pregenerated_inputs_dir.is_dir():
            logger.warning(
                f"Pregenerated inputs directory not found or not a directory: {self.pregenerated_inputs_dir}"
            )
            return

        logger.info(f"Loading pre-generated inputs from {self.pregenerated_inputs_dir}...")
        loaded_count = 0
        for file_path in self.pregenerated_inputs_dir.glob("*.json"):
            statement_id = file_path.stem  # Use filename stem as statement ID
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "inputs" in data and isinstance(data["inputs"], list):
                    self.pregenerated_inputs[statement_id] = [str(item) for item in data["inputs"]]
                    loaded_count += 1
                    if self.verbose:
                        logger.debug(
                            f"Loaded {len(self.pregenerated_inputs[statement_id])} inputs for statement '{statement_id}' from {file_path.name}"
                        )
                else:
                    logger.warning(
                        f"Skipping file {file_path.name}: 'inputs' key not found or not a list."
                    )
            except json.JSONDecodeError:
                logger.warning(f"Skipping file {file_path.name}: Invalid JSON.")
            except Exception as e:
                logger.warning(f"Skipping file {file_path.name}: Error reading file - {e}")

        logger.info(
            f"Finished loading pre-generated inputs. Found inputs for {loaded_count} statements."
        )

    def _get_model_combination_dir(self, is_judgement: bool = False) -> Path:
        """Get the cache directory for a specific statement based on the models involved."""
        # Get model info
        candidate_model = self.organization.get_candidate_model()
        evaluator_model = self.organization.get_evaluator_model()

        candidate_info = candidate_model.get_info()
        evaluator_info = evaluator_model.get_info()

        # Create identifiers for each model
        evaluator_identifier = evaluator_info.get("model_name", "unknown").replace("/", "-")
        candidate_identifier = candidate_info.get("model_name", "unknown").replace("/", "-")

        # Create a directory name that includes all models involved
        if is_judgement:
            # For judgements, use judge provider as prefix directory
            judge_provider = (
                self.organization.judge_provider
                if hasattr(self.organization, "judge_provider")
                else "unknown"
            )
            judge_provider = judge_provider.upper()

            # Structure: judgements/JUDGE_PROVIDER/evaluatorxcandidate
            dir_name = f"{evaluator_identifier}x{candidate_identifier}"
            base_dir = self.judgements_cache_dir / f"JUDGE_{judge_provider}"
        else:
            # For generations, include just evaluator and candidate
            dir_name = f"{evaluator_identifier}x{candidate_identifier}"
            base_dir = self.generations_cache_dir

        return base_dir / dir_name

    def _get_generation_cache_path(self, statement: Statement) -> Path:
        """Get the path to the cache file for generations for a specific statement."""
        cache_dir = self._get_model_combination_dir(is_judgement=False)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{statement.id}.json"

    def _get_judgement_cache_path(self, statement: Statement) -> Path:
        """Get the path to the cache file for judgements for a specific statement."""
        cache_dir = self._get_model_combination_dir(is_judgement=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{statement.id}.json"

    def _load_generations_from_cache(self, statement: Statement) -> Dict[str, str]:
        """
        Load cached generations for a specific statement.

        Returns a dictionary mapping input_text to output_text.
        """
        if self.cache_level not in [CacheLevel.GENERATIONS, CacheLevel.ALL]:
            return {}

        cache_path = self._get_generation_cache_path(statement)

        try:
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)

                # Verify that the evaluator model matches
                evaluator_model = self.organization.get_evaluator_model()
                cached_evaluator_info = cache_data["metadata"].get("evaluator_model", {})
                current_evaluator_info = evaluator_model.get_info()

                if cached_evaluator_info.get("model_name") != current_evaluator_info.get(
                    "model_name"
                ):
                    if self.verbose:
                        logger.info(
                            f"Cache mismatch: evaluator model changed from "
                            f"{cached_evaluator_info.get('model_name')} to "
                            f"{current_evaluator_info.get('model_name')}"
                        )
                    return {}

                if self.verbose:
                    logger.info(
                        f"Loaded {len(cache_data['generations'])} cached generations for statement {statement.id}"
                    )

                return {item["input"]: item["output"] for item in cache_data["generations"]}
            else:
                if self.verbose:
                    logger.info(f"No generation cache found for statement {statement.id}")
        except Exception as e:
            logger.warning(f"Error loading generation cache for statement {statement.id}: {e}")

        return {}

    def _save_generations_to_cache(
        self, statement: Statement, input_output_pairs: List[Dict[str, str]]
    ):
        """Save generated responses to cache for a specific statement."""
        if self.cache_level not in [CacheLevel.GENERATIONS, CacheLevel.ALL]:
            return

        cache_path = self._get_generation_cache_path(statement)

        try:
            candidate_model = self.organization.get_candidate_model()
            evaluator_model = self.organization.get_evaluator_model()

            candidate_info = candidate_model.get_info()
            evaluator_info = evaluator_model.get_info()

            metadata = {
                "candidate_model": candidate_info,
                "evaluator_model": evaluator_info,  # Add evaluator model info
                "timestamp": datetime.datetime.now().isoformat(),
                "statement_id": statement.id,
            }

            cache_data = {"metadata": metadata, "generations": input_output_pairs}

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            if self.verbose:
                logger.info(
                    f"Saved {len(input_output_pairs)} generations to cache for statement {statement.id}"
                )
        except Exception as e:
            logger.warning(f"Error saving generation cache for statement {statement.id}: {e}")

    def _load_judgements_from_cache(self, statement: Statement) -> Dict[str, Dict[str, Any]]:
        """
        Load cached judgements for a specific statement.

        Returns a dictionary mapping (input_text, output_text) tuple to judgment data.
        """
        if self.cache_level != CacheLevel.ALL:
            return {}

        cache_path = self._get_judgement_cache_path(statement)

        try:
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)

                # Verify that the evaluator model matches
                evaluator_model = self.organization.get_evaluator_model()
                cached_evaluator_info = cache_data["metadata"].get("evaluator_model", {})
                current_evaluator_info = evaluator_model.get_info()

                # Also verify the judge model matches
                judge_model = self.organization.get_judge_model()
                cached_judge_info = cache_data["metadata"].get("judge_model", {})
                current_judge_info = judge_model.get_info()

                if cached_evaluator_info.get("model_name") != current_evaluator_info.get(
                    "model_name"
                ) or cached_judge_info.get("model_name") != current_judge_info.get("model_name"):
                    if self.verbose:
                        logger.info(
                            f"Cache mismatch: model changed from "
                            f"{cached_evaluator_info.get('model_name')}/{cached_judge_info.get('model_name')} to "
                            f"{current_evaluator_info.get('model_name')}/{current_judge_info.get('model_name')}"
                        )
                    return {}

                if self.verbose:
                    logger.info(
                        f"Loaded {len(cache_data['judgements'])} cached judgements for statement {statement.id}"
                    )

                # Create a dictionary where the key is a tuple of (input, output)
                return {
                    (item["input"], item["output"]): {
                        "compliant": item["compliant"],
                        "confidence": item["confidence"],
                        "explanation": item["explanation"],
                    }
                    for item in cache_data["judgements"]
                }
            else:
                if self.verbose:
                    logger.info(f"No judgement cache found for statement {statement.id}")
        except Exception as e:
            logger.warning(f"Error loading judgement cache for statement {statement.id}: {e}")

        return {}

    def _save_judgements_to_cache(self, statement: Statement, judgements: List[Dict[str, Any]]):
        """Save judgements to cache for a specific statement."""
        if self.cache_level != CacheLevel.ALL:
            return

        cache_path = self._get_judgement_cache_path(statement)

        try:
            candidate_model = self.organization.get_candidate_model()
            judge_model = self.organization.get_judge_model()
            evaluator_model = self.organization.get_evaluator_model()

            candidate_info = candidate_model.get_info()
            judge_info = judge_model.get_info()
            evaluator_info = evaluator_model.get_info()

            metadata = {
                "candidate_model": candidate_info,
                "judge_model": judge_info,
                "evaluator_model": evaluator_info,  # Add evaluator model info
                "timestamp": datetime.datetime.now().isoformat(),
                "statement_id": statement.id,
            }

            cache_data = {"metadata": metadata, "judgements": judgements}

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            if self.verbose:
                logger.info(
                    f"Saved {len(judgements)} judgements to cache for statement {statement.id}"
                )
        except Exception as e:
            logger.warning(f"Error saving judgement cache for statement {statement.id}: {e}")

    def run(self) -> EvaluationResults:
        """Run the evaluation pipeline."""
        # Get the models from the organization
        candidate_model = self.organization.get_candidate_model()
        evaluator_model = self.organization.get_evaluator_model()
        judge_model = self.organization.get_judge_model()

        # If no evaluator model is provided by the organization, use GPT-4 as default
        if evaluator_model is None:
            try:
                evaluator_model = OpenAIEvaluatorModel()
            except ValueError as e:
                raise ValueError(
                    "No evaluator model provided by the organization and couldn't create a default OpenAI model. "
                    f"Original error: {str(e)}"
                )

        # --- Input Generation ---
        inputs_by_statement: Dict[str, List[str]] = {}
        if self.pregenerated_inputs_dir and self.pregenerated_inputs:
            logger.info(f"Using pre-generated inputs from {self.pregenerated_inputs_dir}")
            for statement in self.statements_to_test:
                inputs = self.pregenerated_inputs.get(statement.id)
                if inputs:
                    # Use num_inputs_per_statement to limit loaded inputs if > 0
                    inputs_to_use = (
                        inputs[: self.num_inputs_per_statement]
                        if self.num_inputs_per_statement > 0
                        else inputs
                    )
                    inputs_by_statement[statement.id] = inputs_to_use
                    if self.verbose:
                        logger.debug(
                            f"Using {len(inputs_to_use)} pre-generated inputs for statement {statement.id}"
                        )
                else:
                    logger.warning(
                        f"No pre-generated inputs found for statement {statement.id} in {self.pregenerated_inputs_dir}. Will skip."
                    )
                    inputs_by_statement[statement.id] = []  # Assign empty list if not found
        else:
            if self.verbose:
                logger.info("Generating inputs using AutoBencher.")
            # Initialize an AutoBencher with the evaluator model
            autobencher = AutoBencher(
                generators=[ModelInputGenerator(evaluator_model)],
                use_cache=self.cache_level != CacheLevel.NONE,
                cache_dir=str(self.autobencher_cache_dir),
                verbose=self.verbose,
            )

            # Generate inputs for the statements
            inputs_by_statement = autobencher.generate_inputs(
                self.statements_to_test, num_inputs_per_statement=self.num_inputs_per_statement
            )
        # --- End Input Generation ---

        # Create test cases
        test_cases = []

        # Count total number of test cases to create
        total_inputs = sum(len(inputs) for inputs in inputs_by_statement.values())

        # Create a progress bar for all test cases
        with tqdm(total=total_inputs, desc="Processing test cases") as pbar:
            for statement in self.statements_to_test:
                inputs = inputs_by_statement.get(statement.id, [])

                # Load cached generations if available
                cached_generations = self._load_generations_from_cache(statement)
                # Load cached judgements if available
                cached_judgements = self._load_judgements_from_cache(statement)

                # Track new generations and judgements to cache
                new_generations = []
                new_judgements = []

                for input_text in inputs:
                    # Check if we already have this generation cached
                    if input_text in cached_generations and self.cache_level in [
                        CacheLevel.GENERATIONS,
                        CacheLevel.ALL,
                    ]:
                        output_text = cached_generations[input_text]
                        pbar.set_description(f"Using cached generation for {statement.id}")
                    else:
                        # Get a response from the candidate model
                        pbar.set_description(
                            f"Candidate model generating response for {statement.id}"
                        )
                        output_text = candidate_model.generate(input_text)
                        # Add to new generations for caching
                        new_generations.append({"input": input_text, "output": output_text})

                    # Check if we already have this judgement cached
                    cache_key = (input_text, output_text)
                    if cache_key in cached_judgements and self.cache_level == CacheLevel.ALL:
                        evaluation = cached_judgements[cache_key]
                        pbar.set_description(f"Using cached judgement for {statement.id}")
                    else:
                        # Evaluate compliance
                        pbar.set_description(
                            f"Judge model evaluating compliance for {statement.id}"
                        )
                        evaluation = judge_model.evaluate_compliance(
                            statement, input_text, output_text
                        )
                        # Add to new judgements for caching
                        new_judgements.append(
                            {
                                "input": input_text,
                                "output": output_text,
                                "compliant": evaluation["compliant"],
                                "confidence": evaluation["confidence"],
                                "explanation": evaluation["explanation"],
                            }
                        )

                    # Create a test case
                    test_case = TestCase(
                        statement=statement,
                        input_text=input_text,
                        output_text=output_text,
                        is_compliant=evaluation["compliant"],
                        confidence=evaluation["confidence"],
                        explanation=evaluation["explanation"],
                        metadata={},
                    )

                    test_cases.append(test_case)
                    pbar.update(1)

                # Cache new generations
                if new_generations and self.cache_level in [CacheLevel.GENERATIONS, CacheLevel.ALL]:
                    self._save_generations_to_cache(statement, new_generations)

                # Cache new judgements
                if new_judgements and self.cache_level == CacheLevel.ALL:
                    self._save_judgements_to_cache(statement, new_judgements)

        # Create and return results
        return EvaluationResults(
            specification=self.specification,
            organization=self.organization,
            test_cases=test_cases,
            timestamp=datetime.datetime.now(),
            metadata=self.metadata,
        )
