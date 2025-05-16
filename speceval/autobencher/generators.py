"""Input generators for the AutoBencher component of SpecEval."""
from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from ..base import Statement, EvaluatorModel


# Set up logging
logger = logging.getLogger(__name__)


class InputGenerator(ABC):
    """Abstract base class for input generators."""

    @abstractmethod
    def generate(
        self,
        statement: Statement,
        previous_inputs: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate an input that might challenge compliance with a statement.

        Args:
            statement: The statement to generate an input for.
            previous_inputs: Optional list of previously generated inputs for this statement.
            **kwargs: Additional arguments.

        Returns:
            A generated input string.
        """
        pass

    def get_id(self) -> str:
        """Return a unique identifier for this generator."""
        return self.__class__.__name__


class ModelInputGenerator(InputGenerator):
    """Generate inputs using an evaluator model."""

    def __init__(self, model: EvaluatorModel):
        """
        Initialize the model input generator.

        Args:
            model: The evaluator model to use for generating inputs.
        """
        self.model = model

    def generate(
        self,
        statement: Statement,
        previous_inputs: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate an input using the evaluator model.

        Args:
            statement: The statement to generate an input for.
            previous_inputs: Optional list of previously generated inputs for this statement.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            A generated input string.
        """
        return self.model.generate_challenging_input(
            statement, previous_inputs=previous_inputs, **kwargs
        )

    def get_id(self) -> str:
        """Return a unique identifier based on the model."""
        model_info = self.model.get_info()
        # Use model name and version as the unique identifier
        if "model_name" in model_info:
            return f"{self.__class__.__name__}_{model_info['model_name']}"
        return f"{self.__class__.__name__}_{id(self.model)}"


class AutoBencher:
    """Class for generating challenging inputs using evaluator models."""

    def __init__(
        self,
        generators: List[InputGenerator],
        use_cache: bool = True,
        cache_dir: str = "data/autobencher_outputs",
        verbose: bool = False,
    ):
        """
        Initialize the AutoBencher.

        Args:
            generators: List of input generators. Must contain at least one generator.
            use_cache: Whether to use caching for generated inputs.
            cache_dir: Directory to store cached inputs.
            verbose: Whether to log detailed information.

        Raises:
            ValueError: If the generators list is empty.
        """
        if not generators:
            raise ValueError("AutoBencher requires at least one InputGenerator")

        self.generators = generators
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose

        # Set up logging level based on verbose flag
        if verbose:
            logging.basicConfig(level=logging.INFO)

        # Create cache directory if it doesn't exist and caching is enabled
        if use_cache:
            self._ensure_cache_dir_exists()

    def _ensure_cache_dir_exists(self):
        """Create cache directory if it doesn't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logger.info(f"Ensured cache directory exists at {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to create cache directory: {e}")
            self.use_cache = False
            logger.warning("Caching disabled due to directory creation failure")

    def _get_cache_path(self, generator: InputGenerator, statement: Statement) -> Path:
        """
        Get the path to the cache file for a specific generator and statement.

        Args:
            generator: The input generator.
            statement: The statement.

        Returns:
            Path to the cache file.
        """
        generator_id = generator.get_id()
        # Create a directory for this generator
        generator_dir = self.cache_dir / generator_id
        generator_dir.mkdir(exist_ok=True)

        # Create a cache file for this statement
        return generator_dir / f"{statement.id}_inputs.json"

    def _load_from_cache(self, generator: InputGenerator, statement: Statement) -> List[str]:
        """
        Load inputs from cache for a specific generator and statement.

        Args:
            generator: The input generator.
            statement: The statement.

        Returns:
            List of cached inputs or empty list if no cache exists.
        """
        if not self.use_cache:
            return []

        cache_path = self._get_cache_path(generator, statement)

        try:
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)

                    if self.verbose:
                        logger.info(
                            f"Loaded {len(cache_data['inputs'])} cached inputs for "
                            f"statement {statement.id}"
                        )

                    return cache_data["inputs"]
            else:
                if self.verbose:
                    logger.info(
                        f"No cache found for statement {statement.id} with "
                        f"generator {generator.get_id()}"
                    )
        except Exception as e:
            logger.warning(f"Error loading cache for statement {statement.id}: {e}")

        return []

    def _save_to_cache(self, generator: InputGenerator, statement: Statement, inputs: List[str]):
        """
        Save inputs to cache for a specific generator and statement.

        Args:
            generator: The input generator.
            statement: The statement.
            inputs: List of inputs to cache.
        """
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(generator, statement)

        try:
            # Get generator-specific metadata
            metadata = {
                "generator_type": generator.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
            }

            # If it's a model generator, add model info
            if isinstance(generator, ModelInputGenerator):
                metadata.update({"evaluator_model": generator.get_id()})

            cache_data = {"metadata": metadata, "inputs": inputs}

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            if self.verbose:
                logger.info(f"Saved {len(inputs)} inputs to cache for statement {statement.id}")
        except Exception as e:
            logger.warning(f"Error saving cache for statement {statement.id}: {e}")

    def generate_inputs(
        self, statements: List[Statement], num_inputs_per_statement: int = 5, **kwargs
    ) -> Dict[str, List[str]]:
        """
        Generate inputs for multiple statements, using cache when available.

        This method ensures that each statement receives the specified number of challenging inputs
        by distributing the generation across all available generators. If there are multiple
        generators, each generator will create approximately num_inputs_per_statement / num_generators
        inputs per statement, ensuring a balanced and diverse set of challenging inputs.

        Args:
            statements: List of statements to generate inputs for.
            num_inputs_per_statement: Number of inputs to generate for each statement.
                This is the total number of inputs across all generators.
            **kwargs: Additional arguments to pass to the generators.

        Returns:
            A dictionary mapping statement IDs to lists of generated inputs.
        """
        result = {}

        # Calculate inputs per generator for each statement
        inputs_per_generator = max(1, num_inputs_per_statement // len(self.generators))
        if self.verbose and len(self.generators) > 1:
            logger.info(
                f"Distributing {num_inputs_per_statement} inputs across {len(self.generators)} generators"
            )
            logger.info(f"Each generator will produce {inputs_per_generator} inputs per statement")

        # Add progress bar for statements
        for statement in tqdm(statements, desc="AutoBencher: Generating inputs for statements"):
            if self.verbose:
                logger.info(f"Generating inputs for statement {statement.id}")

            all_inputs = []

            for generator in self.generators:
                generator_id = generator.get_id()
                if self.verbose:
                    logger.info(f"Using generator {generator_id}")

                # Try to load from cache first
                cached_inputs = self._load_from_cache(generator, statement)

                # Check if we have enough cached inputs
                if len(cached_inputs) >= inputs_per_generator:
                    if self.verbose:
                        logger.info(
                            f"Using {inputs_per_generator} cached inputs for "
                            f"statement {statement.id}"
                        )

                    # Use the first inputs_per_generator cached inputs
                    all_inputs.extend(cached_inputs[:inputs_per_generator])
                else:
                    # We need to generate more inputs
                    num_to_generate = inputs_per_generator - len(cached_inputs)

                    if self.verbose:
                        if cached_inputs:
                            logger.info(
                                f"Found {len(cached_inputs)} cached inputs, generating "
                                f"{num_to_generate} more"
                            )
                        else:
                            logger.info(
                                f"No cached inputs found, generating {num_to_generate} new inputs"
                            )

                    # Add the cached inputs we have
                    new_inputs = list(cached_inputs)

                    # Generate additional inputs, passing previous inputs to ensure diversity
                    # Add inner progress bar for generating individual inputs
                    for _ in tqdm(
                        range(num_to_generate),
                        desc=f"Generating inputs for statement {statement.id}",
                        leave=False,
                    ):
                        # Pass all previously generated inputs for this statement with this generator
                        # This helps the model generate more diverse inputs
                        previous_inputs = new_inputs if new_inputs else None
                        new_input = generator.generate(
                            statement, previous_inputs=previous_inputs, **kwargs
                        )
                        new_inputs.append(new_input)

                    # Save all inputs to cache
                    self._save_to_cache(generator, statement, new_inputs)

                    # Add to the result
                    all_inputs.extend(new_inputs)

            result[statement.id] = all_inputs

        return result
