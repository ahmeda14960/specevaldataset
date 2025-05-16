"""Script to validate input consistency across model generations.

Checks that all models have the same input questions for each specification and statement."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Optional

from speceval.utils.parsing import ALL_KNOWN_MODEL_NAMES, extract_model_name_from_path
from speceval.utils.logging import setup_logging


def find_results_dir(base_dir: Path, max_depth: int = 3) -> Optional[Path]:
    """Recursively search for a results directory up to max_depth levels deep."""
    if max_depth <= 0:
        return None
        
    # Check if current directory is named 'results'
    if base_dir.name == "results" and any(f.suffix == '.json' for f in base_dir.glob('*.json')):
        return base_dir
        
    # Look in immediate subdirectories
    for child in base_dir.iterdir():
        if child.is_dir():
            if child.name == "results" and any(f.suffix == '.json' for f in child.glob('*.json')):
                return child
            # Recursively check deeper
            result = find_results_dir(child, max_depth - 1)
            if result:
                return result
    
    return None


def get_latest_run_dir(model_dir: Path) -> Path:
    """Get the most recent results directory for a model."""
    run_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise ValueError(f"No run directories found in {model_dir}")
    
    # Sort by directory name (assuming YYYYMMDD_HHMMSS format)
    run_dirs.sort(key=lambda x: x.name)
    
    # Try each run directory from newest to oldest
    for run_dir in reversed(run_dirs):
        results_dir = find_results_dir(run_dir)
        if results_dir:
            return results_dir
            
    raise ValueError(f"No results directory with JSON files found in any subdirectory of {model_dir}")


def load_inputs_from_json(json_file: Path, logger) -> List[str]:
    """Load and return list of input texts from a results JSON file, preserving order."""
    try:
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item["input_text"] for item in data if "input_text" in item]
            return []
    except Exception as e:
        logger.error(f"Error reading {json_file}: {e}")
        return []


def validate_inputs(base_dir: Path, logger):
    """Validate input consistency across all models and specifications."""
    # Structure: statement_id -> model -> list[inputs]
    inputs_by_statement: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    
    # Track which models we successfully processed
    processed_models = set()
    
    # First, collect all inputs
    for spec_dir in base_dir.iterdir():
        if not spec_dir.is_dir():
            continue
            
        spec_name = spec_dir.name
        logger.info(f"\nProcessing specification: {spec_name}")
        
        for model_dir in spec_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            try:
                model_name = extract_model_name_from_path(model_dir)
                results_dir = get_latest_run_dir(model_dir)
                
                logger.info(f"  Processing model: {model_name}")
                logger.debug(f"    Using results from: {results_dir}")
                
                json_files = list(results_dir.glob("*.json"))
                if not json_files:
                    logger.warning(f"    No JSON files found in {results_dir}")
                    continue
                    
                logger.info(f"    Found {len(json_files)} statement files")
                
                for json_file in json_files:
                    statement_id = json_file.stem
                    inputs = load_inputs_from_json(json_file, logger)
                    if inputs:
                        inputs_by_statement[statement_id][model_name] = inputs
                        processed_models.add(model_name)
                    else:
                        logger.warning(f"    No inputs found in {json_file.name}")
                    
            except ValueError as e:
                logger.warning(f"  Skipping directory {model_dir.name}: {e}")
                continue
    
    if not processed_models:
        logger.error("\nNo models were successfully processed! Check your directory structure.")
        return
        
    logger.info(f"\nSuccessfully processed {len(processed_models)} models: {', '.join(sorted(processed_models))}")
    
    # Now validate consistency
    discrepancies_found = False
    
    logger.info("\nValidating input consistency across models:")
    for statement_id, model_inputs in inputs_by_statement.items():
        logger.info(f"\nChecking {statement_id}:")
        
        # Get all unique inputs across all models
        all_inputs = set()
        for inputs in model_inputs.values():
            all_inputs.update(inputs)
        
        # Check each input against each model
        for input_text in sorted(all_inputs):
            models_with_input = set()
            models_without_input = set()
            
            for model_name, inputs in model_inputs.items():
                if input_text in inputs:
                    models_with_input.add(model_name)
                else:
                    models_without_input.add(model_name)
            
            if models_without_input:  # If any model is missing this input
                discrepancies_found = True
                logger.warning(f"\n  Input discrepancy found:")
                logger.warning(f"    Input text: {input_text[:200]}...")
                logger.info(f"    Present in models: {', '.join(sorted(models_with_input))}")
                logger.warning(f"    Missing from models: {', '.join(sorted(models_without_input))}")
    
    if not discrepancies_found:
        logger.info("\nAll inputs are consistent across models and specifications! ✨")
    else:
        logger.warning("\nDiscrepancies found - please check the details above.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate input consistency across model generations."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/batched_generations"),
        help="Base directory containing spec subdirectories (default: data/batched_generations)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose, folder_name="input_validation")
    
    validate_inputs(args.base_dir, logger)


if __name__ == "__main__":
    main()
