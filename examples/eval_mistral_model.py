#!/usr/bin/env python3
"""
Example script demonstrating how to use the SpecEval framework
to evaluate compliance with the OpenAI Model Spec for a Mistral model via Together AI.

This script evaluates Mistral's 7B Instruct v0.3 model using GPT-4 as the default
evaluator and judge, but also supports Anthropic and Google models as judges.
"""

import os
import argparse
import logging
import datetime
from pathlib import Path
from typing import Any

from speceval import (
    JsonlParser,
    StandardPipeline,
)
from speceval.pipelines.standard_pipeline import CacheLevel
from speceval.orgs.together import MistralOrganization  # Import MistralOrganization
from speceval.utils.logging import setup_logging


def evaluate_organization(
    organization: Any,
    spec_path: Path,
    num_test_cases: int,
    num_inputs_per_statement: int,
    test_all_statements: bool,
    cache_level: CacheLevel,
    cache_dir: str,
    reports_dir: str,
    verbose: bool,
    logger: logging.Logger,
    pregenerated_inputs_dir: str | None,
) -> None:
    """
    Evaluate a single organization's models.

    Args:
        organization: The organization to evaluate
        spec_path: Path to the specification file
        num_test_cases: Number of test cases to generate
        num_inputs_per_statement: Number of inputs to generate per statement
        test_all_statements: Whether to test all statements
        cache_level: Level of caching to use
        cache_dir: Base directory for caching
        reports_dir: Directory for evaluation reports
        verbose: Whether to enable verbose logging
        logger: Configured logger
        pregenerated_inputs_dir: Optional path to pre-generated inputs
    """
    # Create the specification parser
    parser = JsonlParser()

    # Load the specification
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")

    spec = parser.from_file(spec_path)
    logger.info(f"Loaded specification with {len(spec.statements)} statements")

    # Create the pipeline
    pipeline = StandardPipeline(
        specification=spec,
        organization=organization,
        num_test_cases=num_test_cases,
        statements_to_test=spec.statements if test_all_statements else None,
        num_inputs_per_statement=num_inputs_per_statement,
        cache_level=cache_level,
        cache_dir=cache_dir,
        verbose=verbose,
        pregenerated_inputs_dir=pregenerated_inputs_dir,
    )

    # Run the evaluation
    org_info = organization.get_info()
    logger.info(
        f"Running evaluation for {org_info['name']} with "
        f"Judge: {org_info['judge_model']}, Evaluator: {org_info['evaluator_model']}..."
    )
    results = pipeline.run()

    # Print a summary
    summary = results.get_summary()
    logger.info(f"\nEvaluation Summary for {org_info['name']}:")
    logger.info(f"Total test cases: {summary['total_cases']}")
    logger.info(f"Compliant cases: {summary['compliant_cases']}")
    logger.info(f"Compliance rate: {summary['compliance_rate'] * 100:.2f}%")

    # Save reports
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize model names and create more descriptive filename
    candidate_model_name = org_info["candidate_model"]["model_name"].replace("/", "-")

    # Extract provider information from the model strings
    evaluator_provider, evaluator_model_name = org_info["evaluator_model"].split("/", 1)
    judge_provider, judge_model_name = org_info["judge_model"].split("/", 1)

    # Sanitize model names
    evaluator_model_name = evaluator_model_name.replace("/", "-")
    judge_model_name = judge_model_name.replace("/", "-")

    report_filename = (
        f"{candidate_model_name}_eval_{evaluator_provider}_{evaluator_model_name}_"
        f"judge_{judge_provider}_{judge_model_name}_{timestamp}.json"
    )
    report_path = reports_dir / report_filename

    # Save JSON report
    results.to_json(report_path)
    logger.info(f"Generated JSON report: {report_path.absolute()}")

    # Save HTML report
    html_path = report_path.with_suffix(".html")
    results.to_html(html_path)
    logger.info(f"Generated HTML report: {html_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Mistral model via Together AI for compliance with the OpenAI Model Spec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--spec-path",
        type=str,
        default="data/specs/openai/jsonl/openai.jsonl",
        help="Path to the OpenAI Model Spec JSONL file",
    )
    parser.add_argument(
        "--num-test-cases", type=int, default=5, help="Number of test cases to generate"
    )
    parser.add_argument(
        "--num-inputs-per-statement",
        type=int,
        default=1,
        help="Number of inputs to generate per statement",
    )
    parser.add_argument(
        "--test-all-statements", action="store_true", help="Test all statements instead of sampling"
    )
    parser.add_argument(
        "--cache-level",
        type=str,
        choices=["none", "inputs", "generations", "all"],
        default="all",
        help="Level of caching to use (none, inputs, generations, all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data",
        help="Base directory to store cached data",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="data/reports",
        help="Directory to store evaluation reports",
    )
    parser.add_argument(
        "--pregenerated-inputs-dir",
        type=str,
        default=None,
        help="Optional path to a directory containing pre-generated JSON inputs for statements (filename stem must match statement ID)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # API key arguments
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (alternatively, set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (alternatively, set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--google-api-key",
        type=str,
        default=None,
        help="Google API key (alternatively, set GOOGLE_API_KEY env var)",
    )

    # Judge configuration
    parser.add_argument(
        "--judge-provider",
        type=str,
        choices=["openai", "anthropic", "google"],
        default="openai",
        help="Provider for the judge model",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-2025-04-14",
        help="Model name for the judge",
    )

    args = parser.parse_args()

    # Configure logging
    logger = setup_logging(args.verbose, folder_name="eval_mistral_model")  # Updated folder name

    # Check for Together API key
    together_api_key = os.environ.get("TOGETHER_API_KEY")
    if not together_api_key:
        raise ValueError(
            "TOGETHER_API_KEY environment variable not set. Please set it before running this script."
        )

    # Get API keys from args or environment variables
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    # Validate required API keys based on providers
    if args.judge_provider == "openai" and not openai_api_key:
        raise ValueError(
            "OpenAI API key is required when using OpenAI judge. "
            "Either provide --openai-api-key or set OPENAI_API_KEY env var."
        )
    if args.judge_provider == "anthropic" and not anthropic_api_key:
        raise ValueError(
            "Anthropic API key is required when using Anthropic judge. "
            "Either provide --anthropic-api-key or set ANTHROPIC_API_KEY env var."
        )

    # Map string cache level to enum
    cache_level_map = {
        "none": CacheLevel.NONE,
        "inputs": CacheLevel.INPUTS,
        "generations": CacheLevel.GENERATIONS,
        "all": CacheLevel.ALL,
    }
    cache_level = cache_level_map[args.cache_level]

    # Create the Mistral organization
    mistral_org = MistralOrganization(
        api_key=together_api_key,
        judge_provider=args.judge_provider,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        judge_model_name=args.judge_model,
    )

    # Evaluate the organization
    try:
        evaluate_organization(
            organization=mistral_org,
            spec_path=Path(args.spec_path),
            num_test_cases=args.num_test_cases,
            num_inputs_per_statement=args.num_inputs_per_statement,
            test_all_statements=args.test_all_statements,
            cache_level=cache_level,
            cache_dir=args.cache_dir,
            reports_dir=args.reports_dir,
            verbose=args.verbose,
            logger=logger,
            pregenerated_inputs_dir=args.pregenerated_inputs_dir,
        )
    except Exception as e:
        logger.error(f"Error evaluating {mistral_org.get_info()['name']}: {e}")

    logger.info("\nCompleted evaluation.")


if __name__ == "__main__":
    main()
