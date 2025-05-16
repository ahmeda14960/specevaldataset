#!/usr/bin/env python3
"""
Test script to evaluate the AutoBencher's ability to generate challenging inputs.
This script allows you to see the quality of inputs being generated for specified statements
without running the full evaluation pipeline.
"""

import argparse
import os
import json
from pathlib import Path
import random
from typing import List

from speceval import OpenAI
from speceval.parsers import JsonlParser
from speceval.autobencher import AutoBencher, ModelInputGenerator
from speceval.base import Statement, StatementType


def print_statement(statement: Statement, detailed: bool = False):
    """Print a statement in a formatted way."""
    print(f"\n{'=' * 80}")
    print(f"Statement ID: {statement.id}")
    print(f"Type: {statement.type.value}")
    print(f"Authority Level: {statement.authority_level.value}")

    # Format the text for better readability
    text = statement.text.strip()
    print(f'Statement: "{text}"')

    if detailed:
        if statement.section:
            print(f"Section: {statement.section}")
        if statement.subsection:
            print(f"Subsection: {statement.subsection}")
        if hasattr(statement, "examples") and statement.examples:
            print("\nExamples:")
            for i, example in enumerate(statement.examples):
                print(f"  Example {i+1}: {example}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the AutoBencher's input generation capabilities"
    )
    parser.add_argument(
        "--spec-path",
        type=str,
        default="data/specs/openai_model_spec.jsonl",
        help="Path to the specification JSONL file",
    )
    parser.add_argument(
        "--num-statements",
        type=int,
        default=5,
        help="Number of random statements to test",
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=3,
        help="Number of inputs to generate per statement",
    )
    parser.add_argument(
        "--statement-ids",
        type=str,
        nargs="*",
        help="Specific statement IDs to test (overrides --num-statements)",
    )
    parser.add_argument(
        "--statement-type",
        type=str,
        choices=["prohibition", "action_constraint", "output_constraint", "capability"],
        help="Filter statements by type",
    )
    parser.add_argument(
        "--authority-level",
        type=str,
        choices=["platform", "user", "organization"],
        help="Filter statements by authority level",
    )
    parser.add_argument(
        "--section",
        type=str,
        help="Filter statements by section name",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        help="Filter statements containing keyword in their ID or text",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (alternatively, set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="gpt-4o-2024-08-06",
        help="Model to use for generating inputs",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Optional file to save results to (JSON format)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed statement information",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable caching of generated inputs",
    )

    args = parser.parse_args()

    # If API key is provided, use it; otherwise use environment variable
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Either provide --openai-api-key or set OPENAI_API_KEY env var."
        )

    # Load the specification using JsonlParser
    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")

    spec = JsonlParser.from_file(spec_path)
    print(f"Loaded specification with {len(spec.statements)} statements")

    # Apply filters if specified
    filtered_statements = spec.statements

    # Filter by statement type
    if args.statement_type:
        statement_type_map = {
            "prohibition": StatementType.PROHIBITION,
            "action_constraint": StatementType.ACTION_CONSTRAINT,
            "output_constraint": StatementType.OUTPUT_CONSTRAINT,
            "capability": StatementType.CAPABILITY,
        }
        statement_type = statement_type_map[args.statement_type]
        filtered_statements = [s for s in filtered_statements if s.type == statement_type]
        print(f"Filtered to {len(filtered_statements)} statements of type {args.statement_type}")

    # Filter by authority level
    if args.authority_level:
        filtered_statements = spec.get_statements_by_authority(args.authority_level)
        print(
            f"Filtered to {len(filtered_statements)} statements with authority level {args.authority_level}"
        )

    # Filter by section
    if args.section:
        section_statements = spec.get_statements_by_section(args.section)
        filtered_statements = [s for s in filtered_statements if s in section_statements]
        print(f"Filtered to {len(filtered_statements)} statements in section '{args.section}'")

    # Filter by keyword
    if args.keyword:
        keyword = args.keyword.lower()
        filtered_statements = [
            s for s in filtered_statements if keyword in s.id.lower() or keyword in s.text.lower()
        ]
        print(f"Filtered to {len(filtered_statements)} statements containing '{args.keyword}'")

    # Create the evaluator model
    organization = OpenAI(
        evaluator_model_name=args.evaluator_model,
        api_key=api_key,
    )
    evaluator_model = organization.get_evaluator_model()

    # Initialize AutoBencher
    autobencher = AutoBencher(
        generators=[ModelInputGenerator(evaluator_model)],
        use_cache=not args.disable_cache,
        verbose=True,
    )

    # Select statements to test
    statements_to_test: List[Statement] = []

    if args.statement_ids:
        # Use specific statement IDs
        for statement_id in args.statement_ids:
            statement = spec.get_statement(statement_id)
            if statement:
                statements_to_test.append(statement)
            else:
                print(f"Warning: Statement ID '{statement_id}' not found")
    else:
        # Randomly select statements from filtered list
        if filtered_statements:
            statements_to_test = random.sample(
                filtered_statements, min(args.num_statements, len(filtered_statements))
            )
        else:
            print("No statements match the specified filters")
            return

    print(f"Selected {len(statements_to_test)} statements to test")

    # Generate inputs for each statement
    results = {}

    for statement in statements_to_test:
        print_statement(statement, args.detailed)

        # Generate inputs
        print(f"\nGenerating {args.num_inputs} inputs for statement {statement.id}...")
        inputs = autobencher.generate_inputs(
            [statement], num_inputs_per_statement=args.num_inputs
        ).get(statement.id, [])

        # Print the generated inputs
        print(f"\nGenerated inputs for statement {statement.id}:")
        for i, input_text in enumerate(inputs):
            print(f"\nInput {i+1}:")
            print(f"{input_text}")

        # Store the results
        results[statement.id] = {
            "statement_type": statement.type.value,
            "authority_level": statement.authority_level.value,
            "statement_text": statement.text,
            "inputs": inputs,
        }

    # Save results to file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
