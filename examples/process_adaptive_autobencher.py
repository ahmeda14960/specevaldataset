#!/usr/bin/env python3
"""
Script to process adaptive autobencher outputs and filter questions by rating.
This script will:
1. Read all JSON files from adaptive_autobencher/{judge_mode}/{evaluator_model}x{candidate_model}/
2. Extract unique questions and ratings from the 'trajectory' field.
3. Filter questions based on rating threshold.
4. If --num-questions is set, adjust selection to meet the target count,
   prioritizing lowest ratings first, then adding higher ratings if needed.
5. Output unique filtered/selected questions to
   data/adaptive_autobencher_outputs_from_trajectory/{judge_mode}/{evaluator_model}/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
import datetime
from tqdm import tqdm
import random
from collections import Counter, defaultdict
import glob

from speceval.utils.logging import setup_logging


def process_adaptive_autobencher_files(
    input_dir: Path,
    output_dir: Path,
    rating_threshold: int,
    evaluator_model: str,
    logger: logging.Logger,
    num_questions: Optional[int] = None,
    uniform: bool = False,
) -> Dict[str, List[str]]:
    """
    Process adaptive autobencher outputs (trajectory), filter by rating,
    and ensure a minimum number of questions per statement if specified.

    Args:
        input_dir: Directory or glob pattern containing adaptive autobencher outputs
        output_dir: Directory to write filtered outputs
        rating_threshold: Maximum rating to include initially (1-5)
        evaluator_model: Name of the evaluator model
        logger: Logger instance for detailed logging
        num_questions: Target minimum number of questions per statement
        uniform: If True, select questions to ensure uniform distribution across rating levels

    Returns:
        Dictionary mapping statement names to lists of filtered/selected questions
    """
    filtered_outputs: Dict[str, List[str]] = {}

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of JSON files first, handling glob patterns
    input_path_str = str(input_dir)
    logger.info(f"Input path string: {input_path_str}")

    if "*" in input_path_str:
        # If input_dir contains a glob pattern, use it directly with the raw string
        logger.info(f"Using glob pattern: {input_path_str}")
        json_files = [Path(p) for p in glob.glob(input_path_str, recursive=True)]
        logger.info(f"Files found by glob: {json_files}")
    else:
        # Otherwise, search recursively for all JSON files
        logger.info(f"Using recursive search in: {input_path_str}")
        json_files = list(Path(input_path_str).rglob("*.json"))
        logger.info(f"Files found by rglob: {json_files}")

    # Filter for only JSON files if using glob pattern
    json_files = [f for f in json_files if f.suffix == ".json"]

    logger.info(f"Found {len(json_files)} JSON files to process.")
    if len(json_files) == 0:
        logger.warning(f"No JSON files found in {input_path_str}")

    if num_questions:
        logger.info(f"Target number of questions per statement: {num_questions}")

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            logger.info(f"Processing file: {json_file.name}")
            with open(json_file) as f:
                data = json.load(f)

            statement_name = json_file.stem
            logger.info(f"Statement: {statement_name}")
            logger.info(f"Input data keys: {list(data.keys())}")

            # --- Modification Start: Collect all questions with ratings ---
            all_questions_data: List[Dict[str, Any]] = []
            processed_questions_text: Set[str] = set()

            trajectory = data.get("trajectory", [])
            if not trajectory:
                assert trajectory, f"Trajectory field not found or empty in {json_file.name}"

            logger.info(f"Processing trajectory with {len(trajectory)} rounds...")
            total_evals_in_trajectory = 0

            for round_num, round_data in enumerate(trajectory):
                evaluations = round_data.get("evaluations", [])
                logger.debug(f"Round {round_num+1}: Found {len(evaluations)} evaluations.")

                for evaluation_item in evaluations:
                    total_evals_in_trajectory += 1
                    question = evaluation_item.get("question")
                    rating = evaluation_item.get("rating")

                    if question is None or rating is None or not isinstance(rating, (int, float)):
                        logger.warning(
                            f"Skipping evaluation item due to missing/invalid data: {evaluation_item}"
                        )
                        continue

                    rating = int(round(rating))
                    if 1 <= rating <= 5 and question not in processed_questions_text:
                        all_questions_data.append({"question": question, "rating": rating})
                        processed_questions_text.add(question)

            logger.info(
                f"Processed {total_evals_in_trajectory} evaluations, found {len(all_questions_data)} unique valid questions across all ratings."
            )

            if not all_questions_data:
                logger.warning(f"No valid questions found in trajectory for {statement_name}.")
                continue

            # Group questions by rating
            questions_by_rating = defaultdict(list)
            for item in all_questions_data:
                questions_by_rating[item["rating"]].append(item["question"])

            # Log initial counts per rating
            rating_counts = {r: len(q) for r, q in questions_by_rating.items()}
            logger.info(f"Counts per rating found: {sorted(rating_counts.items())}")

            # --- Initial Filtering based on rating_threshold ---
            initially_filtered_questions: List[Dict[str, Any]] = []
            for r in range(1, rating_threshold + 1):
                if r in questions_by_rating:
                    for q_text in questions_by_rating[r]:
                        initially_filtered_questions.append({"question": q_text, "rating": r})

            logger.info(
                f"Initially filtered {len(initially_filtered_questions)} questions with rating <= {rating_threshold}."
            )

            # --- Apply num_questions logic ---
            final_selected_questions: List[Dict[str, Any]] = []

            if num_questions is not None and num_questions > 0:
                if uniform:
                    # Uniform distribution logic
                    logger.info(
                        f"Using uniform distribution mode to select {num_questions} questions."
                    )

                    # Get all available ratings in the data
                    available_ratings = sorted(list(questions_by_rating.keys()))
                    if not available_ratings:
                        logger.warning(
                            f"No questions with valid ratings found for {statement_name}."
                        )
                        continue

                    # Calculate target questions per rating level
                    num_ratings = len(available_ratings)
                    target_per_rating = num_questions // num_ratings
                    extra = num_questions % num_ratings

                    logger.info(f"Available ratings: {available_ratings}")
                    logger.info(f"Target per rating: {target_per_rating}, with {extra} extra")

                    # Select questions from each rating level
                    for i, rating in enumerate(available_ratings):
                        # Allocate the "extra" questions to the lowest ratings first
                        current_target = target_per_rating + (1 if i < extra else 0)
                        available = questions_by_rating[rating]

                        # Shuffle to randomize selection
                        random.shuffle(available)

                        # Take up to the target number, or all if fewer are available
                        selected_count = min(current_target, len(available))
                        for j in range(selected_count):
                            final_selected_questions.append(
                                {"question": available[j], "rating": rating}
                            )

                        logger.info(
                            f"Selected {selected_count}/{current_target} questions with rating {rating}"
                        )

                    # If we still don't have enough questions, fill in from any rating level
                    if len(final_selected_questions) < num_questions:
                        logger.warning(
                            f"Could not get {num_questions} questions with uniform distribution. Selected {len(final_selected_questions)}."
                        )

                        # Make a list of all remaining questions
                        remaining_questions = []
                        for rating in available_ratings:
                            # Count how many we've already selected from this rating
                            selected_from_rating = sum(
                                1 for q in final_selected_questions if q["rating"] == rating
                            )
                            # Add remaining questions from this rating that weren't selected
                            remaining = questions_by_rating[rating][selected_from_rating:]
                            for q in remaining:
                                remaining_questions.append({"question": q, "rating": rating})

                        # Shuffle and select additional questions if available
                        random.shuffle(remaining_questions)
                        needed = num_questions - len(final_selected_questions)
                        final_selected_questions.extend(remaining_questions[:needed])

                        logger.info(
                            f"Added {min(needed, len(remaining_questions))} additional questions to reach target count."
                        )
                else:
                    # Original filtering logic
                    count_filtered = len(initially_filtered_questions)

                    if count_filtered < num_questions:
                        # Case 1: Need more questions, add higher ratings
                        logger.info(
                            f"Count ({count_filtered}) is less than target ({num_questions}). Adding questions from higher ratings."
                        )
                        final_selected_questions.extend(initially_filtered_questions)
                        needed = num_questions - count_filtered
                        ratings_added_log = Counter()

                        for rating_to_add in range(rating_threshold + 1, 6):
                            if rating_to_add in questions_by_rating:
                                available_higher = questions_by_rating[rating_to_add]
                                random.shuffle(available_higher)
                                can_add_count = min(needed, len(available_higher))

                                if can_add_count > 0:
                                    for i in range(can_add_count):
                                        q_text = available_higher[i]
                                        final_selected_questions.append(
                                            {"question": q_text, "rating": rating_to_add}
                                        )
                                        ratings_added_log[rating_to_add] += 1
                                    needed -= can_add_count
                                    logger.info(
                                        f"Added {can_add_count} questions with rating {rating_to_add}."
                                    )

                            if needed == 0:
                                break

                        if needed > 0:
                            logger.warning(
                                f"Could not reach target {num_questions}. Only found {len(final_selected_questions)} questions in total."
                            )

                    else:
                        # Case 2: Have enough or too many, select lowest ratings first
                        logger.info(
                            f"Count ({count_filtered}) meets or exceeds target ({num_questions}). Selecting {num_questions} questions with the lowest ratings."
                        )
                        initially_filtered_questions.sort(key=lambda x: x["rating"])
                        final_selected_questions = initially_filtered_questions[:num_questions]

            else:
                # No num_questions target, use all initially filtered
                logger.info(
                    f"No target number specified, using all {len(initially_filtered_questions)} questions with rating <= {rating_threshold}."
                )
                final_selected_questions = initially_filtered_questions

            # --- End Apply num_questions logic ---

            if final_selected_questions:
                # Log final composition
                final_composition = Counter(item["rating"] for item in final_selected_questions)
                logger.info(
                    f"Final selection: {len(final_selected_questions)} questions. Composition: {sorted(final_composition.items())}"
                )

                # Extract question strings for output
                final_question_list = [item["question"] for item in final_selected_questions]
                filtered_outputs[statement_name] = final_question_list

                # Write filtered questions to output file
                output_file = output_dir / f"{statement_name}.json"
                output_data = {
                    "metadata": {
                        "generator_type": "AdaptiveAutobencherTrajectoryProcessor",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "evaluator_model": evaluator_model,
                        "rating_threshold": rating_threshold,
                        "num_questions_target": num_questions,
                        "uniform_distribution": uniform,
                        "final_question_count": len(final_question_list),
                        "final_rating_composition": dict(sorted(final_composition.items())),
                        "source_file": str(json_file.name),
                    },
                    "inputs": final_question_list,
                }

                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Wrote final questions to: {output_file}")
            else:
                logger.info("No questions selected after applying filters and target number.")

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}", exc_info=True)

    return filtered_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Process adaptive autobencher outputs (trajectory), filter by rating, and select a target number of questions."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory or glob pattern containing adaptive autobencher outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/adaptive_autobencher_outputs",
        help="Base directory for output files (default: data/adaptive_autobencher_outputs_from_trajectory)",
    )
    parser.add_argument(
        "--rating-threshold",
        type=int,
        choices=range(1, 6),
        default=5,
        help="Include questions with rating <= threshold (1-5) (default: 5)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Target minimum number of questions per statement. If set, selection logic applies. (default: None)",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        required=True,
        help="Name of the evaluator model (used for organizing output)",
    )
    parser.add_argument(
        "--candidate-model",
        type=str,
        required=True,
        help="Name of the candidate model (used for organizing output)",
    )
    parser.add_argument(
        "--judge-mode",
        type=str,
        required=True,
        choices=["response_judge", "input_judge"],
        help="The judge mode used in generating the inputs (response_judge or input_judge)",
    )
    parser.add_argument(
        "--spec",
        type=str,
        required=True,
        choices=["openai", "google", "anthropic"],
        help="Name of the model specification or organization (openai, google, anthropic)",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="If set and num-questions is specified, selects questions with uniform distribution across rating levels",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose, folder_name="process_adaptive_autobencher_trajectory")

    # Keep the raw input string for glob patterns
    input_dir_str = args.input_dir

    # Handle directory existence check differently for wildcard and non-wildcard paths
    if "*" in input_dir_str:
        # For wildcard paths, check if the parent directory up to the wildcard exists
        parent_path = str(Path(input_dir_str).parent)
        parent_path = parent_path.split("*")[0]  # Get the part before the first wildcard
        logger.info(f"Checking parent path: {parent_path}")
        if not Path(parent_path).exists():
            logger.error(f"Parent directory not found: {parent_path}")
            return
    else:
        # For non-wildcard paths, check if the directory itself exists
        input_dir = Path(input_dir_str)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return

    # Pass the raw input string as a Path object
    input_dir = Path(input_dir_str)

    # Verify the input directory structure matches the judge_mode
    if args.judge_mode not in input_dir_str:
        logger.error(
            f"Input directory structure does not match the specified judge_mode. Expected path containing '{args.judge_mode}'"
        )
        return

    output_base = Path(args.output_dir)
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Create model combination string
    model_combo = f"{args.evaluator_model}x{args.candidate_model}"

    # Update output directory structure to include spec, judge_mode, model combo, and timestamp
    output_dir = output_base / args.spec / args.judge_mode / model_combo / current_date

    logger.info(f"Processing trajectory data from: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Initial rating threshold: {args.rating_threshold}")
    logger.info(f"Judge mode: {args.judge_mode}")
    logger.info(f"Model combination: {model_combo}")
    if args.uniform:
        logger.info("Using uniform distribution mode for question selection")
    if args.num_questions:
        logger.info(f"Target number of questions: {args.num_questions}")

    try:
        filtered_outputs = process_adaptive_autobencher_files(
            input_dir=input_dir,
            output_dir=output_dir,
            rating_threshold=args.rating_threshold,
            evaluator_model=args.evaluator_model,
            logger=logger,
            num_questions=args.num_questions,
            uniform=args.uniform,
        )

        # Print summary
        total_statements = len(filtered_outputs)
        total_questions = sum(len(questions) for questions in filtered_outputs.values())

        logger.info("\nProcessing Summary:")
        logger.info(f"Total statements processed with output questions: {total_statements}")
        logger.info(f"Total final questions generated across all statements: {total_questions}")

        # Print breakdown by statement
        if filtered_outputs:
            logger.info("\nQuestions per statement (final count):")
            for statement, questions in filtered_outputs.items():
                logger.info(f"  {statement}: {len(questions)} questions")

        logger.info(f"\nFiltered outputs written to: {output_dir}")

    except Exception as e:
        logger.critical(f"An critical error occurred during processing: {e}", exc_info=True)


if __name__ == "__main__":
    main()
