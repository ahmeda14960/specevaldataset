"""Script for running batch inference using Together AI's async client."""
import asyncio
import os
import json
import logging
import time
import argparse
import sys  # For checking if tqdm is installed
from pathlib import Path
import datetime
from typing import List, Dict, Any, Optional
import random  # Added for jitter in retry delays

# Try importing tqdm, prompt for installation if missing
try:
    from tqdm import tqdm

    # No longer importing tqdm_asyncio
except ImportError:
    print("Error: 'tqdm' library not found. Please install it: pip install tqdm")
    sys.exit(1)

try:
    from together import AsyncTogether
    from together.types import ChatCompletionResponse
except ImportError:
    print("Error: 'together' library not found. Please install it: pip install together")
    sys.exit(1)

# Import the framework's logging setup
from speceval.utils.logging import setup_logging

# Import model constants - these are used for reference and documentation
# flake8: noqa: F401
from speceval.models.together import (
    DEEPSEEK_V3,
    QWEN_2_72B_INSTRUCT,
    LLAMA_3_1_405B_TURBO,
    QWEN_235B_FP8,
    QWEN_2_5_72B_TURBO,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_retriable_error(result_value: Optional[str], retriable_substrings: List[str]) -> bool:
    """Checks if the result_value (an error string) is a retriable error."""
    if not isinstance(result_value, str) or not result_value.startswith("ERROR:"):
        return False
    for substring in retriable_substrings:
        if substring.lower() in result_value.lower():  # Case-insensitive check
            return True
    return False


def load_inputs_from_dir(
    pregenerated_inputs_dir: Path, specification: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Load inputs from JSON files in the specified directory.

    Each JSON file should have a name matching a statement_id and contain
    a list under the key "inputs".

    Args:
        pregenerated_inputs_dir: Path to the directory containing input JSON files.
        specification: Optional Specification object (currently unused, for potential future use
                       if statement details are needed beyond the ID).

    Returns:
        A list of dictionaries, each containing:
        {'custom_id': str, 'input_text': str, 'statement_id': str, 'original_index': int}
    """
    logger.info(f"Loading inputs from {pregenerated_inputs_dir}...")
    all_prompts_data = []
    if not pregenerated_inputs_dir.is_dir():
        logger.error(f"Input directory not found: {pregenerated_inputs_dir}")
        return []

    loaded_files = 0
    loaded_prompts = 0
    for file_path in pregenerated_inputs_dir.glob("*.json"):
        statement_id = file_path.stem
        # Optional: Validate statement_id against specification if provided
        # statement = specification.get_statement(statement_id)
        # if not statement:
        #     logger.warning(f"Skipping file {file_path.name}: No matching statement found for ID '{statement_id}'.")
        #     continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "inputs" in data and isinstance(data["inputs"], list):
                inputs_for_statement = [str(item) for item in data["inputs"]]
                for i, input_text in enumerate(inputs_for_statement):
                    custom_id = f"{statement_id}_{i}"
                    all_prompts_data.append(
                        {
                            "custom_id": custom_id,
                            "input_text": input_text,
                            "statement_id": statement_id,
                            "original_index": i,
                        }
                    )
                loaded_prompts += len(inputs_for_statement)
                loaded_files += 1
                logger.debug(
                    f"Loaded {len(inputs_for_statement)} inputs for statement '{statement_id}' from {file_path.name}"
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
        f"Finished loading inputs. Found {loaded_prompts} prompts across {loaded_files} statement files."
    )
    return all_prompts_data


async def run_together_batch(
    async_client: AsyncTogether,
    model_name: str,
    prompts_batch: List[Dict[str, Any]],
    qps_limit: int,
) -> Dict[str, Optional[str]]:
    """
    Run a batch of prompts concurrently using Together AI's async client,
    respecting rate limits.

    Args:
        async_client: An initialized AsyncTogether client.
        model_name: The name of the Together model to use.
        prompts_batch: A list of prompt data dictionaries, each containing at least
                       'custom_id' and 'input_text'.
        qps_limit: Maximum requests per second allowed (used for semaphore).

    Returns:
        A dictionary mapping custom_id to the generated text or an error string.
    """
    # Special handling for extremely low rate limits like Deepseek's free/low tiers
    # Using 3 RPM = 0.05 QPS as an example threshold
    # TODO: Get actual Deepseek tier limits programmatically if possible or make configurable
    # Refined check: Apply specific low limits only if model name suggests DeepSeek R1
    # Note: This is heuristic. Adjust if model naming changes or more precise check is needed.
    is_deepseek_r1_low_tier = (
        "deepseek" in model_name.lower() and "r1" in model_name.lower()
    )  # Example: Check for both keywords
    deepseek_sleep_time = 20  # Corresponds to 3 RPM

    if is_deepseek_r1_low_tier:
        logger.warning(
            f"Detected low-rate model ('{model_name}'). Switching to sequential processing with {deepseek_sleep_time}s delay between requests."
        )
        results_list = []
        # Process sequentially without tqdm wrapper
        for prompt_data in tqdm(prompts_batch, desc="Processing sequentially", leave=False):
            start_req_time = time.time()
            try:
                response = await async_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_data["input_text"]}],
                    max_tokens=4096,
                    chat_template_kwargs={**chat_template_kwargs},
                    temperature=0.0,
                )
                results_list.append(response)
            except Exception as e:
                results_list.append(e)  # Store exception to process later
            finally:
                # Sleep AFTER the request completes/fails
                await asyncio.sleep(deepseek_sleep_time)
                end_req_time = time.time()
                logger.debug(
                    f"Processed request for {prompt_data['custom_id']} in {end_req_time - start_req_time:.2f}s (including sleep)"
                )

        results = results_list  # Use the collected results/exceptions

    else:
        # Use Semaphore for standard rate limiting
        semaphore = asyncio.Semaphore(qps_limit)
        tasks = []

        if model_name == QWEN_235B_FP8:
            chat_template_kwargs = {"enable_thinking": False}
        else:
            chat_template_kwargs = {}

        async def _get_completion(prompt_data: Dict[str, Any]):
            async with semaphore:
                # Small delay *before* acquiring semaphore slot can sometimes help smooth bursts
                await asyncio.sleep(0.01)
                return await async_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_data["input_text"]}],
                    max_tokens=4096,
                    temperature=0.0,
                    chat_template_kwargs={**chat_template_kwargs},
                )

        for prompt_data in prompts_batch:
            tasks.append(_get_completion(prompt_data))

        start_time = time.time()
        logger.info(
            f"Submitting {len(tasks)} requests (QPS Limit: {qps_limit}) for model {model_name}..."
        )

        # Use return_exceptions=True to capture errors without stopping the whole batch
        # Use standard asyncio.gather
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        logger.info(
            f"Received {len(results)} responses/exceptions in {end_time - start_time:.2f} seconds."
        )

    # --- Process results (common for both methods) ---
    processed_results: Dict[str, Optional[str]] = {}
    for i, result in enumerate(results):
        custom_id = prompts_batch[i]["custom_id"]
        if isinstance(result, Exception):
            error_message = f"ERROR: {type(result).__name__}: {str(result)}"
            logger.warning(f"Request for {custom_id} failed: {error_message}")
            processed_results[custom_id] = error_message
        elif isinstance(result, ChatCompletionResponse):
            try:
                # Add basic rate limit handling for specific models if needed here
                # Although asyncio handles concurrency, explicit sleeps might still be needed
                # if 'deepseek' in model_name.lower():
                #      await asyncio.sleep(1) # Example: short sleep after each deepseek success
                processed_results[custom_id] = result.choices[0].message.content.strip()
            except (AttributeError, IndexError, TypeError) as e:
                error_message = f"ERROR: Could not parse successful response structure: {e}"
                logger.warning(f"Failed to parse response for {custom_id}: {error_message}")
                processed_results[custom_id] = error_message
        else:
            # Should not happen with return_exceptions=True
            error_message = f"ERROR: Unexpected result type: {type(result).__name__}"
            logger.warning(f"Unexpected result for {custom_id}: {error_message}")
            processed_results[custom_id] = error_message

    return processed_results


def main():
    """Run the main batch inference pipeline."""
    parser = argparse.ArgumentParser(description="Run batch inference using Together AI async API.")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name of the Together AI model (e.g., 'Qwen/Qwen2-72B-Instruct').",
    )
    parser.add_argument(
        "--pregenerated-inputs-dir",
        required=True,
        type=Path,
        help="Directory containing pre-generated inputs as JSON files (filename = statement_id).",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default="data/together_batch_generations",
        help="Base directory to store results and metadata.",
    )
    parser.add_argument(
        "--spec",
        type=str,
        required=True,
        choices=["openai", "google", "anthropic"],
        help="Name of the model specification or organization (openai, google, anthropic)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of concurrent requests to send in each asyncio batch.",
    )
    parser.add_argument(
        "--qps",
        type=int,
        default=10,
        help="Maximum requests per second to send (controls concurrency). Adjust based on your rate limit tier.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for a failed sub-batch (default: 3).",
    )
    parser.add_argument(
        "--initial-retry-delay",
        type=float,
        default=10.0,
        help="Initial delay in seconds before the first retry attempt (default: 10.0).",
    )
    parser.add_argument(
        "--retry-backoff-factor",
        type=float,
        default=2.0,
        help="Multiplier for exponential backoff of retry delay (default: 2.0).",
    )
    parser.add_argument(
        "--retriable-error-substrings",
        type=lambda s: [item.strip() for item in s.split(",")],
        default="ServiceUnavailableError,503,500,overloaded,not ready yet,upstream connect error",
        help="Comma-separated list of substrings in error messages that should trigger a retry (case-insensitive). Default: 'ServiceUnavailableError,503,500,overloaded,not ready yet,upstream connect error'.",
    )

    args = parser.parse_args()

    # Configure logging using the utility function
    # Note: The logger instance is retrieved within functions using logging.getLogger(__name__)
    setup_logging(args.verbose, folder_name="together_batch_inference")
    logger = logging.getLogger(__name__)  # Get the logger for main scope

    # --- Setup API Client ---
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        logger.error("No Together AI API key provided. Use --api-key or set TOGETHER_API_KEY.")
        exit(1)
    async_client = AsyncTogether(api_key=api_key)

    # --- Setup Output Directory ---
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_name = args.model_name.replace("/", "-")
    run_output_dir = args.output_base_dir / args.spec / model_dir_name / run_timestamp
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {run_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory: {e}")
        exit(1)

    # --- Load Specification (Optional) ---
    # spec = None
    # if args.spec_path:
    #     try:
    #         # Assuming a load function exists for Specification
    #         # spec = load_specification(args.spec_path)
    #         logger.info(f"Loaded specification from {args.spec_path}")
    #     except Exception as e:
    #         logger.warning(f"Could not load specification: {e}")

    # --- Load Inputs ---
    all_prompts_data = load_inputs_from_dir(args.pregenerated_inputs_dir)  # , spec)
    if not all_prompts_data:
        logger.error("No prompts loaded. Exiting.")
        exit(1)

    total_prompts = len(all_prompts_data)
    logger.info(f"Loaded {total_prompts} prompts. Processing in batches of {args.batch_size}...")

    # --- Run Inference in Batches ---
    all_results: Dict[str, Optional[str]] = {}
    num_batches = (total_prompts + args.batch_size - 1) // args.batch_size

    # Iterate through batches without tqdm wrapper
    for i in tqdm(range(0, total_prompts, args.batch_size), desc="Processing Batches"):
        batch_num = (i // args.batch_size) + 1
        logger.info(f"--- Processing batch {batch_num}/{num_batches} --- ")
        batch_data = all_prompts_data[i : i + args.batch_size]

        # Initial attempt for the current batch
        initial_batch_run_results_dict = asyncio.run(
            run_together_batch(async_client, args.model_name, batch_data, args.qps)
        )

        # --- Retry Logic for Failed Prompts in this Batch ---
        current_batch_final_results: Dict[str, Optional[str]] = {}
        prompts_initially_failed_and_retriable: List[Dict[str, Any]] = []

        for prompt_detail in batch_data:
            custom_id = prompt_detail["custom_id"]
            result_or_error = initial_batch_run_results_dict.get(custom_id)

            if is_retriable_error(result_or_error, args.retriable_error_substrings):
                logger.warning(
                    f"Prompt {custom_id} from batch {batch_num} failed with retriable error: {result_or_error}. Queuing for retry."
                )
                prompts_initially_failed_and_retriable.append(prompt_detail)
            else:
                if isinstance(result_or_error, str) and result_or_error.startswith("ERROR:"):
                    logger.error(
                        f"Prompt {custom_id} from batch {batch_num} failed with NON-retriable error: {result_or_error}."
                    )
                current_batch_final_results[custom_id] = result_or_error

        if prompts_initially_failed_and_retriable:
            prompts_queued_for_next_retry_attempt = list(prompts_initially_failed_and_retriable)

            for retry_attempt_num in range(args.max_retries):
                if not prompts_queued_for_next_retry_attempt:
                    logger.info(
                        f"All pending retries for batch {batch_num} succeeded before exhausting max retries ({args.max_retries} attempts)."
                    )
                    break

                prompts_for_this_api_call = list(prompts_queued_for_next_retry_attempt)
                prompts_queued_for_next_retry_attempt.clear()

                logger.info(
                    f"--- Batch {batch_num}/{num_batches}, Retry Attempt {retry_attempt_num + 1}/{args.max_retries} for {len(prompts_for_this_api_call)} prompts ---"
                )

                delay = args.initial_retry_delay * (args.retry_backoff_factor**retry_attempt_num)
                jitter = random.uniform(-0.1 * delay, 0.1 * delay)  # +/- 10% jitter
                actual_delay = max(1.0, delay + jitter)  # Ensure delay is at least 1s

                logger.info(f"Waiting for {actual_delay:.2f} seconds before this retry attempt...")
                time.sleep(actual_delay)  # Synchronous sleep as this is between asyncio runs

                retry_run_results_dict = asyncio.run(
                    run_together_batch(
                        async_client, args.model_name, prompts_for_this_api_call, args.qps
                    )
                )

                for current_prompt_data in prompts_for_this_api_call:
                    custom_id = current_prompt_data["custom_id"]
                    result_from_this_retry = retry_run_results_dict.get(custom_id)

                    error_is_still_retriable_type = is_retriable_error(
                        result_from_this_retry, args.retriable_error_substrings
                    )

                    if (
                        not error_is_still_retriable_type
                        and isinstance(result_from_this_retry, str)
                        and result_from_this_retry.startswith("ERROR:")
                    ):
                        logger.error(
                            f"Prompt {custom_id} (batch {batch_num}) failed on retry {retry_attempt_num + 1} with a NON-retriable error: {result_from_this_retry}"
                        )
                        current_batch_final_results[custom_id] = result_from_this_retry
                    elif error_is_still_retriable_type:
                        if retry_attempt_num < args.max_retries - 1:
                            logger.warning(
                                f"Prompt {custom_id} (batch {batch_num}) failed on retry {retry_attempt_num + 1} with retriable error: {result_from_this_retry}. Queuing for next attempt."
                            )
                            prompts_queued_for_next_retry_attempt.append(current_prompt_data)
                        else:
                            logger.error(
                                f"Prompt {custom_id} (batch {batch_num}) failed on final retry attempt {retry_attempt_num + 1}/{args.max_retries} with error: {result_from_this_retry}"
                            )
                            current_batch_final_results[custom_id] = result_from_this_retry
                    else:  # Succeeded on this retry
                        logger.info(
                            f"Prompt {custom_id} (batch {batch_num}) succeeded on retry attempt {retry_attempt_num + 1}."
                        )
                        current_batch_final_results[custom_id] = result_from_this_retry

                if not prompts_queued_for_next_retry_attempt:
                    logger.info(
                        f"No more prompts queued for retry after attempt {retry_attempt_num + 1} for batch {batch_num}."
                    )
                    break  # Exit retry loop for this batch if all are resolved

            # After all retry attempts, if any prompts are still in prompts_queued_for_next_retry_attempt,
            # it means they failed the last attempt and were retriable, but exhausted retries.
            # Their final error should already be in current_batch_final_results.
            if prompts_queued_for_next_retry_attempt:
                logger.error(
                    f"{len(prompts_queued_for_next_retry_attempt)} prompts in batch {batch_num} remained in retry queue after all {args.max_retries} attempts. Their final errors should have been logged and stored."
                )
                for p_data in prompts_queued_for_next_retry_attempt:  # Fallback logging
                    cid = p_data["custom_id"]
                    if (
                        cid not in current_batch_final_results
                    ):  # Should not happen if logic above is correct
                        logger.critical(
                            f"CRITICAL: Prompt {cid} from batch {batch_num} was in final retry queue but its error was not captured in final results. Marking with generic error."
                        )
                        current_batch_final_results[
                            cid
                        ] = f"ERROR: Prompt {cid} failed after max retries, specific final error not captured."

        # Update the global results with the outcomes from this batch (including all retries)
        all_results.update(current_batch_final_results)
        logger.info(
            f"--- Finished processing and all retries for batch {batch_num}/{num_batches} --- "
        )

    logger.info(f"Finished processing all {num_batches} batches.")

    # --- Consolidate and Save Results ---
    final_output = {
        "metadata": {
            "model_name": args.model_name,
            "input_directory": str(args.pregenerated_inputs_dir),
            "output_directory": str(run_output_dir),
            "batch_size": args.batch_size,
            "total_prompts": total_prompts,
            "run_timestamp": run_timestamp,
            "completed_timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        "results": [],  # Store as list of dicts for easier parsing
    }

    # Map results back to original structure if needed, or save the flat dict
    # Here, we store as a list matching the input structure conceptually
    missing_results = 0
    successful_results = 0
    failed_results = 0
    for prompt_data in all_prompts_data:
        custom_id = prompt_data["custom_id"]
        result = all_results.get(custom_id)
        if result is None:
            logger.warning(f"Result missing for custom_id: {custom_id}")
            final_output["results"].append(
                {
                    "statement_id": prompt_data["statement_id"],
                    "original_index": prompt_data["original_index"],
                    "input_text": prompt_data["input_text"],
                    "output_text": "ERROR: Result missing",
                }
            )
            missing_results += 1
        elif isinstance(result, str) and result.startswith("ERROR:"):
            final_output["results"].append(
                {
                    "statement_id": prompt_data["statement_id"],
                    "original_index": prompt_data["original_index"],
                    "input_text": prompt_data["input_text"],
                    "output_text": result,  # Keep error message
                }
            )
            failed_results += 1
        else:
            final_output["results"].append(
                {
                    "statement_id": prompt_data["statement_id"],
                    "original_index": prompt_data["original_index"],
                    "input_text": prompt_data["input_text"],
                    "output_text": result,
                }
            )
            successful_results += 1

    results_file = run_output_dir / "results.json"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Successfully saved results ({successful_results} success, {failed_results} failed, {missing_results} missing) to {results_file}"
        )
    except IOError as e:
        logger.error(f"Failed to save results file: {e}")

    logger.info("Together AI batch inference script finished.")


if __name__ == "__main__":
    main()
