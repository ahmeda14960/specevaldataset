"""Pipeline for running batched generation using pre-generated inputs."""
import copy
import datetime
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..base import Specification, BatchedModel

# Assuming Organization will be adapted or a new provider mechanism for BatchedModel
# from ..base import Organization

# Set up logging
logger = logging.getLogger(__name__)


class BatchedGenerationPipeline:
    """Runs batched inference using pre-generated inputs and a BatchedModel."""

    def __init__(
        self,
        specification: Specification,
        # organization: Organization, # Adapt or replace later
        batched_model: BatchedModel,  # Directly pass BatchedModel for now
        pregenerated_inputs_dir: str,
        batch_size: int = 100,  # Default batch size
        output_base_dir: str = "data/batched_generations",
        verbose: bool = False,
        temperature: Optional[float] = 0.0,
    ):
        """
        Initialize the batched generation pipeline.

        Args:
            specification: The specification object (used to map inputs to statements).
            batched_model: An instance of a BatchedModel implementation.
            pregenerated_inputs_dir: Directory containing pre-generated inputs as JSON files (filename = statement_id).
            batch_size: The desired number of prompts per batch request.
            output_base_dir: Base directory to store batch results and metadata.
            verbose: Enable verbose logging.
            temperature: Optional temperature setting for the model. Defaults to greedy decoding.
        """
        self.specification = specification
        self.batched_model = batched_model
        self.pregenerated_inputs_dir = Path(pregenerated_inputs_dir)
        self.batch_size = batch_size
        self.output_base_dir = Path(output_base_dir)
        self.verbose = verbose
        self.temperature = temperature

        # Get model info for output directory
        model_info = self.batched_model.get_info()
        generator_model_name = model_info.get("model_name", "unknown_generator").replace("/", "-")
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create unique output directory for this run
        self.run_output_dir = self.output_base_dir / generator_model_name / run_timestamp
        self.metadata_dir = self.run_output_dir / "metadata"
        self.results_dir = self.run_output_dir / "results"

        try:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)  # Create results dir too
            if self.verbose:
                logger.info(f"Created output directories at: {self.run_output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directories: {e}")
            raise  # Re-raise the error as directory creation is critical

        self.all_prompts: List[Dict[str, Any]] = []
        self.batch_metadata_list: List[Dict[str, Any]] = []

    def _load_all_inputs(self):
        """Load all inputs from the pregenerated_inputs_dir."""
        logger.info(f"Loading inputs from {self.pregenerated_inputs_dir}...")
        self.all_prompts = []
        if not self.pregenerated_inputs_dir.is_dir():
            logger.error(f"Pregenerated inputs directory not found: {self.pregenerated_inputs_dir}")
            return

        loaded_files = 0
        loaded_prompts = 0
        for file_path in self.pregenerated_inputs_dir.glob("*.json"):
            statement_id = file_path.stem
            statement = self.specification.get_statement(statement_id)
            if not statement:
                logger.warning(
                    f"Skipping file {file_path.name}: No matching statement found for ID '{statement_id}'."
                )
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "inputs" in data and isinstance(data["inputs"], list):
                    inputs_for_statement = [str(item) for item in data["inputs"]]
                    for i, input_text in enumerate(inputs_for_statement):
                        # Store original context along with the prompt
                        self.all_prompts.append(
                            {
                                "original_index": i,  # Index within the statement file
                                "statement_id": statement_id,
                                "input_text": input_text,
                            }
                        )
                    loaded_prompts += len(inputs_for_statement)
                    loaded_files += 1
                    if self.verbose:
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

    def _submit_batches(self):
        """Submit all prompts in batches."""
        if not self.all_prompts:
            logger.warning("No prompts loaded, cannot submit batches.")
            return

        logger.info(
            f"Submitting {len(self.all_prompts)} prompts in batches of size {self.batch_size}..."
        )
        self.batch_metadata_list = []
        for i in range(0, len(self.all_prompts), self.batch_size):
            batch_prompts_data = self.all_prompts[i : i + self.batch_size]
            # Extract just the input text for the model API

            try:
                logger.info(
                    f"Submitting batch {len(self.batch_metadata_list) + 1} with ({len(batch_prompts_data)} prompts)..."
                )  # Log count from data
                # Assume generate_batch returns metadata including 'batch_id' and 'status'
                # Pass the run output directory and the full prompt data list to the model
                batch_kwargs = {}
                if self.temperature is not None:
                    batch_kwargs["temperature"] = self.temperature

                batch_metadata = self.batched_model.generate_batch(
                    prompts_data=batch_prompts_data,  # Pass the list of dicts
                    batch_output_dir=self.run_output_dir,
                    **batch_kwargs,  # Pass temperature (and potentially other future params)
                )

                # --- Store original prompt data with metadata --- statement_id, original_index, input_text
                # This is crucial for linking results back later
                batch_metadata["original_prompts_data"] = batch_prompts_data
                # Ensure batch_id is present
                if "batch_id" not in batch_metadata:
                    logger.error(f"Batch metadata missing 'batch_id': {batch_metadata}")
                    # Handle error - maybe assign a temporary ID or skip?
                    # For now, let's log and potentially skip saving this batch's metadata.
                    continue  # Skip saving metadata for this errored batch

                self.batch_metadata_list.append(batch_metadata)

                # --- Save metadata immediately ---
                metadata_file_path = self.metadata_dir / f"{batch_metadata['batch_id']}.json"
                try:
                    with open(metadata_file_path, "w", encoding="utf-8") as f:
                        json.dump(batch_metadata, f, indent=2)
                    if self.verbose:
                        logger.debug(
                            f"Saved metadata for batch {batch_metadata['batch_id']} to {metadata_file_path}"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to save metadata for batch {batch_metadata['batch_id']}: {e}"
                    )
                    # Decide how to proceed: maybe remove the batch_metadata from list?

            except Exception as e:
                logger.error(f"Failed to submit batch starting at index {i}: {e}")
                # Potentially implement retry logic here or mark these prompts as failed

        logger.info(f"Finished submitting {len(self.batch_metadata_list)} batches.")

    def _monitor_progress(self, check_interval_seconds: int = 60, timeout_minutes: int = 1440):
        """Monitor the progress of submitted batches. Default timeout is 24 hours (1440 mins)."""
        if not self.batch_metadata_list:
            logger.warning("No batches were submitted, skipping monitoring.")
            return

        logger.info("Starting progress monitoring loop...")
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        active_batches = copy.deepcopy(
            self.batch_metadata_list
        )  # Deep copy so modifications don't affect the original metadata list

        while active_batches:
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Monitoring timed out after {timeout_minutes} minutes.")
                break

            logger.info(f"Checking status of {len(active_batches)} active batches...")
            next_active_batches = []
            all_completed = True  # Assume completion until proven otherwise

            for batch_meta in active_batches:
                batch_id = batch_meta.get("batch_id")
                try:
                    # Assume check_batch_progress takes the metadata dict
                    updated_meta = self.batched_model.check_batch_progress(batch_meta)
                    status = updated_meta.get("status").lower()

                    # Update the status in our tracked list and potentially the file
                    batch_meta["status"] = status  # Update in-memory copy
                    metadata_file_path = self.metadata_dir / f"{batch_id}.json"
                    # Iterate through metadata files and update them with the latest status
                    # TODO: If we do async metadata status updates in the future we need to make this thread safe
                    if metadata_file_path.exists():
                        try:
                            # Re-read potentially updated original data, update status, write back
                            with open(metadata_file_path, "r", encoding="utf-8") as f:
                                file_data = json.load(f)
                            file_data["status"] = status  # Update status
                            # Add any other fields returned by check_batch_progress
                            for key, value in updated_meta.items():
                                if key not in [
                                    "batch_id",
                                    "original_prompts_data",
                                ]:  # Avoid overwriting key fields
                                    file_data[key] = value
                            with open(metadata_file_path, "w", encoding="utf-8") as f:
                                json.dump(file_data, f, indent=2)
                        except Exception as e:
                            logger.warning(
                                f"Could not update metadata file {metadata_file_path}: {e}"
                            )

                    print(f"  Batch {batch_id}: {status}")  # Simple print for now

                    if status in ["processing", "submitted", "pending", "running"]:
                        next_active_batches.append(batch_meta)  # Still active
                        all_completed = False
                    elif status == "completed":
                        logger.info(f"Batch {batch_id} completed.")
                        # Here you would typically trigger result retrieval
                        self._retrieve_and_save_results(batch_meta)  # Placeholder for future
                    elif status == "failed":
                        logger.error(
                            f"Batch {batch_id} failed. Reason: {updated_meta.get('error_message', 'Unknown')}"
                        )
                        # Handle failure - log, maybe retry later?
                        all_completed = False  # Count failure as not fully completed cycle
                    else:
                        logger.warning(f"Batch {batch_id} has unknown status: {status}")
                        next_active_batches.append(batch_meta)  # Keep checking unknown status
                        all_completed = False

                except Exception as e:
                    logger.error(f"Failed to check progress for batch {batch_id}: {e}")
                    next_active_batches.append(batch_meta)  # Keep trying to check it
                    all_completed = False

            active_batches = next_active_batches

            if not active_batches:
                logger.info("All batches have reached a terminal state (completed or failed).")
                break

            if not all_completed:
                logger.info(f"Waiting {check_interval_seconds} seconds before next check...")
                time.sleep(check_interval_seconds)

        logger.info("Finished progress monitoring.")

    def _retrieve_and_save_results(self, completed_batch_meta: Dict[str, Any]):
        """Retrieve and save the results for a completed batch.

        Args:
            completed_batch_meta: Metadata of the completed batch, including the batch ID
                and original prompts data.

        Raises:
            Exception: If there's an error during the retrieval or saving of results.
        """
        # Access keys directly - missing keys indicate an internal error
        batch_id = completed_batch_meta["batch_id"]  # Raise KeyError if missing
        original_prompts_data = completed_batch_meta[
            "original_prompts_data"
        ]  # Raise KeyError if missing

        if not original_prompts_data:
            logger.warning(
                f"Original prompt data list is empty for batch {batch_id}. Cannot save results."
            )
            return

        logger.info(f"Retrieving results for completed batch {batch_id}...")
        results_dict: Dict[str, Optional[str]] = {}
        try:
            results_dict = self.batched_model.get_batch_results(completed_batch_meta)
            logger.info(
                f"Successfully retrieved results dict for batch {batch_id}. Processing {len(results_dict)} entries."
            )
        except Exception as e:
            logger.error(f"Failed to retrieve results for batch {batch_id}: {e}")
            # Even if retrieval fails, we might want to mark expected prompts as failed?
            # For now, just return and don't save anything for this batch.
            return

        # Group results by statement ID
        statement_results: Dict[str, List[Dict[str, Any]]] = {}
        missing_results_count = 0
        for prompt_data in original_prompts_data:
            statement_id = prompt_data.get("statement_id")
            original_index = prompt_data.get("original_index")
            input_text = prompt_data.get("input_text")

            if statement_id is None or original_index is None:
                logger.warning(
                    f"Skipping prompt data in batch {batch_id} due to missing fields: {prompt_data}"
                )
                continue

            custom_id = f"{statement_id}_{original_index}"
            result_text = results_dict.get(custom_id)

            if result_text is None:
                logger.warning(
                    f"Result for custom_id '{custom_id}' not found in results dict for batch {batch_id}."
                )
                result_text = "ERROR: Result missing in batch output."
                missing_results_count += 1

            result_record = {
                "original_index": original_index,
                "input_text": input_text,
                "output_text": result_text,
                "batch_id": batch_id,  # Add batch_id for traceability
            }

            if statement_id not in statement_results:
                statement_results[statement_id] = []
            statement_results[statement_id].append(result_record)

        if missing_results_count > 0:
            logger.warning(f"{missing_results_count} results were missing for batch {batch_id}.")

        # Save grouped results to files, merging with existing data
        logger.info(f"Saving processed results from batch {batch_id} to {self.results_dir}...")
        saved_count = 0
        failed_save_count = 0
        for statement_id, new_results_list in statement_results.items():
            output_path = self.results_dir / f"{statement_id}.json"
            all_statement_results: List[
                Dict[str, Any]
            ] = []  # Holds results from file + new results

            # 1. Read existing data if file exists
            if output_path.exists():
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                        # Basic check if it's a list of dicts (our expected format)
                        if isinstance(existing_results, list) and all(
                            isinstance(item, dict) for item in existing_results
                        ):
                            all_statement_results = existing_results
                            if self.verbose:
                                logger.debug(
                                    f"Read {len(existing_results)} existing results from {output_path}"
                                )
                        else:
                            logger.warning(
                                f"Existing results file {output_path} is not a list of dicts. Overwriting."
                            )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode existing JSON from {output_path}. Overwriting."
                    )
                except Exception as e:
                    logger.warning(
                        f"Error reading existing results file {output_path}: {e}. Overwriting."
                    )

            # 2. Merge new results, using original_index as key to handle overlaps/updates
            # Store temporarily in a dict keyed by original_index for efficient merging/overwriting
            merged_results_dict = {res["original_index"]: res for res in all_statement_results}
            update_count = 0
            new_count = 0
            for new_res in new_results_list:
                if new_res["original_index"] in merged_results_dict:
                    update_count += 1
                else:
                    new_count += 1
                merged_results_dict[
                    new_res["original_index"]
                ] = new_res  # Overwrite if duplicate index exists

            # 3. Convert back to list and sort
            final_results_list = sorted(
                merged_results_dict.values(), key=lambda x: x.get("original_index", -1)
            )

            # 4. Write back to file
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure results dir exists
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(final_results_list, f, indent=2)
                if self.verbose:
                    logger.debug(
                        f"Saved {len(final_results_list)} results ({new_count} new, {update_count} updated) for statement {statement_id} to {output_path}"
                    )
                saved_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to write results for statement {statement_id} to {output_path}: {e}"
                )
                failed_save_count += 1

        logger.info(
            f"Finished saving results for batch {batch_id}. Saved/updated files for {saved_count} statements. Failed saves for {failed_save_count} statements."
        )

    def run(self):
        """Execute the batched generation pipeline."""
        logger.info("Starting Batched Generation Pipeline...")

        # 1. Load all inputs
        self._load_all_inputs()
        if not self.all_prompts:
            logger.error("Failed to load any prompts. Aborting pipeline.")
            return

        # 2. Submit inputs in batches
        self._submit_batches()
        if not self.batch_metadata_list:
            logger.error("Failed to submit any batches. Aborting pipeline.")
            return

        # 3. Monitor progress (simple loop for now)
        self._monitor_progress()

        # 4. (Future) Consolidate results if needed

        logger.info("Batched Generation Pipeline finished.")
