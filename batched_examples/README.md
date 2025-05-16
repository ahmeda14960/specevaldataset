# SpecEval Batched API Examples and Design

This directory contains examples and documentation related to using batched inference within the SpecEval framework.

## Design Decisions & Considerations (OpenAI Batch API Integration)

When integrating the OpenAI Batch API, several design choices were made to balance the specifics of OpenAI's implementation with the goal of maintaining a consistent `BatchedModel` interface within SpecEval.

1.  **Abstraction Level (Approach 1 Chosen):** We opted for "Full Abstraction". The `OpenAIBatchedModel` class encapsulates all interactions with the OpenAI Batch API, including:
    *   Creating the required `.jsonl` input file locally.
    *   Uploading the input file via the OpenAI Files API.
    *   Creating the batch job via the OpenAI Batches API.
    *   Checking the batch job status.
    *   Downloading the output and error files upon completion.
    *   Parsing these files and re-ordering the results based on an internal mapping.
    This keeps the `BatchedGenerationPipeline` simple and unaware of the underlying file-based mechanism of the OpenAI API.

2.  **File Handling:**
    *   **Input File (`.jsonl`):** The `OpenAIBatchedModel.generate_batch` method creates a uniquely named `.jsonl` file for each batch request within a `preprocessed/` subdirectory of the pipeline's run-specific output directory (e.g., `data/batched_generations/TIMESTAMP/MODEL/preprocessed/`). This file contains the prompts formatted according to OpenAI's requirements, including a unique `custom_id` for each request.
    *   **Persistence:** These locally generated input files are *not* automatically deleted. This aids debugging and allows inspection of exactly what was sent to OpenAI.
    *   **Output/Error Files:** The `OpenAIBatchedModel.get_batch_results` method downloads the corresponding output and error files from OpenAI when results are requested. These are processed in memory to extract and reorder results, and the downloaded files are *not* persisted locally by the model itself (though the raw content could be saved by a results pipeline if needed).

3.  **Metadata Management:**
    *   The `OpenAIBatchedModel.generate_batch` returns a metadata dictionary containing the OpenAI `batch_id` and internal information needed for later steps (`_internal_input_file_id`, `_internal_input_filepath`, `_internal_custom_id_map`).
    *   The `BatchedGenerationPipeline` takes this metadata, adds its own context (`original_prompts_data`), and saves it immediately to a JSON file (`metadata/{batch_id}.json`).
    *   During monitoring, `check_batch_progress` updates the status and potentially adds output/error file IDs (`_internal_output_file_id`, `_internal_error_file_id`) to the metadata dictionary passed into it. The pipeline is responsible for persisting these updates back to the metadata file.

4.  **Result Ordering:** OpenAI's Batch API does not guarantee the order of results in the output file. The `OpenAIBatchedModel` handles this by:
    *   Creating a unique `custom_id` for each request when generating the input file.
    *   Storing a mapping (`_internal_custom_id_map`) from this `custom_id` back to the original 0-based index of the prompt in the input list.
    *   Using this map within `get_batch_results` to place the parsed results into a new list at the correct original index.

5.  **Error Handling:** The implementation includes `try...except` blocks around file operations and API calls. Errors during file upload or batch creation will raise exceptions. Errors during status checks are logged, and the pipeline will retry. Errors within individual requests in a batch are captured from the output/error files and returned as strings like `"ERROR: ..."` in the final results list.

## How the Batched Pipeline Works

1.  **Initialization:** The `BatchedGenerationPipeline` is created with a `Specification`, a `BatchedModel` instance (e.g., `OpenAIBatchedModel`), the path to pre-generated inputs, a batch size, and an output base directory.
2.  **Input Loading:** It reads all `.json` files from the `pregenerated_inputs_dir`, extracting inputs and associating them with their `statement_id` and original index within the file.
3.  **Batch Submission:** It iterates through the loaded inputs in chunks defined by `batch_size`.
    *   For each chunk, it calls `batched_model.generate_batch()`, passing the prompts and the run's output directory.
    *   The model (e.g., `OpenAIBatchedModel`) creates the local `.jsonl` file in `preprocessed/`, uploads it, creates the OpenAI batch job, and returns metadata (including `batch_id` and internal details).
    *   The pipeline adds the original prompt data to this metadata and saves it immediately to `metadata/{batch_id}.json`.
4.  **Progress Monitoring:** The pipeline enters a loop, checking the status of submitted batches periodically.
    *   It reads the metadata files (or uses its in-memory list).
    *   It calls `batched_model.check_batch_progress()` for each active batch, passing the metadata.
    *   The model retrieves the latest status from the provider (e.g., OpenAI) and updates the metadata dictionary (adding status, potentially output/error file IDs).
    *   The pipeline updates the corresponding `metadata/{batch_id}.json` file with the latest status and info from the model.
    *   The loop continues until all batches are in a terminal state (`completed` or `failed`) or a timeout occurs (default 24 hours).
5.  **Result Retrieval (Future/Separate Pipeline):**
    *   A separate process or a continuation of the current pipeline could be implemented.
    *   This process would iterate through the metadata files in the run directory.
    *   For batches marked as `completed`, it would read the metadata file.
    *   It would call `batched_model.get_batch_results()`, passing the complete metadata.
    *   The model (e.g., `OpenAIBatchedModel`) would download the necessary output/error files from the provider, parse them, use the internal `custom_id` map (retrieved from the metadata) to reorder results, and return the ordered list of strings/errors.
    *   The results retrieval pipeline would then take this ordered list, map it back to the `original_prompts_data` (also stored in the metadata file), and save the final structured results (e.g., grouped by `statement_id`) to the `results/` directory.

This design allows for robust batch processing using the OpenAI API while keeping the pipeline logic relatively clean and enabling potential future integration with other batch-capable providers through the `BatchedModel` interface.

## Examples (OpenAI)

To run the batch submission and monitoring pipeline using pre-generated inputs:

### gpt 4.1 openai model spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gpt-4.1-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```

### gpt4o (2024-11-20) openai model spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gpt-4o-2024-11-20 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```

### gpt4o_mini openai model spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gpt-4o-mini-2024-07-18 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```

### gpt4.1_mini openai model spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gpt-4.1-mini-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```

### gpt4.1 nano openai model spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gpt-4.1-nano-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```


### gpt 4.1 anthropic model spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gpt-4.1-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### gpt4o (2024-11-20) anthropic spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gpt-4o-2024-11-20 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### gpt4o_mini anthropic spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gpt-4o-mini-2024-07-18 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### gpt-4.1-mini anthropic spec

```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gpt-4.1-mini-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### gpt-4.1-mini anthropic spec

```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gpt-4.1-nano-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```


### gpt 4.1 google model spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gpt-4.1-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

### gpt4o (2024-11-20) google spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gpt-4o-2024-11-20 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

### gpt4o_mini google spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gpt-4o-mini-2024-07-18 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

### gpt4.1_mini google spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gpt-4.1-mini-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

### gpt4.1_nano google spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gpt-4.1-nano-2025-04-14 \
    --batch-size 50000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

## Example Usage (Anthropic)

To run the batch submission and monitoring pipeline using Anthropic:

### claude 3.7 sonnet, openai spec

```bash
# Set API Key
export ANTHROPIC_API_KEY='your_anthropic_api_key_here'

python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name claude-3-7-sonnet-20250219 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```

### claude 3.5 sonnet, openai spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name claude-3-5-sonnet-20240620 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```

### claude 3.5 haiku, openai spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name claude-3-5-haiku-20241022 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```


### claude 3.7 sonnet, anthropic spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name claude-3-7-sonnet-20250219 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### claude 3.5 sonnet, anthropic spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name claude-3-5-sonnet-20240620 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### claude 3.5 haiku, anthropic spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name claude-3-5-haiku-20241022 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### claude 3.7 sonnet, google spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name claude-3-7-sonnet-20250219 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

### claude 3.5 sonnet, google spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name claude-3-5-sonnet-20240620 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

### claude 3.5 haiku, google spec
```bash
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name claude-3-5-haiku-20241022 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

# TODO: Fix broken gemini google batch
# gemini on openai spec
```bash
# DO NOT USE, BROKEN!
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gemini-2.0-flash-001 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name openai \
     --input-bucket gs://levanter-data/model_spec/ \
    --output-bucket gs://levanter-data/model_spec/output \
    --verbose
```

# gemini on anthropic spec
```bash
# DO NOT USE, BROKEN!
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gemini-2.0-flash-001 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name anthropic \
     --input-bucket gs://levanter-data/model_spec/ \
    --output-bucket gs://levanter-data/model_spec/output \
    --verbose
```

# gemini on google spec
```bash
# DO NOT USE, BROKEN!
python -m batched_examples.run_batched_generation \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gemini-2.0-flash-001 \
    --batch-size 10000 \
    --output-base-dir data/batched_generations \
    --temperature 0.0 \
    --spec-name google \
     --input-bucket gs://levanter-data/model_spec/ \
    --output-bucket gs://levanter-data/model_spec/output \
    --verbose
```
**Note:** The script `run_batched_generation.py` only submits the jobs and monitors their status. A separate script or pipeline step is required to iterate through the output metadata, call `get_batch_results` for completed batches, and process/save the actual generations. Anthropic's Batch API supports up to 100,000 requests per batch; adjust `--batch-size` accordingly, keeping in mind the script will issue a warning if you exceed this, as the API call might fail.

## Alternative: Concurrent Batching with Together AI (Asyncio)

For providers like Together AI that offer an asynchronous client suitable for concurrent requests (using Python's `asyncio`), a different approach can be simpler and faster for scenarios where immediate results are desired.

The `batched_examples/together_batch_inference.py` script demonstrates this approach.

### How it Works (Together AI Example)

1.  **Initialization:** The script takes similar inputs: model name, pre-generated input directory, output directory, and a `batch_size` which now controls the number of *concurrent* `asyncio` requests.
    It also takes a `--qps` (Queries Per Second) argument to control the rate limiting.
2.  **Input Loading:** Loads prompts from JSON files, similar to the pipeline.
3.  **Concurrent Execution:**
    *   It iterates through the prompts in chunks defined by `batch_size`.
    *   For each chunk, it creates multiple `asyncio` tasks using `together.AsyncTogether.chat.completions.create`.
    *   **Rate Limiting:**
        *   For most models, an `asyncio.Semaphore` is used, controlled by the `--qps` argument, to limit the number of simultaneously active API requests.
        *   For models identified as having extremely low rate limits (specifically DeepSeek R1 based on checks in the script), it switches to sequential processing, sending one request at a time with a significant delay (`asyncio.sleep`) between each to avoid hitting the limit.
    *   `asyncio.gather(*tasks, return_exceptions=True)` is used to run these rate-limited API calls concurrently (or sequentially for low-limit models) within the script's execution.
    *   Crucially, when `asyncio.gather` returns, the script *immediately* has the results (or exceptions) for that chunk. There's no separate submission ID, status monitoring loop, or delayed result retrieval step needed.
4.  **Result Handling:** Errors are captured directly from `asyncio.gather`. Successful responses are parsed immediately.
5.  **Saving Results:** All results (outputs or error messages) are collected and saved to a single `results.json` file in a timestamped run directory at the end of the script.

### Trade-offs vs. OpenAI Batch API Pipeline

*   **Simplicity:** The `asyncio` approach is significantly simpler, requiring less code and no complex state management (metadata files, monitoring loops).
*   **Speed (for immediate results):** Since results are available as soon as the concurrent calls finish, this method is much faster if you need the generations immediately. The OpenAI Batch API can take minutes to hours.
*   **Rate Limiting (Updated):** The script now includes built-in rate limiting using `asyncio.Semaphore` controlled by `--qps` for standard models, and specific sequential delays for known ultra-low-limit models (like DeepSeek R1). This makes it more robust against 429 errors, but you still need to set `--qps` appropriately based on your Together AI tier and the specific model's limits.
*   **Scalability & Cost:** OpenAI's Batch API is designed for very large, potentially lower-priority workloads and may offer cost advantages at massive scale compared to running thousands of individual concurrent API calls.
*   **Resilience:** The `BatchedGenerationPipeline` with its persistent metadata is more resilient to script interruptions. You can potentially resume monitoring or retrieve results later. The `asyncio` script needs to run to completion to get all results; if interrupted, completed requests in that run are lost unless intermediate saving is added.

**Conclusion:** Use the `BatchedGenerationPipeline` (like `run_batched_generation.py`) for true asynchronous batch APIs like OpenAI's, especially for very large, non-time-sensitive jobs. Use the direct `asyncio` approach (like `together_batch_inference.py`) for providers like Together AI when you want simpler code and faster turnaround for concurrent requests, managing rate limits carefully.

### Example Usage (Together AI)

qwen3 235B
```bash
python -m batched_examples.together_batch_inference \
    --model-name Qwen/Qwen3-235B-A22B-fp8-tput \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --output-base-dir data/together_generations \
    --batch-size 10 \
    --qps 10 \
    --spec openai \
    --verbose
```

If you hit really bad rate limits just go back to sync basically

```bash
python -m batched_examples.together_batch_inference \
    --model-name Qwen/Qwen3-235B-A22B-fp8-tput \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --output-base-dir data/together_generations \
    --batch-size 1 \
    --qps 1 \
    --spec openai \
    --verbose
```



meta llama 4

openai spec
```bash
python -m batched_examples.together_batch_inference \
    --model-name meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --output-base-dir data/together_generations \
    --batch-size 10 \
    --qps 10 \
    --spec openai \
    --verbose
```

google spec
```bash
python -m batched_examples.together_batch_inference \
    --model-name meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --output-base-dir data/together_generations \
    --batch-size 10 \
    --qps 10 \
    --spec google \
    --verbose
```

anthropic spec
```bash
python -m batched_examples.together_batch_inference \
    --model-name meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --output-base-dir data/together_generations \
    --batch-size 10 \
    --qps 10 \
    --spec anthropic \
    --verbose
```

```bash
python -m batched_examples.together_batch_inference     --model-name Qwen/Qwen2-72B-Instruct     --pregenerated-inputs-dir data/adaptive_autobencher_outputs/gpt-4.1-2025-04-14/2025-05-01     --output-base-dir data/together_generations     --batch-size 100     --qps 10 --verbose
```

## Batched Ranking Pipeline

The `BatchedRankingPipeline` allows for ranking pre-generated outputs from two or more candidate models against each other, based on their alignment with policy statements from specification files. It uses a batch-capable LLM (e.g., from OpenAI or Anthropic) as the "judge" to perform the pairwise comparisons.

### How it Works

1.  **Input**:
    *   One or more specification files (JSONL format).
    *   Paths to directories, where each directory contains the generation results for a single candidate model. The name of the directory is used as the candidate model's identifier.
    *   Within each candidate model's directory, there should be JSON files named `{statement_id}.json`. These files should contain a list of objects, each with `"input_text"` and `"output_text"` keys, representing the model's responses to various inputs for that statement.
2.  **Model Pairing**: The pipeline creates all ordered pairs of candidate models (e.g., ModelA vs. ModelB, and ModelB vs. ModelA).
3.  **Prompt Generation**: For each statement and each common input found across a pair of candidate models, a ranking prompt is constructed. This prompt asks the judge model to compare `output_A` and `output_B` based on the given policy statement.
4.  **Batch Submission**: These ranking prompts are collected and submitted in batches to the specified ranking judge model (e.g., `OpenAIBatchedModel` or `AnthropicBatchedModel`). Metadata for each batch (including the original tasks) is saved.
5.  **Monitoring & Results**: The pipeline monitors the status of these batches. Once a batch is completed, the results (judge's raw responses) are retrieved.
6.  **Parsing & Saving**: The judge's raw text responses (e.g., "Model: A") are parsed into numerical scores (1 for A, -1 for B, 0 for equal/N/A). The final structured ranking results are saved in JSON files.

### Output Structure

Rankings are saved under:
`{output_dir_base}/spec_{SPEC}/judge_{JUDGE_MODEL}/{model_A_name}x{model_B_name}/results/{statement_id}.json`

Metadata for each batch is stored in a `metadata` subdirectory within the `{model_A_name}x{model_B_name}` folder.

### Example Usage (`run_batched_ranking.py`)

**1. Using CLI Arguments:**

This is possible but preferred method is yaml

**1. Using a YAML Configuration File:**

Create a `config_ranking.yaml` file:

```yaml
spec_name: openai
spec_path: data/specs/openai/jsonl/openai.jsonl
candidate_generation_dirs:
  - data/batched_generations/openai/claude-3-7-sonnet-20250219/20250506_115521/results # Replace with actual path to Model A's results
  - data/batched_generations/openai/gpt-4.1-2025-04-14/20250506_115500/results # Replace with actual path to Model B's results
ranking_judge_model_name: gpt-4o-mini-2024-07-18
ranking_judge_org: openai
output_dir_base: data/batched_rankings
batch_size: 25 # OpenAI Batch API has per-minute rate limits on batch creation too
temperature: 0.0
verbose: true
# openai_api_key: YOUR_KEY (or set OPENAI_API_KEY env var)
# anthropic_api_key: YOUR_KEY (or set ANTHROPIC_API_KEY env var)
```

Then run:

```bash
python -m batched_examples.run_batched_ranking --config-file batched_examples/configs/openai_ranking_config.yaml
```

**3. Using Google Gemini Batch Ranking:**

Export your Google API key (or use Application Default Credentials):
```bash
export GOOGLE_API_KEY='your_google_api_key_here'
```
Then either point to your Google config:
```bash
python -m batched_examples.run_batched_ranking --config-file batched_examples/configs/sample_google_ranking_config.yaml
```
or run directly:
```bash
python -m batched_examples.run_batched_ranking \
  --spec-name google \
  --spec-path data/specs/google/jsonl/google_spec.jsonl \
  --candidate-generation-dirs \
      data/batched_generations/modelA/202505XX_XXXXXX/results \
      data/batched_generations/modelB/202505XX_XXXXXX/results \
  --ranking-judge-model-name gemini-2.0-flash-001 \
  --ranking-judge-org google \
  --input-bucket gs://levanter-data/model_spec/ \
  --output-bucket gs://levanter-data/model_spec/ranking_outputs \
  --output-dir-base data/batched_rankings \
  --batch-size 10000 \
  --temperature 0.0 \
  --verbose
```

## Batched Binary Compliance Evaluation

The `run_batched_binary_compliance.py` script demonstrates how to evaluate each model's outputs for compliance against a policy specification using a batch-capable judge (OpenAI or Anthropic).

Example YAML configs live in `batched_examples/binary_configs/`:

```bash
python -m batched_examples.run_batched_binary_compliance --config-file batched_examples/binary_configs/openai_spec_openai_judge.yaml
```

This will:
1. Load the spec at `data/specs/openai/jsonl/openai.jsonl`.
2. Iterate over each candidate model directory from the config.
3. Batch and submit compliance-evaluation prompts (using `PROMPT_SUFFIX_COMPLIANCE_JUDGE`).
4. Poll the batch job until completion, retrieve raw judge responses, parse them into
   `{compliant: bool, confidence: float, explanation: string}`.
5. Write one JSON file per statement under:
   `data/batched_compliance/spec_openai/judge_<judge_model>/<candidate_model>/results/{statement_id}.json`

For more examples, see the six sample configs in `batched_examples/binary_configs/`.

### Binary compliance google judge

The batch mode for google is unfortunately broken, but the gemini flash model runs quickly enough that it's easy to just make a separate script.

```bash
python -m batched_examples.batched_binary_compliance_google --config-file batched_examples/binary_configs_google/anthropic_spec_google_judge.yaml

```

## Google model Generation

Batched generation is broken for google somehow...

gemini 2 flash
```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gemini-2.0-flash-001 \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

gemini 1.5 pro
```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gemini-1.5-pro	 \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

gemini 2.5 pro
```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/google/jsonl/google.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --model-name gemini-2.5-pro-preview-05-06	 \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name google \
    --verbose
```

### gemini 2.0 flash on Anthropic Spec

```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gemini-2.0-flash-001 \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### gemini 2.0 flash on OpenAI Spec

```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gemini-2.0-flash-001 \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```

### gemini 1.5 pro on Anthropic Spec

```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gemini-1.5-pro \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### gemini 1.5 pro on OpenAI Spec

```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gemini-1.5-pro \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```

### gemini 2.5 pro on Anthropic Spec

```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/anthropic/jsonl/anthropic.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --model-name gemini-2.5-pro-preview-05-06 \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name anthropic \
    --verbose
```

### gemini 2.5 pro on OpenAI Spec

```bash
python  examples/generate_google_models.py \
    --spec-path data/specs/openai/jsonl/openai.jsonl \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --model-name gemini-2.5-pro-preview-05-06 \
    --output-base-dir data/batched_google_generations \
    --temperature 0.0 \
    --spec-name openai \
    --verbose
```
