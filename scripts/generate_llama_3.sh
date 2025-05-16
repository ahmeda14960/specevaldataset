#!/bin/bash

# Script to run together_batch_inference three times sequentially for different specs

COMMAND_OPENAI="python -m batched_examples.together_batch_inference \
    --model-name meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/openai/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250504_1935 \
    --output-base-dir data/together_generations \
    --batch-size 1 \
    --qps 1 \
    --spec openai \
    --verbose"

COMMAND_GOOGLE="python -m batched_examples.together_batch_inference \
    --model-name meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/google/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1859 \
    --output-base-dir data/together_generations \
    --batch-size 1 \
    --qps 1 \
    --spec google \
    --verbose"

COMMAND_ANTHROPIC="python -m batched_examples.together_batch_inference \
    --model-name meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo \
    --pregenerated-inputs-dir data/adaptive_autobencher_outputs/anthropic/input_judge/gpt-4.1-2025-04-14xgpt-4.1-2025-04-14/20250506_1851 \
    --output-base-dir data/together_generations \
    --batch-size 1 \
    --qps 1 \
    --spec anthropic \
    --verbose"

echo "Running OpenAI spec command:"
eval $COMMAND_OPENAI
echo "OpenAI spec command finished."

echo "Running Google spec command:"
eval $COMMAND_GOOGLE
echo "Google spec command finished."

echo "Running Anthropic spec command:"
eval $COMMAND_ANTHROPIC
echo "Anthropic spec command finished."

echo "All commands finished."
