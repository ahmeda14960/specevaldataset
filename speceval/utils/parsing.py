"""Utility functions for parsing strings and paths within SpecEval."""

from pathlib import Path
from typing import Set, List

# --- Model Name Constants --- #
# OpenAI
GPT_4O = "gpt-4o-2024-11-20"
GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
GPT_4_1 = "gpt-4.1-2025-04-14"
GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"

# Anthropic
CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"

# Google
GEMINI_2_0_FLASH = "gemini-2.0-flash-001"
GEMINI_1_5_FLASH = "gemini-1.5-pro"

# Together
DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
QWEN_235B_FP8 = "Qwen/Qwen3-235B-A22B-fp8-tput"
QWEN_2_5_72B_TURBO = "Qwen/Qwen2.5-72B-Instruct-Turbo"
QWEN_2_72B_INSTRUCT = "Qwen/Qwen2-72B-Instruct"
LLAMA_4_MAVERICK_17B = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
LLAMA_3_1_405B_TURBO = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

ALL_KNOWN_MODEL_NAMES: Set[str] = {
    GPT_4O,
    GPT_4O_MINI,
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
    CLAUDE_3_7_SONNET,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_5_HAIKU,
    GEMINI_2_0_FLASH,
    GEMINI_1_5_FLASH,
    DEEPSEEK_V3,
    QWEN_235B_FP8,
    QWEN_2_5_72B_TURBO,
    QWEN_2_72B_INSTRUCT,
    LLAMA_4_MAVERICK_17B,
    LLAMA_3_1_405B_TURBO,
}


def extract_model_name_from_path(dir_path: Path) -> str:
    """
    Extracts a unique, known model name from any part of the given directory path string.

    Args:
        dir_path: The Path object representing the directory path.

    Returns:
        The single model name found in the path.

    Raises:
        ValueError: If zero or more than one known model name is found in the path string.
    """
    path_str = str(dir_path)
    found_model_names: List[str] = []

    for model_name in ALL_KNOWN_MODEL_NAMES:
        # Replace slashes with dashes in model names so we don't confuse them with directory separators
        model_name = model_name.replace("/", "-")
        if model_name in path_str:
            found_model_names.append(model_name)

    if not found_model_names:
        raise ValueError(
            f"No known model name found in path: {path_str}. Known models: {ALL_KNOWN_MODEL_NAMES}"
        )

    if len(found_model_names) > 1:
        # Try to find the longest match if multiple are found (e.g. gpt-4o and gpt-4o-mini)
        # This handles cases where one model name is a substring of another.
        longest_match = ""
        for name in found_model_names:
            if len(name) > len(longest_match):
                longest_match = name

        # Check if all other found names are substrings of the longest match
        # This is a simple heuristic. More complex disambiguation might be needed for tricky cases.
        all_substrings_of_longest = True
        temp_found_models = []
        for name in found_model_names:
            if name == longest_match:
                temp_found_models.append(name)
            elif name not in longest_match:
                all_substrings_of_longest = False
                break  # Not a simple substring case

        if all_substrings_of_longest and longest_match:
            # Check if after removing substrings, we are left with only one distinct superstring model name
            distinct_superstrings = {name for name in found_model_names if name == longest_match}
            if len(distinct_superstrings) == 1:
                return longest_match
            else:
                # This case implies multiple different "longest" matches or complex overlaps not handled by simple substring logic.
                raise ValueError(
                    f"Ambiguous model names found in path: {path_str}. Matches: {found_model_names}. "
                    f"Distinct superstrings considered: {distinct_superstrings}. Please ensure paths are uniquely identifiable."
                )
        else:
            # If not all are substrings of the longest, it's a genuine ambiguity
            raise ValueError(
                f"Multiple distinct (non-substring) known model names found in path: {path_str}. Matches: {found_model_names}. "
                f"Please ensure paths uniquely identify a single model from the known list."
            )

    return found_model_names[0]
