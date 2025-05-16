#!/usr/bin/env python3
"""
Example script for running batched binary compliance evaluation of model outputs.

This script takes generated outputs for each statement from one or more candidate model directories,
then uses a batch-capable compliance judge (OpenAI or Anthropic) to evaluate each response
for compliance with the given policy statements. Results (compliant flag, confidence, explanation)
are written per statement.
"""

import os
import argparse
import yaml
import time
import json
import datetime
from pathlib import Path
import logging
from collections import defaultdict

from speceval.parsers.jsonl import JsonlParser
from speceval.utils.logging import setup_logging
from speceval.utils.prompts import (
    PROMPT_SUFFIX_COMPLIANCE_JUDGE,
    build_evaluation_prompt_prefix,
    parse_judge_response,
)
from speceval.utils.parsing import extract_model_name_from_path
from speceval.models.openai import OpenAIBatchedModel, GPT_4O_MINI
from speceval.models.anthropic import AnthropicBatchedModel, CLAUDE_3_5_HAIKU


def main():
    parser = argparse.ArgumentParser(
        description="Run Batched Binary Compliance Evaluation for model outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", type=str, help="YAML configuration file path.")
    parser.add_argument("--spec-path", type=str, help="Path to a specification JSONL file.")
    parser.add_argument(
        "--spec-name",
        type=str,
        choices=["openai", "anthropic", "google"],
        help="Name of the spec organization (openai, anthropic, google).",
    )
    parser.add_argument(
        "--candidate-generation-dirs",
        type=str,
        nargs="+",
        help="List of directories containing model generation JSON files.",
    )
    parser.add_argument(
        "--compliance-judge-model-name",
        type=str,
        help=f"Batch-capable judge model (e.g., {GPT_4O_MINI}, {CLAUDE_3_5_HAIKU}).",
    )
    parser.add_argument(
        "--compliance-judge-org",
        type=str,
        choices=["openai", "anthropic"],
        help="Organization of the compliance judge model (openai or anthropic).",
    )
    parser.add_argument(
        "--output-dir-base",
        type=str,
        default="data/batched_compliance",
        help="Base directory to store compliance results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of compliance prompts per batch request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the compliance judge model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var).",
    )

    args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as cf:
            config = yaml.safe_load(cf) or {}

    # Helper: CLI overrides YAML
    def get(arg_val, cfg_key, default=None):
        if arg_val is not None and arg_val != default:
            return arg_val
        return config.get(cfg_key, default)

    # Resolve settings
    cfg_spec_path = get(args.spec_path, "spec_path")
    if not cfg_spec_path:
        parser.error("Argument --spec-path is required.")
    cfg_spec_name = get(args.spec_name, "spec_name")
    if not cfg_spec_name:
        parser.error("Argument --spec-name is required.")
    cfg_candidate_dirs = get(args.candidate_generation_dirs, "candidate_generation_dirs")
    if not cfg_candidate_dirs:
        parser.error("Argument --candidate-generation-dirs is required.")
    cfg_judge_model = get(args.compliance_judge_model_name, "compliance_judge_model_name")
    if not cfg_judge_model:
        parser.error("Argument --compliance-judge-model-name is required.")
    cfg_judge_org = get(args.compliance_judge_org, "compliance_judge_org")
    if not cfg_judge_org:
        parser.error("Argument --compliance-judge-org is required.")

    base = get(args.output_dir_base, "output_dir_base", "data/batched_compliance")
    output_base = Path(base) / f"spec_{cfg_spec_name}" / f"judge_{cfg_judge_model}"
    cfg_batch_size = get(args.batch_size, "batch_size", 100)
    cfg_temperature = get(args.temperature, "temperature", 0.0)
    cfg_verbose = args.verbose or config.get("verbose", False)
    cfg_openai_api_key = get(args.openai_api_key, "openai_api_key") or os.environ.get(
        "OPENAI_API_KEY"
    )
    cfg_anthropic_api_key = get(args.anthropic_api_key, "anthropic_api_key") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )

    # Setup logging
    logger = setup_logging(verbose=cfg_verbose, folder_name="run_batched_binary_compliance")
    logger.info("Starting Batched Binary Compliance Script...")

    # Resolve API key
    api_key = cfg_openai_api_key if cfg_judge_org == "openai" else cfg_anthropic_api_key
    if not api_key:
        raise ValueError(f"{cfg_judge_org.capitalize()} API key is required.")

    # Instantiate judge model
    if cfg_judge_org == "openai":
        judge = OpenAIBatchedModel(model_name=cfg_judge_model, api_key=api_key)
    else:
        judge = AnthropicBatchedModel(model_name=cfg_judge_model, api_key=api_key)

    # Load specification
    parser_jsonl = JsonlParser()
    spec = parser_jsonl.from_file(Path(cfg_spec_path))
    if not spec.statements:
        logger.error("No statements found in spec. Exiting.")
        return

    # Stage 1: submit all batches for all models
    contexts_by_model = {}
    metadata_dirs_by_model = {}
    results_dirs_by_model = {}
    all_batch_items = []  # list of dicts: {model, batch_meta}
    for model_dir in cfg_candidate_dirs:
        model_dir = Path(model_dir)
        model_name = extract_model_name_from_path(model_dir)
        logger.info(f"Submitting compliance tasks for model: {model_name}")

        metadata_dir = output_base / model_name / "metadata"
        results_dir = output_base / model_name / "results"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Build prompts and contexts
        prompts = []
        contexts = []
        for statement in spec.statements:
            gen_file = model_dir / f"{statement.id}.json"
            if not gen_file.exists():
                logger.warning(
                    f"Missing generation file for statement {statement.id} at {gen_file}"
                )
                continue
            with open(gen_file, "r", encoding="utf-8") as gf:
                data = json.load(gf)
            for i, entry in enumerate(data):
                inp = entry.get("input_text")
                out = entry.get("output_text")
                if inp is None or out is None:
                    logger.warning(f"Skipping invalid entry in {gen_file}: {entry}")
                    continue
                prompt_str = (
                    build_evaluation_prompt_prefix(statement, inp, out)
                    + PROMPT_SUFFIX_COMPLIANCE_JUDGE
                )
                cid = f"{statement.id}_{i}"
                prompts.append({"custom_id": cid, "input_text": prompt_str})
                contexts.append(
                    {
                        "custom_id": cid,
                        "statement_id": statement.id,
                        "input": inp,
                        "output": out,
                    }
                )
        if not prompts:
            logger.info(f"No valid prompts for model {model_name}. Skipping.")
            continue
        contexts_by_model[model_name] = contexts
        metadata_dirs_by_model[model_name] = metadata_dir
        results_dirs_by_model[model_name] = results_dir

        # Submit all batches for this model
        for start in range(0, len(prompts), cfg_batch_size):
            batch_chunk = prompts[start : start + cfg_batch_size]
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            batch_dir = metadata_dir / f"batch_{ts}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  - Submitting batch of {len(batch_chunk)} prompts for {model_name}")
            batch_meta = judge.generate_batch(
                prompts_data=batch_chunk,
                batch_output_dir=batch_dir,
                temperature=cfg_temperature,
            )
            all_batch_items.append({"model": model_name, "batch_meta": batch_meta})

    # Stage 2: monitor all submitted batches
    raw_map_by_model = defaultdict(dict)
    active_items = all_batch_items.copy()
    while active_items:
        next_active = []
        for item in active_items:
            batch_meta = item["batch_meta"]
            updated_meta = judge.check_batch_progress(batch_meta)
            status = updated_meta.get("status", "").lower()
            if status == "completed":
                raw_map = judge.get_batch_results(updated_meta)
                raw_map_by_model[item["model"]].update(raw_map)
            elif status in [
                "processing",
                "submitted",
                "pending",
                "running",
                "validating",
                "in_progress",
                "finalizing",
            ]:
                next_active.append({"model": item["model"], "batch_meta": updated_meta})
            else:
                logger.error(f"Batch {updated_meta.get('batch_id')} ended with status: {status}")
        active_items = next_active
        if active_items:
            logger.info(f"Waiting for {len(active_items)} batches to complete...")
            time.sleep(10)

    # Stage 3: aggregate and save results per statement
    for model_name, contexts in contexts_by_model.items():
        raw_map = raw_map_by_model.get(model_name, {})
        results_by_statement = {}
        for ctx in contexts:
            parsed = parse_judge_response(
                raw_map.get(ctx["custom_id"], ""), mode=None, statement_id=ctx["statement_id"]
            )
            results_by_statement.setdefault(ctx["statement_id"], []).append(
                {
                    "input": ctx["input"],
                    "output": ctx["output"],
                    "compliant": parsed.get("compliant"),
                    "confidence": parsed.get("confidence"),
                    "explanation": parsed.get("explanation"),
                }
            )
        for statement_id, res_list in results_by_statement.items():
            out_file = results_dirs_by_model[model_name] / f"{statement_id}.json"
            with open(out_file, "w", encoding="utf-8") as outf:
                json.dump(
                    {
                        "metadata": {
                            "model": model_name,
                            "statement_id": statement_id,
                            "batch_ids": [
                                bm["batch_meta"].get("batch_id")
                                for bm in all_batch_items
                                if bm["model"] == model_name
                            ],
                        },
                        "results": res_list,
                    },
                    outf,
                    indent=2,
                )

    logger.info("Batched binary compliance evaluation finished.")


if __name__ == "__main__":
    main()
