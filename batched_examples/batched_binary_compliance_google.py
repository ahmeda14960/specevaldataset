#!/usr/bin/env python3
"""
Script for running binary compliance evaluation of model outputs using Google Gemini as judge.
"""
import os
import argparse
import yaml
import json
from pathlib import Path

from speceval.parsers.jsonl import JsonlParser
from speceval.utils.logging import setup_logging
from speceval.utils.parsing import extract_model_name_from_path
from speceval.models.google import GoogleJudgeModel


def main():
    parser = argparse.ArgumentParser(
        description="Run Binary Compliance Evaluation using Google Gemini as judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", type=str, help="YAML configuration file path.")
    parser.add_argument("--spec-path", type=str, help="Path to a specification JSONL file.")
    parser.add_argument(
        "--spec-name",
        type=str,
        choices=["openai", "anthropic", "google"],
        help="Name of the spec organization.",
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
        help="Google Gemini model to use as compliance judge (e.g., gemini-2.0-flash-001).",
    )
    parser.add_argument(
        "--compliance-judge-org",
        type=str,
        help="Must be 'google' for this script.",
    )
    parser.add_argument(
        "--output-dir-base",
        type=str,
        default="data/batched_compliance",
        help="Base directory to store compliance results.",
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

    args = parser.parse_args()

    # Load YAML config if provided
    config = {}
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as cf:
            config = yaml.safe_load(cf) or {}

    def get(arg_val, cfg_key, default=None):
        if arg_val is not None and arg_val != default:
            return arg_val
        return config.get(cfg_key, default)

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
    if cfg_judge_org.lower() != "google":
        parser.error("--compliance-judge-org must be 'google'")

    cfg_output_base = get(args.output_dir_base, "output_dir_base", "data/batched_compliance")
    cfg_temperature = get(args.temperature, "temperature", 0.0)
    cfg_verbose = args.verbose or config.get("verbose", False)

    # Setup logging
    logger = setup_logging(verbose=cfg_verbose, folder_name="batched_binary_compliance_google")
    logger.info("Starting Google Binary Compliance Script...")

    # Instantiate Google judge model
    judge = GoogleJudgeModel(model_name=cfg_judge_model)

    # Load specification
    parser_jsonl = JsonlParser()
    spec = parser_jsonl.from_file(Path(cfg_spec_path))
    if not spec.statements:
        logger.error("No statements found in spec. Exiting.")
        return

    output_base = Path(cfg_output_base) / f"spec_{cfg_spec_name}" / f"judge_{cfg_judge_model}"

    # Iterate candidate models
    for model_dir in cfg_candidate_dirs:
        model_dir = Path(model_dir)
        model_name = extract_model_name_from_path(model_dir)
        logger.info(f"Evaluating compliance for model: {model_name}")

        results_dir = output_base / model_name / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Collect and evaluate
        results_by_statement = {}
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
                # Evaluate compliance synchronously
                res = judge.evaluate_compliance(statement, inp, out, temperature=cfg_temperature)
                results_by_statement.setdefault(statement.id, []).append(
                    {
                        "input": inp,
                        "output": out,
                        "compliant": res.get("compliant"),
                        "confidence": res.get("confidence"),
                        "explanation": res.get("explanation"),
                    }
                )

        # Write per-statement results
        for statement_id, res_list in results_by_statement.items():
            out_file = results_dir / f"{statement_id}.json"
            with open(out_file, "w", encoding="utf-8") as outf:
                json.dump(
                    {
                        "metadata": {
                            "model": model_name,
                            "statement_id": statement_id,
                            "judge_model": cfg_judge_model,
                        },
                        "results": res_list,
                    },
                    outf,
                    indent=2,
                )

    logger.info("Google Binary Compliance Evaluation finished.")


if __name__ == "__main__":
    main()
