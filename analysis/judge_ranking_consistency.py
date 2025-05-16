#!/usr/bin/env python3
"""
Script to check consistency of rankings between two flipped-order directories from the same judge.
For each statement, verifies that:
  1) metadata.model_a and model_b are swapped
  2) each ranking entry's 'rank' is inverted (1 -> -1, -1 -> 1, 0 -> 0)
Outputs a report with total statements, matched count, and details of any mismatches.
"""
import argparse
import json
import logging
from pathlib import Path
import yaml


def load_ranking_files(dir_path: Path):
    """
    Load all JSON files in the directory into a dict mapping statement_id -> data.
    """
    data = {}
    for file in dir_path.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                record = json.load(f)
            stmt = record.get("metadata").get("statement_id")
            if not stmt:
                logging.warning(f"No statement_id in metadata of {file.name}, skipping.")
                continue
            data[stmt] = record
        except Exception as e:
            logging.warning(f"Failed to load {file}: {e}")
    return data


def invert_rank(rank_value):
    """
    Invert rank: 1->-1, -1->1, 0->0
    """
    if rank_value == 1:
        return -1
    if rank_value == -1:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check consistency of A_vs_B vs B_vs_A ranking directories."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="analysis/judge_ranking_consistency.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as cfgf:
        cfg = yaml.safe_load(cfgf)

    # Load input directories from config
    dir1 = Path(cfg["first_dir"])
    dir2 = Path(cfg["second_dir"])
    # Base directory for report output
    report_base = Path(cfg.get("output_report", "analysis/judge_ranking_consistency/"))
    verbose = cfg.get("verbose", False)

    # setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Load both directories
    logger.info(f"Loading first directory from {dir1}")
    logger.info(f"Loading second directory from {dir2}")
    data1 = load_ranking_files(dir1)
    data2 = load_ranking_files(dir2)

    # Determine canonical (alphabetical) model order from first directory's metadata
    if data1:
        sample = next(iter(data1.values()))
        m1, m2 = sample["metadata"].get("model_a"), sample["metadata"].get("model_b")
        canonical = sorted([m1, m2]) if m1 and m2 else [m1 or m2, m2 or m1]
    else:
        canonical = ["modelA", "modelB"]
    canA, canB = canonical[0], canonical[1]

    # Assign which folder is A_vs_B and which is flipped by matching metadata
    if data1 and (m1, m2) == (canA, canB):
        data_ab, data_ba = data1, data2
        dir_ab, dir_ba = dir1, dir2
    else:
        data_ab, data_ba = data2, data1
        dir_ab, dir_ba = dir2, dir1

    logger.info(f"Using canonical model order: {canA} vs {canB}")
    logger.info(f"Loading canonical A_vs_B from {dir_ab}")
    logger.info(f"Loading flipped B_vs_A from {dir_ba}")

    # Derive judge_dir from A_vs_B metadata
    if data_ab:
        rec0 = next(iter(data_ab.values()))
        metadata = rec0.get("metadata")
        judge_info = metadata.get("ranking_judge_model_info", {})
        if not judge_info:
            judge_info = metadata.get("ranking_model")
        provider = judge_info.get("provider", "unknown_provider")
        judge_model = judge_info.get("model_name", "unknown_model")
        judge_dir = f"{provider}-{judge_model}"
    else:
        judge_dir = "unknown_judge"

    # Use canonical pair directory name
    pair_dir = f"{canA}x{canB}"
    report_dir = report_base / judge_dir / pair_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "consistency_report.json"

    # Initialize prompt-level counters
    total_prompts = 0
    matched_prompts = 0
    # accumulate for correlation
    sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0
    mismatches = []

    for stmt, rec_ab in data_ab.items():
        if stmt not in data_ba:
            mismatches.append({"statement_id": stmt, "reason": "missing flipped file"})
            continue
        rec_ba = data_ba[stmt]
        # check metadata flip
        a1 = rec_ab["metadata"].get("model_a")
        b1 = rec_ab["metadata"].get("model_b")
        a2 = rec_ba["metadata"].get("model_a")
        b2 = rec_ba["metadata"].get("model_b")
        if not (a1 and b1 and a2 and b2):
            mismatches.append({"statement_id": stmt, "reason": "missing model metadata"})
            continue
        if not (a2 == b1 and b2 == a1):
            mismatches.append(
                {
                    "statement_id": stmt,
                    "reason": "metadata not flipped",
                    "models_ab": (a1, b1),
                    "models_ba": (a2, b2),
                }
            )
            continue
        # build mapping from input text to rank for both runs
        map_ab = {item.get("input"): item.get("rank") for item in rec_ab.get("rankings", [])}
        map_ba = {item.get("input"): item.get("rank") for item in rec_ba.get("rankings", [])}
        # ensure both runs cover the same set of inputs
        inputs_ab = set(map_ab.keys())
        inputs_ba = set(map_ba.keys())
        if inputs_ab != inputs_ba:
            mismatches.append(
                {
                    "statement_id": stmt,
                    "reason": "input set mismatch",
                    "missing_in_ab": list(inputs_ba - inputs_ab),
                    "missing_in_ba": list(inputs_ab - inputs_ba),
                }
            )
            continue
        # compare ranks per input (prompt-level)
        for inp in inputs_ab:
            r1 = map_ab.get(inp)
            expected = invert_rank(r1)
            r2 = map_ba.get(inp)
            # update counters
            total_prompts += 1
            # accumulate for correlation between expected and actual
            sum_x += expected
            sum_y += r2
            sum_xy += expected * r2
            sum_x2 += expected * expected
            sum_y2 += r2 * r2
            if r2 == expected:
                matched_prompts += 1
            else:
                mismatches.append(
                    {
                        "statement_id": stmt,
                        "input": inp,
                        "rank_ab": r1,
                        "expected_ba": expected,
                        "rank_ba": r2,
                    }
                )

    # Compute prompt-level statistics
    mismatch_prompts = total_prompts - matched_prompts
    match_fraction = (matched_prompts / total_prompts) if total_prompts > 0 else 0.0
    # Compute Pearson correlation between expected and actual ranks
    if total_prompts > 0:
        mean_x = sum_x / total_prompts
        mean_y = sum_y / total_prompts
        cov = (sum_xy / total_prompts) - (mean_x * mean_y)
        var_x = (sum_x2 / total_prompts) - (mean_x * mean_x)
        var_y = (sum_y2 / total_prompts) - (mean_y * mean_y)
        if var_x > 0 and var_y > 0:
            corr = cov / ((var_x * var_y) ** 0.5)
        else:
            corr = 0.0
    else:
        corr = 0.0
    report = {
        "total_statements": len(data_ab),
        "total_prompts": total_prompts,
        "matched_prompts": matched_prompts,
        "mismatched_prompts": mismatch_prompts,
        "match_fraction": match_fraction,
        "prompt_correlation": corr,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    # Write out the consistency report
    with open(report_path, "w") as outf:
        json.dump(report, outf, indent=2)

    logger.info(
        f"Done: {matched_prompts}/{total_prompts} statements matched, {len(mismatches)} mismatches."
    )
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
