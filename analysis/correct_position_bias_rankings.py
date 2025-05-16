#!/usr/bin/env python3
"""
Correct position bias in batched ranking JSONs by producing root-level corrected statement files.
For each spec_<spec>/judge_<judge>/<modelA>x<modelB> folder, it finds both A×B and B×A runs,
compares each statement's rankings by input, and only awards a win if rank_AB == -rank_BA;
otherwise marks a tie (rank=0). Outputs new JSONs named <statement>.json directly under the
canonical modelA×modelB directory.
"""
import sys
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debias batched ranking results by requiring inverted consistency."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/batched_rankings",
        help="Base directory containing spec_<spec>/judge_<judge>/<modelAxmodelB>/results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/batched_rankings_corrected",
        help="Base output directory where corrected JSONs will be saved (mirrors input structure)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base = Path(args.base_dir)
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    if not base.is_dir():
        sys.exit(f"Error: base_dir not found: {base}")

    for spec_dir in sorted(base.iterdir()):
        if not spec_dir.is_dir():
            continue
        spec = (
            spec_dir.name.removeprefix("spec_")
            if spec_dir.name.startswith("spec_")
            else spec_dir.name
        )
        for judge_dir in sorted(spec_dir.iterdir()):
            if not judge_dir.is_dir():
                continue
            judge = (
                judge_dir.name.removeprefix("judge_")
                if judge_dir.name.startswith("judge_")
                else judge_dir.name
            )
            print(f"Processing spec={spec}, judge={judge}")
            # collect all ordered-pair directories
            ordered = [d for d in judge_dir.iterdir() if d.is_dir() and "x" in d.name]
            # build unique unordered pairs
            unique_pairs = set()
            for d in ordered:
                a, b = d.name.split("x", 1)
                unique_pairs.add(tuple(sorted((a, b))))
            # process each canonical pair
            for a, b in sorted(unique_pairs):
                canon_name = f"{a}x{b}"
                dir_ab = judge_dir / canon_name
                dir_ba = judge_dir / f"{b}x{a}"
                if not (dir_ab.is_dir() and dir_ba.is_dir()):
                    print(f"  Skipping pair {a}×{b}: missing both orders")
                    continue
                res_ab = dir_ab / "results"
                res_ba = dir_ba / "results"
                if not (res_ab.is_dir() and res_ba.is_dir()):
                    print(f"  Skipping pair {a}×{b}: missing 'results' dir in one order")
                    continue
                # find shared statements
                stems_ab = {p.stem for p in res_ab.glob("*.json")}
                stems_ba = {p.stem for p in res_ba.glob("*.json")}
                common = sorted(stems_ab & stems_ba)
                if not common:
                    print(f"  Skipping pair {a}×{b}: no common statement files")
                    continue
                # prepare mirrored results directory
                out_res = out_base / spec_dir.name / judge_dir.name / canon_name / "results"
                out_res.mkdir(parents=True, exist_ok=True)
                # write corrected JSONs at root of dir_ab
                for stmt in common:
                    file_ab = res_ab / f"{stmt}.json"
                    file_ba = res_ba / f"{stmt}.json"
                    data_ab = json.load(open(file_ab, "r"))
                    data_ba = json.load(open(file_ba, "r"))
                    map_ab = {r["input"]: r for r in data_ab.get("rankings", [])}
                    map_ba = {r["input"]: r["rank"] for r in data_ba.get("rankings", [])}
                    corrected = []
                    for inp, rec in map_ab.items():
                        rank_ab = rec.get("rank", 0)
                        rank_ba = map_ba.get(inp)
                        final_rank = rank_ab if rank_ab == -rank_ba else 0
                        corrected.append(
                            {
                                "input": inp,
                                "output_a": rec.get("output_a", ""),
                                "output_b": rec.get("output_b", ""),
                                "rank": final_rank,
                            }
                        )
                    out_path = out_res / f"{stmt}.json"
                    with open(out_path, "w") as f:
                        json.dump({"rankings": corrected}, f, indent=2)
                print(f"  Wrote {len(common)} corrected JSONs to {out_res}")


if __name__ == "__main__":
    main()
