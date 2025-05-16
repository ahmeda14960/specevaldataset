#!/usr/bin/env python3
"""
Wrapper to run ranking for each unique pair of candidate generation directories.
Reads a base YAML config containing 'candidate_generation_dirs' and other settings,
enumerates all combinations of two directories, writes a temporary config for each pair,
and calls examples/rank_outputs_singleton.py via subprocess.
"""

import argparse
import yaml
import itertools
import tempfile
import subprocess
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run rank_outputs_singleton.py for each unique pair of candidate dirs."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to base YAML config with candidate_generation_dirs and other settings",
    )
    parser.add_argument(
        "--script-path",
        type=str,
        default="examples/rank_outputs_singleton.py",
        help="Path to the singleton ranking script",
    )
    args = parser.parse_args()

    # Load base config
    base_config_path = Path(args.config_file)
    if not base_config_path.exists():
        print(f"Base config not found: {base_config_path}")
        exit(1)
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    cand_dirs = base_config.get("candidate_generation_dirs")
    if not cand_dirs or len(cand_dirs) < 2:
        print("Config must include at least two 'candidate_generation_dirs'")
        exit(1)

    # Enumerate unique unordered pairs and launch processes
    processes = []
    for dir_a, dir_b in itertools.combinations(cand_dirs, 2):
        print(f"Launching ranking for pair:\n  A: {dir_a}\n  B: {dir_b}\n")
        # Prepare pair-specific config
        pair_config = dict(base_config)
        pair_config["candidate_generation_dirs"] = [dir_a, dir_b]

        # Write to temporary YAML file
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(pair_config, tf)
        tf.flush()
        temp_path = tf.name
        tf.close()

        # Invoke the singleton ranking script in background
        cmd = [os.getenv("PYTHON", "python3"), args.script_path, "--config-file", temp_path]
        try:
            proc = subprocess.Popen(cmd)
            processes.append((proc, temp_path, dir_a, dir_b))
        except Exception as e:
            print(f"Failed to start process for {dir_a} vs {dir_b}: {e}")
            os.remove(temp_path)

    # Wait for all processes to complete
    for proc, temp_path, dir_a, dir_b in processes:
        ret = proc.wait()
        if ret != 0:
            print(f"Error in process for pair {dir_a} vs {dir_b}: exit code {ret}")
        # Clean up temporary config file
        try:
            os.remove(temp_path)
        except OSError as e:
            print(f"Failed to remove temp config {temp_path}: {e}")


if __name__ == "__main__":
    main()
