"""
Run the full linkage pipeline end-to-end.

Steps:
    1. linkage_prepare.py      - build user splits and group manifests
    2. linkage_build_pairs.py  - build training pairs with features
    3. linkage_train_model.py  - train and calibrate pairwise model
    4. linkage_pair_eval.py    - score all pairs per group and sweep thresholds

All arguments passed after -- are forwarded as step-specific overrides; use
--step-N-args to set arguments for step N. Example:

        python linkage_run_all.py \
                --step-1-args "--min-queries-per-user 20 --train-users 5000" \
                --step-2-args "--positive-pairs-per-user 200 --negative-to-positive-ratio 2" \
                --step-3-args "--max-iter 300" \
                --step-4-args "--split val --max-groups 20 --max-queries-per-user 100 --top-n 20"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STEPS = [
    ("1", "linkage_prepare.py"),
    ("2", "linkage_build_pairs.py"),
    ("3", "linkage_train_model.py"),
    ("4", "linkage_pair_eval.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full linkage pipeline end-to-end."
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Step to start from (default: 1). Useful for resuming after a failure.",
    )
    parser.add_argument(
        "--stop-step",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Step to stop after (default: 4).",
    )
    for num, _ in STEPS:
        parser.add_argument(
            f"--step-{num}-args",
            default="",
            help=f"Extra arguments for step {num} (as a quoted string).",
        )
    return parser.parse_args()


def run_step(step_num: str, script_name: str, extra_args: list[str]) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"\n{'=' * 60}")
    print(f"Step {step_num}: {script_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"\nStep {step_num} failed with exit code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nStep {step_num} completed successfully.")


def main() -> None:
    args = parse_args()

    if args.start_step > args.stop_step:
        print("--start-step must be <= --stop-step.", file=sys.stderr)
        sys.exit(1)

    step_extra_args = {
        "1": shlex.split(args.step_1_args),
        "2": shlex.split(args.step_2_args),
        "3": shlex.split(args.step_3_args),
        "4": shlex.split(args.step_4_args),
    }

    for num, script in STEPS:
        if int(num) < args.start_step or int(num) > args.stop_step:
            continue
        run_step(num, script, step_extra_args[num])

    print(f"\n{'=' * 60}")
    print("All requested steps completed.")


if __name__ == "__main__":
    main()
