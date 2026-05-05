"""
Prepare user-level splits and 50-user evaluation groups for query linkage experiments.

This script does NOT train a model. It creates:
- train/val/test user splits by AnonID
- manifests of sampled 50-user groups for validation and test

Example:
    python linkage_prepare.py \
        --raw-path raw.parquet \
        --out-dir linkage_work \
        --min-queries-per-user 20 \
        --max-queries-per-user 5000 \
        --train-users 5000 \
        --val-users 25000 \
        --test-users 5000 \
        --group-size 50 \
        --val-groups 500 \
        --test-groups 100 \
        --seed 13
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_PATH = BASE_DIR / "raw.parquet"
DEFAULT_OUT_DIR = BASE_DIR / "linkage_work"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AnonID splits and 50-user group manifests for linkage experiments."
        )
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help=f"Path to raw.parquet (default: {DEFAULT_RAW_PATH}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--min-queries-per-user",
        type=int,
        default=20,
        help="Keep users with at least this many queries.",
    )
    parser.add_argument(
        "--max-queries-per-user",
        type=int,
        default=5000,
        help="Keep users with at most this many queries.",
    )
    parser.add_argument(
        "--train-users",
        type=int,
        default=5000,
        help="Number of users in training split.",
    )
    parser.add_argument(
        "--val-users",
        type=int,
        default=25000,
        help="Number of users in validation split.",
    )
    parser.add_argument(
        "--test-users",
        type=int,
        default=5000,
        help="Number of users in test split.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=50,
        help="Users per evaluation group.",
    )
    parser.add_argument(
        "--val-groups",
        type=int,
        default=500,
        help="Number of sampled validation groups.",
    )
    parser.add_argument(
        "--test-groups",
        type=int,
        default=100,
        help="Number of sampled test groups.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )
    return parser.parse_args()


def load_user_counts(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"raw.parquet not found: {raw_path}")

    df = pd.read_parquet(raw_path, columns=["AnonID"])
    counts = (
        df["AnonID"]
        .dropna()
        .astype("int64")
        .value_counts(sort=False)
        .rename_axis("AnonID")
        .reset_index(name="query_count")
    )
    return counts


def sample_user_splits(
    users: np.ndarray,
    train_users: int,
    val_users: int,
    test_users: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    needed = train_users + val_users + test_users
    if users.size < needed:
        raise ValueError(
            f"Not enough eligible users ({users.size:,}) for requested splits ({needed:,})."
        )

    rng = np.random.default_rng(seed)
    shuffled = users.copy()
    rng.shuffle(shuffled)

    train = shuffled[:train_users]
    val = shuffled[train_users : train_users + val_users]
    test = shuffled[train_users + val_users : needed]
    return train, val, test


def build_group_manifest(
    split_name: str,
    split_users: np.ndarray,
    group_size: int,
    n_groups: int,
    seed: int,
) -> pd.DataFrame:
    if split_users.size < group_size:
        raise ValueError(
            f"Split {split_name} has {split_users.size:,} users, needs at least {group_size}."
        )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, int | str]] = []

    for group_id in range(n_groups):
        picked = rng.choice(split_users, size=group_size, replace=False)
        for anon_id in picked:
            rows.append(
                {
                    "split": split_name,
                    "group_id": group_id,
                    "AnonID": int(anon_id),
                }
            )

    return pd.DataFrame(rows)


def write_users(path: Path, split: str, users: np.ndarray) -> None:
    out = pd.DataFrame({"split": split, "AnonID": users.astype("int64")})
    out.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()

    if args.min_queries_per_user <= 0:
        raise ValueError("--min-queries-per-user must be > 0")
    if args.max_queries_per_user < args.min_queries_per_user:
        raise ValueError("--max-queries-per-user must be >= --min-queries-per-user")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AnonID counts from: {args.raw_path}")
    counts = load_user_counts(args.raw_path)

    eligible = counts[
        (counts["query_count"] >= args.min_queries_per_user)
        & (counts["query_count"] <= args.max_queries_per_user)
    ].copy()

    print(f"Total users: {len(counts):,}")
    print(
        "Eligible users after query-count filter "
        f"[{args.min_queries_per_user}, {args.max_queries_per_user}]: {len(eligible):,}"
    )

    train_users, val_users, test_users = sample_user_splits(
        users=eligible["AnonID"].to_numpy(dtype=np.int64),
        train_users=args.train_users,
        val_users=args.val_users,
        test_users=args.test_users,
        seed=args.seed,
    )

    write_users(out_dir / "train_users.parquet", "train", train_users)
    write_users(out_dir / "val_users.parquet", "val", val_users)
    write_users(out_dir / "test_users.parquet", "test", test_users)

    val_manifest = build_group_manifest(
        split_name="val",
        split_users=val_users,
        group_size=args.group_size,
        n_groups=args.val_groups,
        seed=args.seed + 101,
    )
    test_manifest = build_group_manifest(
        split_name="test",
        split_users=test_users,
        group_size=args.group_size,
        n_groups=args.test_groups,
        seed=args.seed + 202,
    )

    group_manifest = pd.concat([val_manifest, test_manifest], ignore_index=True)
    group_manifest.to_parquet(out_dir / "group_manifest.parquet", index=False)

    summary = pd.DataFrame(
        [
            {
                "total_users": int(len(counts)),
                "eligible_users": int(len(eligible)),
                "train_users": int(len(train_users)),
                "val_users": int(len(val_users)),
                "test_users": int(len(test_users)),
                "group_size": int(args.group_size),
                "val_groups": int(args.val_groups),
                "test_groups": int(args.test_groups),
                "seed": int(args.seed),
                "min_queries_per_user": int(args.min_queries_per_user),
                "max_queries_per_user": int(args.max_queries_per_user),
            }
        ]
    )
    summary.to_parquet(out_dir / "split_summary.parquet", index=False)

    print(f"Wrote: {out_dir / 'train_users.parquet'}")
    print(f"Wrote: {out_dir / 'val_users.parquet'}")
    print(f"Wrote: {out_dir / 'test_users.parquet'}")
    print(f"Wrote: {out_dir / 'group_manifest.parquet'}")
    print(f"Wrote: {out_dir / 'split_summary.parquet'}")


if __name__ == "__main__":
    main()
