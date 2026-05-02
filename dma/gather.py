"""
Build dma/raw.parquet from all user-ct-test-collection*.txt AOL files by loading
them into one DataFrame, coercing missing fields to NULL, dropping rows missing
AnonID/Query/QueryTime, compressing adjacent duplicates by (AnonID, Query,
ItemRank, ClickURL) while keeping only the last record in each run, typing
columns, and sorting by QueryTime ascending.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DESCRIPTION = (
    "Build dma/raw.parquet from all user-ct-test-collection*.txt AOL files by loading "
    "them into one DataFrame, coercing missing fields to NULL, dropping rows missing "
    "AnonID/Query/QueryTime, compressing adjacent duplicates by (AnonID, Query, "
    "ItemRank, ClickURL) while keeping only the last record in each run, typing "
    "columns, and sorting by QueryTime ascending."
)

COLUMNS = ["AnonID", "Query", "QueryTime", "ItemRank", "ClickURL"]
DATA_GLOB = "user-ct-test-collection*.txt"
data_path = Path("paul/GitHub/my-code/aol-data")


def resolve_data_path() -> Path:
    """Resolve the configured dataset directory with practical fallbacks."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        data_path,
        Path("C:/") / data_path.as_posix().lstrip("/"),
        repo_root,
    ]

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob(DATA_GLOB)):
            return candidate

    raise FileNotFoundError(
        f"Could not find files matching {DATA_GLOB!r} in: "
        + ", ".join(str(c) for c in candidates)
    )


def load_source_files(source_dir: Path) -> pd.DataFrame:
    """Read all matching source text files and concatenate into one DataFrame."""
    files = sorted(source_dir.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {DATA_GLOB!r} in {source_dir}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        print(f"Reading {file_path.name}")
        frame = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=COLUMNS,
            dtype="string",
            keep_default_na=True,
            engine="python",
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def compress_adjacent_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Collapse adjacent rows that share (AnonID, Query, ItemRank, ClickURL), keeping
    only the last row of each adjacent run. Missing values are treated as equal.
    """
    if df.empty:
        return df.copy(), 0, 0

    key_cols = ["AnonID", "Query", "ItemRank", "ClickURL"]
    same_as_prev = pd.Series(True, index=df.index)
    same_as_next = pd.Series(True, index=df.index)

    for col in key_cols:
        series = df[col]
        prev = series.shift(1)
        nxt = series.shift(-1)

        eq_prev = series.eq(prev).fillna(False) | (series.isna() & prev.isna())
        eq_next = series.eq(nxt).fillna(False) | (series.isna() & nxt.isna())

        same_as_prev &= eq_prev
        same_as_next &= eq_next

    same_as_prev.iloc[0] = False
    same_as_next.iloc[-1] = False

    # Avoid cumsum on Arrow-backed boolean dtype (unsupported in some pyarrow setups).
    group_start_flags = (~same_as_prev).to_numpy(dtype=np.int64)
    group_ids = pd.Series(np.cumsum(group_start_flags), index=df.index)
    group_sizes = group_ids.groupby(group_ids).size()
    compressed_groups_mask = group_sizes >= 2
    compressed_groups = int(compressed_groups_mask.sum())
    dropped_records = int((group_sizes[compressed_groups_mask] - 1).sum())

    keep_mask = ~same_as_next
    compressed = df.loc[keep_mask].copy()
    return compressed, compressed_groups, dropped_records


def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Apply de-duplication, typing, null handling, filtering, and sorting."""
    df, compressed_groups, dropped_records = compress_adjacent_records(df)

    df["AnonID"] = pd.to_numeric(df["AnonID"], errors="coerce").astype("Int64")
    df["ItemRank"] = pd.to_numeric(df["ItemRank"], errors="coerce").astype("Int64")
    df["QueryTime"] = pd.to_datetime(
        df["QueryTime"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
    )
    df["Query"] = df["Query"].astype("string")
    df["ClickURL"] = df["ClickURL"].astype("string")

    required_mask = (
        df["AnonID"].notna()
        & df["QueryTime"].notna()
        & df["Query"].notna()
        & (df["Query"].str.strip().str.len() > 0)
    )
    df = df.loc[required_mask, COLUMNS].copy()

    df = df.sort_values("QueryTime", ascending=True, kind="mergesort").reset_index(drop=True)
    return df, compressed_groups, dropped_records


def main() -> None:
    print(DESCRIPTION)
    source_dir = resolve_data_path()
    print(f"Using source directory: {source_dir}")

    all_rows = load_source_files(source_dir)
    cleaned, compressed_groups, dropped_records = transform(all_rows)

    out_path = Path(__file__).resolve().parent / "raw.parquet"
    cleaned.to_parquet(out_path, index=False)
    print(f"Compressed adjacent duplicate groups: {compressed_groups:,}")
    print(f"Dropped records from compression: {dropped_records:,}")
    print(f"Wrote {len(cleaned):,} rows to {out_path}")


if __name__ == "__main__":
    main()
