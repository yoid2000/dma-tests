import os
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

raw_path = Path(r"c:/paul/GitHub/dma-tests/dma/raw.parquet")
db_path = Path(r"c:/paul/GitHub/dma-tests/dma/_tmp_pair_counts.db")
if db_path.exists():
    db_path.unlink()

conn = sqlite3.connect(db_path.as_posix())
cur = conn.cursor()
cur.execute("PRAGMA journal_mode = OFF")
cur.execute("PRAGMA synchronous = OFF")
cur.execute("PRAGMA temp_store = MEMORY")
cur.execute("PRAGMA cache_size = -100000")
cur.execute("CREATE TABLE grp (Query TEXT, day TEXT, ClickURL TEXT, AnonID INTEGER, n INTEGER)")
conn.commit()

scanner = ds.dataset(raw_path.as_posix(), format="parquet").scanner(
    columns=["AnonID", "Query", "QueryTime", "ClickURL"],
    batch_size=500_000,
)

rows_in = 0
rows_grouped = 0
for i, batch in enumerate(scanner.to_batches(), start=1):
    df = batch.to_pandas(types_mapper=pd.ArrowDtype)
    rows_in += len(df)

    df["AnonID"] = pd.to_numeric(df["AnonID"], errors="coerce").astype("Int64")
    df["Query"] = df["Query"].astype("string")
    df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")
    df["ClickURL"] = df["ClickURL"].astype("string").fillna("")

    df = df.dropna(subset=["AnonID", "Query", "QueryTime"]).copy()
    df = df[df["Query"].str.strip().str.len() > 0].copy()
    if df.empty:
        continue

    df["day"] = df["QueryTime"].dt.floor("D").astype("string")

    g = (
        df.groupby(["Query", "day", "ClickURL", "AnonID"], sort=False, dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    rows_grouped += len(g)

    g["Query"] = g["Query"].astype(str)
    g["day"] = g["day"].astype(str)
    g["ClickURL"] = g["ClickURL"].astype(str)
    g["AnonID"] = g["AnonID"].astype("int64")
    g["n"] = g["n"].astype("int64")

    g.to_sql("grp", conn, if_exists="append", index=False)

    if i % 20 == 0:
        conn.commit()
        print(f"processed batches={i}, rows_in={rows_in:,}, grouped_rows_written={rows_grouped:,}")

conn.commit()

cur.execute("""
CREATE TABLE agg AS
SELECT Query, day, ClickURL, AnonID, SUM(n) AS n
FROM grp
GROUP BY Query, day, ClickURL, AnonID
""")
conn.commit()

cur.execute("""
SELECT
  SUM(CASE WHEN max_n >= 2 THEN 1 ELSE 0 END) AS same_anon_key_count,
  SUM(CASE WHEN distinct_anon >= 2 THEN 1 ELSE 0 END) AS diff_anon_key_count,
  SUM(CASE WHEN total_n >= 2 THEN 1 ELSE 0 END) AS any_pair_key_count,
  SUM(CASE WHEN max_n >= 2 AND distinct_anon >= 2 THEN 1 ELSE 0 END) AS both_conditions_key_count,
  COUNT(*) AS total_unique_keys
FROM (
  SELECT
    Query,
    day,
    ClickURL,
    MAX(n) AS max_n,
    COUNT(*) AS distinct_anon,
    SUM(n) AS total_n
  FROM agg
  GROUP BY Query, day, ClickURL
)
""")
res = cur.fetchone()

print("\nRESULTS")
print(f"same_anon_key_count={res[0]:,}")
print(f"diff_anon_key_count={res[1]:,}")
print(f"any_pair_key_count={res[2]:,}")
print(f"both_conditions_key_count={res[3]:,}")
print(f"total_unique_keys={res[4]:,}")

conn.close()
try:
    os.remove(db_path)
except OSError:
    pass
