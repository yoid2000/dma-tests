# DMA Data Pipeline

This directory contains two scripts:

- `gather.py`: builds `raw.parquet` from AOL source text files.
- `label.py`: reads `raw.parquet`, labels query spans, and writes `labeled.parquet`.

# `gather.py`

`gather.py` reads files matching:

- `user-ct-test-collection*.txt`

Expected tab-separated input columns:

- `AnonID`
- `Query`
- `QueryTime`
- `ItemRank`
- `ClickURL`

Processing steps:

1. Coerce missing fields to nulls.
2. Compress adjacent duplicate runs by (`AnonID`, `Query`, `ItemRank`, `ClickURL`) and keep only the last row in each run.
3. Drop rows missing `AnonID`, `Query`, or `QueryTime`.
4. Type columns as int, text, datetime, int, text.
5. Sort by `QueryTime` ascending.

Script metrics printed at runtime:

- number of adjacent duplicate groups compressed
- number of records dropped by that compression
- final rows written

# `raw.parquet`

`raw.parquet` is the cleaned, typed, de-duplicated, and time-sorted dataset built from all source text files.

# `label.py`

`label.py` is designed for distributed labeling in three modes:

1. `python label.py make_distinct`
2. `python label.py sample [N]`
3. `python label.py <i>` where `i` is a 0-based chunk index
4. `python label.py create`

Mode details:

1. `make_distinct`: reads `raw.parquet` and writes `distinct_queries.parquet`.
2. `sample`: labels a random sample from all entries in `distinct_queries.parquet`, writes `samples.parquet`, and prints highest 10 and lowest 10 scored spans per label (including query text and span offsets). The optional second positional integer `N` sets the sample size (default `1000`).
3. `<i>`: labels the `i`th chunk of distinct queries, where chunks are floor-partitioned across `num_chunks` (default 200), writes `label_work/<i>.parquet`, and prints highest 10 and lowest 10 scored spans per label for that chunk.
4. `create`: reads all `label_work/0.parquet` through `label_work/199.parquet`, joins labels back to `raw.parquet`, and writes `labeled.parquet`.

Label types:

- `full_name`
- `street_city_address`
- `place_name`
- `profession`
- `disease`
- `crime`
- `finance`
- `social_security_number`
- `email_address`
- `credit_card_number`
- `phone_number`

Labeling approach:

1. Use `GLiNER` to extract span candidates for the label set.
2. Post-validate `full_name` spans with `probablepeople`:
   - require model confidence >= `--full-name-threshold` (default `0.8`)
   - require plausible name text (at least first + last; reject url/email/alphanumeric patterns)
   - require `probablepeople` to classify as `Person` with both `GivenName` and `Surname`
3. Post-validate `email_address` spans with `email-validator`:
   - require `@` in the span
   - reject domain-only values like `example.com`
4. Post-validate `credit_card_number` spans with `python-stdnum` (`stdnum.luhn`):
   - require 13-19 digits (spaces/hyphens allowed in source text)
   - require Luhn checksum validity
5. Post-validate `phone_number` spans with `phonenumbers`:
   - reject alphabetic/alphanumeric strings
   - require 10-15 digits and valid parse
   - normalize kept values to E.164 format
6. Post-validate location spans with `usaddress`:
   - `street_city_address` requires at least street + city components.
   - location spans without street components are labeled `place_name`.
   - street-only spans (street present, city missing) are dropped.

`QueryLabels` format:

- list of dictionaries per row
- each dictionary has: `label`, `text`, `start`, `end`, `score`
- `start` and `end` are 0-based character offsets into `Query` (`end` is exclusive)

`label.py` supports CLI options for input path, output path, distinct path, label work directory, number of chunks, model id, threshold, full-name threshold, batch size, and sample output path.

# `run.sbatch`

`run.sbatch` submits a Slurm job array with 200 concurrent tasks:

- array indices: `0-199`
- each task runs: `python label.py $SLURM_ARRAY_TASK_ID`

Typical workflow:

1. Generate distinct queries once: `python label.py make_distinct`
2. Submit array labeling: `sbatch run.sbatch`
3. After array completion, create final output: `python label.py create`

# `labeled.parquet`

`labeled.parquet` contains all columns from `raw.parquet` plus:

- `QueryLabels`: extracted labels for the row's `Query`
