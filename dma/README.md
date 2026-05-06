# DMA Data Pipeline

This directory contains nine primary scripts:

- `gather.py`: builds `raw.parquet` from AOL source text files.
- `label.py`: reads `raw.parquet`, labels query spans, and writes `labeled.parquet`.
- `linkage_prepare.py`: builds user splits and sampled 50-user group manifests for linkage experiments.
- `linkage_build_pairs.py`: builds positive and hard-negative query pairs for pairwise linkage model training.
- `linkage_train_model.py`: trains and calibrates a pairwise same-user probability model.
- `linkage_score_pairs.py`: scores pair rows with calibrated same-user probabilities.
- `linkage_pair_eval.py`: scores all query pairs within each group, sweeps thresholds, and reports precision/recall/F1.
- `linkage_run_all.py`: orchestrates the full 4-step linkage pipeline in one command.

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

Parameters:

- `mode`: operation to run — `make_distinct`, `sample`, `create`, or an integer chunk index.
- `sample_size` (default `1000`): number of rows to sample; only used in `mode=sample`.
- `--num-chunks` (default `200`): total number of chunks used for distributed labeling.
- `--model-id` (default `gliner-community/gliner_medium-v2.5`): GLiNER model identifier passed to `from_pretrained()`.
- `--threshold` (default `0.5`): minimum GLiNER span confidence to keep a label.
- `--full-name-threshold` (default `0.8`): minimum GLiNER confidence required before the `full_name` post-validation step runs.
- `--batch-size` (default `128`): number of queries sent to GLiNER per inference batch.

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

# Linkage Prep And Pair Building

The linkage pipeline currently has two implemented preparation steps.

## `linkage_prepare.py`

Purpose:

- build `AnonID` splits for linkage experiments by user (not by row)
- sample repeated 50-user evaluation groups for validation/test scenarios

Typical command:

- `python linkage_prepare.py`

Default outputs under `linkage_work/`:

- `train_users.parquet`
- `val_users.parquet`
- `test_users.parquet`
- `group_manifest.parquet`
- `split_summary.parquet`

Parameters:

- `--min-queries-per-user` (default `20`): exclude users with fewer queries than this.
- `--max-queries-per-user` (default `5000`): exclude users with more queries than this.
- `--train-users` (default `5000`): number of users assigned to the training split.
- `--val-users` (default `25000`): number of users assigned to the validation split.
- `--test-users` (default `5000`): number of users assigned to the test split.
- `--group-size` (default `50`): number of users per evaluation group.
- `--val-groups` (default `500`): number of sampled validation groups written to `group_manifest.parquet`.
- `--test-groups` (default `100`): number of sampled test groups.
- `--seed` (default `13`): random seed for all sampling.

Notes:

- train/val/test user splits are non-overlapping.

## `linkage_build_pairs.py`

Purpose:

- load training users from `linkage_work/train_users.parquet`
- pull matching rows from `raw.parquet`
- sample positive pairs (same `AnonID`)
- sample hard negatives (different `AnonID`, preferring same day with token overlap)
- write feature-engineered pair dataset for model training

Typical command:

- `python linkage_build_pairs.py --out-dir linkage_work --positive-pairs-per-user 200 --negative-to-positive-ratio 2 --max-queries-per-user 300 --hard-negative-try-count 24`

Parameters:

- `--max-queries-per-user` (default `300`): cap on queries loaded per user to keep pair generation bounded.
- `--positive-pairs-per-user` (default `200`): maximum number of same-user (positive) pairs sampled per user.
- `--negative-to-positive-ratio` (default `2.0`): number of hard-negative pairs to sample per positive pair.
- `--hard-negative-try-count` (default `24`): number of same-day candidate queries probed when building each hard negative.
- `--seed` (default `13`): random seed.

Outputs:

- `pair_queries.parquet`: sampled query rows with `row_id`
- `train_pairs.parquet`: pair labels and engineered features

Main features in `train_pairs.parquet`:

- temporal: `same_day`, `day_gap`
- text shape: lengths, prefix equality, digit flags
- lexical overlap: token overlap/union/jaccard
- exact normalized query match flag
- click-url signals: `has_url_a`, `has_url_b`, `both_have_url`, `url_exact_match`, `url_domain_match`

# Linkage Model Training And Scoring

This pipeline learns a supervised pairwise linkage model that estimates the probability that two queries come from the same anonymized user. It combines lexical overlap, temporal proximity, query-shape, click-URL, and label signals in a gradient-boosted classifier, then calibrates the output with Platt scaling so scores are more interpretable as probabilities. In operation, the trained model is applied to large sets of query pairs within sampled user groups, and score thresholds are swept to expose precision-recall trade-offs for different matching objectives.

## `linkage_run_all.py`

Purpose:

- run the full 4-step linkage flow in order (`prepare -> build_pairs -> train_model -> pair_eval`)
- stop immediately if any step fails
- optionally pass per-step CLI arguments and resume from a chosen step

Typical command:

- `python linkage_run_all.py`

Resume from step 4 example:

- `python linkage_run_all.py --start-step 4 --step-4-args "--split val --max-groups 20 --max-queries-per-user 100 --top-n 20"`

Parameters:

- `--start-step` (default `1`): step number to begin from; steps 1–4 map to `prepare`, `build_pairs`, `train_model`, `pair_eval`. Useful for resuming after a failure.
- `--stop-step` (default `4`): step number to stop after (inclusive).
- `--step-N-args` (for N in 1–4): extra CLI arguments forwarded verbatim to step N, supplied as a single quoted string.

## `linkage_train_model.py`

Purpose:

- train a pairwise classifier on `train_pairs.parquet`
- calibrate predicted probabilities on a holdout split
- write reusable model artifacts and metrics

Typical command:

- `python linkage_train_model.py --pairs-path linkage_work/train_pairs.parquet --out-dir linkage_work/model`

Parameters:

- `--calib-user-fraction` (default `0.2`): fraction of users held out from training and used to fit the sigmoid (Platt) calibrator.
- `--max-iter` (default `300`): maximum boosting iterations for `HistGradientBoostingClassifier`.
- `--seed` (default `13`): random seed.

Outputs:

- `linkage_pair_model.pkl`: base model + sigmoid (Platt) calibrator + feature list
- `linkage_pair_metrics.json`: calibration-holdout metrics (`roc_auc`, `pr_auc`, `brier`)

## `linkage_score_pairs.py`

Purpose:

- load a trained `linkage_pair_model.pkl`
- score pair rows and output calibrated same-user probabilities

Typical command:

- `python linkage_score_pairs.py --pairs-path linkage_work/train_pairs.parquet --model-path linkage_work/model/linkage_pair_model.pkl --output-path linkage_work/model/train_pairs_scored.parquet`

Output columns include:

- `proba_base`
- `proba_calibrated`

These calibrated probabilities are the confidence values used by downstream clustering/assignment.

# Pairwise Evaluation

## `linkage_pair_eval.py`

Purpose:

- load sampled 50-user groups from `group_manifest.parquet`
- score every query pair within each group using `linkage_pair_model.pkl`
- make a binary same-user / different-user decision by thresholding the calibrated pair score
- sweep thresholds and report precision/recall/F1/pair-coverage
- print and save the top-N highest-scoring pairs

Key terms:

- **precision**: fraction of predicted same-user pairs (score ≥ threshold) that are truly from the same user.
- **recall**: fraction of all true same-user pairs across all groups that are captured at the chosen threshold.
- **F1**: harmonic mean of precision and recall.
- **pair_coverage**: fraction of all scored pairs that are predicted positive (score ≥ threshold).

Typical command:

- `python linkage_pair_eval.py --split val --max-groups 20 --max-queries-per-user 100 --sweep-steps 30 --top-n 20`

Parameters:

- `--split` (default `val`): which manifest split to run — `val`, `test`, or `all`.
- `--max-groups` (default `0` = all): cap on groups to process; `0` runs every group in the split.
- `--max-queries-per-user` (default `100`): cap on queries loaded per user per group.
- `--min-save-score` (default `0.3`): only checkpoint pairs with score ≥ this value; lower values improve low-threshold recall estimates at the cost of more disk usage.
- `--sweep-steps` (default `30`): number of evenly-spaced threshold steps in the precision/recall sweep, from `--min-save-score` to `1.0`.
- `--threshold` (optional): score threshold for the final result. If omitted, the threshold with the best F1 is chosen automatically.
- `--top-n` (default `20`): number of highest-scoring pairs to print and save.
- `--merge-only`: skip scoring; merge existing per-group checkpoints and recompute the sweep.

Checkpoint and resume behavior:

- each processed group is flushed immediately under `out-dir/groups/` as:
   - `<split>_<group_id>_scored.parquet`: pairs above `--min-save-score` with score and ground-truth label
   - `<split>_<group_id>_top.parquet`: top-500 pairs with query text for display
   - `<split>_<group_id>_meta.json`: total pair count and total same-user pair count for the group
- rerunning the same command skips already checkpointed groups
- use `--merge-only` to recompute the sweep from existing checkpoints without re-scoring

Outputs:

- `pair_eval_sweep.parquet` / `pair_eval_sweep.json`: precision/recall/F1/coverage at each threshold step
- `pair_eval_top_pairs.parquet`: top-N pairs by score with query text and ground-truth label
- `pair_eval_result.json`: metrics at the chosen (or best-F1) threshold
- `groups/`: per-group checkpoint files written incrementally

Threshold guidance:

- sort `pair_eval_sweep.parquet` by `precision` descending and choose the highest `pair_coverage` row that still meets your precision target (for example `>= 0.95`).
- use `--threshold` to re-emit `pair_eval_result.json` at a specific value without re-scoring: combine with `--merge-only`.
