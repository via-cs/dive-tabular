# Reproducible multi-run experiments

`multirun_experiment.py` coordinates three frozen train/test splits, agentic
constraint discovery, four synthesizers, constraint postprocessing, evaluation,
and aggregation. Run commands from the repository root with the `uv` managed
environment.

## Dataset inputs

The dataset directory should contain:

```text
dataset/<name>/
├── data.csv
├── info.json
├── utility_feature.json
└── meta.json
```

`utility_feature.json` supplies the downstream target and features. `info.json`
supplies categorical and numerical types. All three agentic pipelines use
`meta.json` for the dataset and column descriptions.

All directory-based inputs also have exact-file overrides: `--data-file`,
`--info-file`, `--utility-feature-file`, and `--meta-file`.

## Prepare and configure

Create the default three 70/30 splits:

```bash
uv run python multirun_experiment.py prepare \
  --dataset-dir dataset/taxi \
  --output-dir experiments_multirun/taxi
```

The default split seeds are `42`, `43`, and `44`. Classification targets are
stratified when class counts and split sizes permit it. Regression targets use
seeded random splitting. Every split manifest records whether stratification
was used and, if not, why.

Preparation also writes `experiment_config.json`. Review it before running
costly stages. It contains explicit configurations for:

- the categorical-dependency discovery model;
- the equational discovery and fix-generation model;
- the linear discovery model;
- CTGAN;
- TVAE;
- Gaussian Copula; and
- TabDDPM.

Every published linear constraint has passed full-training-data verification
and has a distinct canonical half-space geometry. After all three discovery
pipelines finish, a linear constraint is removed when all of its columns already
belong to the union of columns used by accepted equational constraints.

The default generation design trains each enabled generator once on each split
and samples it three times with seeds `1000`, `1001`, and `1002`. Each sample
has the same row count as its frozen training split.

Custom split seeds and an explicit split ratio may be supplied during initial
preparation:

```bash
uv run python multirun_experiment.py prepare \
  --dataset-dir dataset/taxi \
  --output-dir experiments_multirun/taxi \
  --test-size 0.3 \
  --split-seeds 42 43 44
```

## Run individual stages

Run constraint discovery on each training split:

```bash
uv run python multirun_experiment.py discover \
  --experiment-dir experiments_multirun/taxi
```

The same ordered generator can be run directly:

```bash
uv run python -m agentic_pipeline.generate_constraints \
  dataset/taxi \
  dataset/taxi \
  --output-dir agentic_constraints/taxi
```

This invokes `agentic_pipeline.generate_constraints` once per split. It runs
categorical, equational, and linear discovery in that order. Equational fix
generation is required because its executable fixes are used during
postprocessing. The final linear file is filtered against equational-column
coverage before the discovery command is marked complete. To
validate all inputs and write model-request previews without API calls:

```bash
uv run python multirun_experiment.py discover \
  --experiment-dir experiments_multirun/taxi \
  --dry-run
```

Train and sample every enabled generator:

```bash
uv run python multirun_experiment.py generate \
  --experiment-dir experiments_multirun/taxi
```

Apply each split's proposed constraints to its generated files:

```bash
uv run python multirun_experiment.py postprocess \
  --experiment-dir experiments_multirun/taxi
```

Evaluate all raw and constrained files against those same proposed constraints,
then aggregate file, within-split, across-split, and overall results:

```bash
uv run python multirun_experiment.py evaluate \
  --experiment-dir experiments_multirun/taxi
```

Each split stores `real_metrics_cache.json`. The first unfinished evaluation
for that split computes TRTR and the real-training constraint metrics; all
generator and raw/constrained evaluations then reuse those values. Synthetic
quality, utility, and constraint metrics remain specific to each generated
file. The cache is automatically invalidated when the train/test data, utility
metadata, configured real metrics, or proposed constraint files change.

Multi-run utility evaluation reports TRTR as the real-data baseline and TSTR
for each synthetic file. TSRTR is neither computed nor included in multi-run
evaluation JSON or aggregate summaries. The underlying evaluation CLI remains
configurable through `--utility-regimes` for experiments outside multirun.

Equational postprocessing defaults to the static global planner via
`postprocessing.equational_strategy: static-global`. Set it to
`dynamic-greedy` only for legacy comparisons. The global planner maximizes the
worst predicted per-column KSComplement delta, then the total predicted delta.

Both stages support artifact-based continuation after an interrupted run:

```bash
uv run python multirun_experiment.py postprocess \
  --experiment-dir experiments_multirun/taxi \
  --continue

uv run python multirun_experiment.py evaluate \
  --experiment-dir experiments_multirun/taxi \
  --continue
```

For postprocessing, `--continue` skips a generator variant when all expected
`fix_report/synthetic_<index>.json` files are present. For evaluation, it skips
a raw or constrained variant only when `evaluation.json` is valid, contains one
result for every configured sample and every configured metric, and, when the
constraint metric is enabled, `constraint_evaluation_details.json` is also
valid. Unlike normal manifest-based resume, this artifact check intentionally
ignores a changed configuration fingerprint.

After configuration has been reviewed, all stages can be run together:

```bash
uv run python multirun_experiment.py all \
  --experiment-dir experiments_multirun/taxi
```

## Output layout

```text
experiments_multirun/taxi/
├── experiment_config.json
├── evaluation_summary.json
└── splits/
    ├── split_00_seed_42/
    │   ├── split_manifest.json
    │   ├── train.csv
    │   ├── test.csv
    │   ├── real_metrics_cache.json
    │   ├── constraints/
    │   │   ├── categorical_dependency_constraint.json
    │   │   ├── equational_constraint.json
    │   │   ├── linear_constraint.json
    │   │   ├── run_report_all_constraints.json
    │   │   ├── run_report_categorical.json
    │   │   ├── run_report_equational.json
    │   │   └── run_report_linear.json
    │   └── generators/
    │       ├── ctgan/
    │       │   ├── raw/
    │       │   │   ├── train.csv
    │       │   │   ├── test.csv
    │       │   │   ├── run_manifest.json
    │       │   │   ├── evaluation.json
    │       │   │   └── synthetic/
    │       │   │       ├── synthetic_0.csv
    │       │   │       ├── synthetic_1.csv
    │       │   │       └── synthetic_2.csv
    │       │   └── constrained/
    │       │       ├── evaluation.json
    │       │       ├── fix_report/
    │       │       └── synthetic/
    │       ├── tvae/
    │       ├── gaussian_copula/
    │       └── tabddpm/
    ├── split_01_seed_43/
    └── split_02_seed_44/
```

## Reproducibility and resuming

Split manifests contain source, train, and test SHA-256 hashes. Every expensive
command has its own command manifest containing the resolved argument vector and
an input/configuration fingerprint. A completed stage is skipped only when its
manifest fingerprint still matches and its expected outputs exist.

If a source or frozen split hash changes, later stages stop instead of silently
using mixed artifacts. Use a stage's `--force` flag only when intentionally
replacing outputs for that exact experiment.

The final `evaluation_summary.json` retains every file-level record and reports:

- mean and sample standard deviation across the three samples within a split;
- mean and sample standard deviation across the three split means; and
- an overall summary across all nine files for each generator and variant.

Raw and constrained variants are always reported separately. Constraint metrics
use the proposed split-specific constraints, not dataset expert constraints.
