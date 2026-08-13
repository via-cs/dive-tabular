# Equational constraint discovery

This directory contains a deliberately small agentic pipeline. One selected model
receives the dataset description, descriptions of numerical columns, and a
sample of 100 numerical rows. It may call one local tool that runs proposed
check code against the complete CSV and returns violation rates and up to 20
sampled counterexamples.

By default, the pipeline runs up to three discovery phases. Each phase receives
a new, non-overlapping 100-row sample plus concise summaries of previously
accepted and rejected hypotheses. A discovery invocation proposes new ideas and
performs their first verifier call. Each failed candidate gets an independent refinement conversation for at most
three rounds. That conversation persists across its attempts and contains only
the involved-column descriptions, 100 stable random rows projected to those
columns, and the accumulated verification history, including all sampled
counterexamples. Accepted constraints and other failed candidates are omitted.

The unordered set of a rule's columns is its equation-family signature. The
discovery prompts discourage repeated signatures, but every distinct candidate
is verified and accepted independently. This allows duplicates accumulated
across discovery phases to be handled uniformly at the end.

After discovery, consolidation is fully deterministic and has two stages.
First, exact unordered column-set duplicates are reduced to the candidate with
the lowest violation rate, then the simplest and earliest accepted candidate
when quality ties. Second, a binary optimization selects a maximum-cardinality
subset in which every published rule has at least one column used by no other
published rule. Equal-cardinality optima use the same stable quality ordering.
The solver is `scipy.optimize.milp` backed by HiGHS; no model call or
consolidation prompt is used. Removed overlap rules are not described as
logically subsumed because column overlap alone does not prove derivability.

After successful consolidation, a fix-generation role independently attempts
to reconstruct every involved column of every retained constraint. For a target
column, its initial 100-row sample contains only the other involved columns.
The host gives generated `fix(df)` code a source-only dataframe, replaces the
target in a private full-data copy, and runs the frozen `check(df)`. A fix is
accepted at a post-repair violation rate of at most 0.5%. A failed attempt can
be refined up to three times; its feedback contains up to 20 original involved
rows, including the original target, plus `_proposed_<target>`. An unavailable
path is published with `code: null`.

Run a dry request preview without an API call:

~~~bash
uv run python -m agentic_pipeline.equational \
  dataset/flights/meta.json dataset/flights/data.csv \
  --output-dir agentic_constraints/flights \
  --dry-run
~~~

Run discovery with `OPENAI_API_KEY` set (or present in `.env`):

~~~bash
uv run python -m agentic_pipeline.equational \
  dataset/flights/meta.json dataset/flights/data.csv \
  --output-dir agentic_constraints/flights
~~~

OpenAI and Anthropic are supported natively. For Claude Sonnet 5, set
`ANTHROPIC_API_KEY` and add `--provider anthropic --model claude-sonnet-5`.
See [`../MODEL_PROVIDERS.md`](../MODEL_PROVIDERS.md) for pipeline examples.

The same repair stage is also a standalone CLI for an existing constraint
artifact:

~~~bash
uv run python -m agentic_pipeline.equational.fix_generation \
  agentic_constraints/flights/equational_constraints.json \
  dataset/flights/data.csv
~~~

The standalone positional arguments can instead be directories containing
`equational_constraints.json` and `data.csv`. It finds `meta.json` beside the
data when available; `--meta` accepts an explicit metadata file or directory.
By default it atomically updates the input constraint file. `--output` accepts
an explicit output file or directory, and `--report` does the same for
`fix_generation_report.json`. Use `--dry-run` to validate every frozen check and
render all source-only request previews without an API call or constraint-file
mutation.

Both positional inputs also accept directories containing meta.json and
data.csv, respectively. When info.json is next to meta.json, its col_types
entries select the columns marked num; otherwise pandas numeric dtypes are used.
Descriptions always come from meta.json. The discovery model can be overridden
with `--model`, and `--fix-model` controls the repair role. Loop bounds are
controlled by `--max-discovery-phases`, `--max-refinement-rounds`, and
`--refinement-sample-rows` (default 100); repair options include
`--fix-sample-rows`, `--max-fix-refinements`, and
`--fix-violation-threshold`. Use `--skip-consolidation` to bypass both
deterministic consolidation stages and publish every verifier-accepted rule,
and use `--skip-fix-generation` when only executable checks are needed.

Outputs:

- equational_constraint.json: the published list augmented with rationale and
  an ordered fix_code entry for every involved column; unsuccessful paths use
  code: null.
- run_report_equational.json: configuration, sample counts, exact accepted
  submissions, rejected hypotheses, discovery/tool-call trace, deterministic
  deduplication and optimization decisions, and the complete per-target
  fix-generation trace.
- fix_generation_report.json: standalone repair configuration, model proposals,
  verifier results, counterexamples, and final path status. Dataset row indices
  are never included in reports or request previews.
- request_preview_equational.json: generated only by --dry-run.

Accepted submissions are frozen directly from verifier inputs and are never
rewritten during consolidation. Generated code is statically restricted and
executed in a timed subprocess. This is a defense-in-depth boundary, not a
container-grade security sandbox; run the pipeline only in an environment
where local dataset access is appropriate.

The implementation is split by responsibility: `dataset_io.py` handles file
resolution and atomic JSON output; `verifier.py` verifies checks;
`fix_verifier.py` verifies repairs; `fix_generation.py` owns the reusable repair
loop and standalone CLI; `consolidation.py` owns deterministic deduplication and
binary optimization; `models.py` owns structured schemas; and `prompting.py`
only renders files under `prompt_templates/`. Discovery, refinement, and repair
prompt text can therefore be edited without changing Python source.

Live runs send the configured row samples to the selected provider and model. Initial
repair requests omit target values; repair refinement feedback includes the
original target and the failed proposed value as described above.
