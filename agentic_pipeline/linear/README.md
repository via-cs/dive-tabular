# Linear constraint discovery

This package discovers row-wise linear inequalities in canonical form
`sum(a_j * x_j) >= rhs`. The model sees descriptions of columns marked `num`
in `info.json` and a representative sample, then submits typed candidates to a
deterministic full-data verifier.

Candidates with violation rate at most `0.0005` are frozen. Each failure gets an
independent refinement conversation for up to three rounds. The refiner sees
only the involved columns’ descriptions, 100 stable random rows projected to
those columns, and the candidate’s accumulated verification history, including
all previously sampled counterexamples. Every verifier-accepted constraint
has a distinct canonical geometry and is published directly. The host derives
formula, columns, sense, and `mutable_columns`; every involved column is
mutable.

Preview NEWS requests without an API call:

~~~bash
uv run python -m agentic_pipeline.linear \
  dataset/news/meta.json dataset/news/data.csv \
  --output-dir agentic_constraints/news-linear \
  --dry-run
~~~

Remove `--dry-run` to run discovery with `OPENAI_API_KEY` set. For Claude
Sonnet 5, set `ANTHROPIC_API_KEY` and pass
`--provider anthropic --model claude-sonnet-5`. Both positional arguments may
instead be directories containing `meta.json` and `data.csv`; the metadata
directory must also contain `info.json`. See
[`../MODEL_PROVIDERS.md`](../MODEL_PROVIDERS.md) for provider behavior and full
examples.

Outputs are `linear_constraint.json`, `run_report.json`, or
`request_preview_linear.json` for a dry run.
