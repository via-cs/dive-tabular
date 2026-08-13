# Agentic categorical-dependency discovery

This pipeline discovers logical dependencies among columns marked `cat` in the
dataset's `info.json`. Each constraint has one or more determinant columns,
exactly one dependent column, and one unified set-valued correspondence table.
There is no `constraint_type` field.

The proposer runs in a persistent native tool-calling conversation. It may
inspect categorical profiles, analyze a determinant/dependent signature, query
dependent frequencies for a Cartesian determinant configuration, and submit a
constraint. Submission requires a cited `analyze_dependency` result for the
same signature. Accepted submissions are stored by the host rather than copied
from the model's final response.

## Run

Both positional arguments accept either the specific JSON/CSV file or a
directory containing `meta.json`/`data.csv`:

```bash
uv run python -m agentic_pipeline.categorical \
  dataset/flights \
  dataset/flights \
  --runs 3 \
  --output-dir agentic_constraints/flights/categorical
```

Use `--dry-run` to validate inputs and write the prompts, tool schemas, and
completion schema without calling a model. The default acceptance rule is
`support > 0` and `violation_rate < 0.005`.

Multiple `--runs` share one accepted dependency graph. The first accepted
direction wins; analysis and submission reject any later direct or indirect edge
that would create a cycle. Consolidation repeats that check in input order, then
canonicalizes, deduplicates, prunes strict subsumption, merges compatible value
tables, and verifies every result before publication.

## DSL

```json
{
  "dsl_version": "2.0",
  "constraints": [
    {
      "id": "code_to_name",
      "description": "Each code determines its full name.",
      "rationale": "The fields are two representations of one entity.",
      "determinants": ["code"],
      "dependent": "name",
      "value_table": [
        {
          "determinant_values": [["A", "B"]],
          "dependent_values": ["Alpha-Beta Group"]
        }
      ],
      "support": 0.4,
      "violation_rate": 0.0
    }
  ]
}
```

Within one value-table row, each determinant list is an OR and the lists across
determinant columns form a Cartesian AND. Rows outside all determinant
configurations are not applicable. Applicable rows violate the constraint only
when their dependent value is absent from the matching admissible list.

For a large exact dependency, the proposer submits an empty value table. The
host materializes every observed exact determinant tuple with its majority
dependent value and then performs the authoritative full-data verification.

## Repair

The repair implementation is `constraints/categorical/fix.py`. It requires
DSL 2.0 and an acyclic dependency graph, processes dependent columns in stable
topological order, and intersects all applicable allowed sets for each row. It
samples invalid set-valued dependents from the reference conditional
distribution. A row is dropped when that intersection is empty.
