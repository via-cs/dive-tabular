$common_rules

This invocation is the discovery stage. Search broadly for distinct semantic relationships, but propose only simple equations supported by the metadata and sample. Consider different plausible derived quantities and accounting identities; do not emit algebraic rearrangements or tolerance variants of the same idea. Do not repeat any accepted or rejected hypothesis listed in the user prompt.

Avoid proposing multiple constraints with the same unordered column set. If verified duplicates nevertheless accumulate across discovery phases, the host deterministically keeps one preferred rule per exact column set during final consolidation.

Submit every candidate together in exactly one verifier tool call. This stage only proposes and performs the first verification; do not revise failures yet. After the tool call, a separate refinement prompt will handle failures. If there are no defensible new candidates, finish with rejected_hypotheses=[] without calling the tool.

## Output examples

The following fictional examples demonstrate the expected verifier tool-call arguments. They illustrate structure and reasoning only. Never copy their IDs, columns, or equations into a real dataset unless that dataset's own metadata independently establishes the same relationship. Each discovery phase still uses exactly one verifier call containing all candidates for that phase.

### Example output 1:

Suppose a fictional warehouse dataset defines `opening_units`, `received_units`, `shipped_units`, and `closing_units` as exact item counts. A valid `verify_equational_constraints` call would use:

```json
{
  "constraints": [
    {
      "id": "closing_inventory_balance",
      "description": "Closing inventory equals opening inventory plus received units minus shipped units.",
      "rationale": "The four columns are exact counts from one inventory balance, so inflows add to stock and outflows subtract from it.",
      "columns": [
        "opening_units",
        "received_units",
        "shipped_units",
        "closing_units"
      ],
      "check_code": "def check(df):\n    expected = df[\"opening_units\"] + df[\"received_units\"] - df[\"shipped_units\"]\n    return df[\"closing_units\"] == expected"
    }
  ]
}
```
### Example output 2:

Suppose a fictional sensor dataset explicitly defines `distance_m` and `distance_km` as the same measurement stored in metres and kilometres. A valid `verify_equational_constraints` call would use:

```json
{
  "constraints": [
    {
      "id": "distance_unit_conversion",
      "description": "Distance in metres equals 1,000 times distance in kilometres.",
      "rationale": "The metadata identifies the columns as two unit representations of the same measurement, so the conversion factor is fixed by definition.",
      "columns": [
        "distance_m",
        "distance_km"
      ],
      "check_code": "def check(df):\n    expected_metres = 1000.0 * df[\"distance_km\"]\n    return (df[\"distance_m\"] - expected_metres).abs() <= 1e-6"
    }
  ]
}
```

After the verifier call, accepted constraints are already stored by the host. The phase-completion response must therefore contain only deliberately abandoned candidates, for example `{"rejected_hypotheses":[]}` when none are abandoned.
