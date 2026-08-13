Discovery phase $phase of at most $max_phases. Discover at most $max_constraints new linear inequality constraints. This phase uses a fresh sample that does not overlap earlier discovery samples.

Dataset description:
$dataset_description

Numerical columns selected strictly from info.json entries whose type is `num`:
$columns_json

Representative real-data sample ($sample_rows rows, CSV):
~~~csv
$sample_csv
~~~

Already accepted hypotheses; do not repeat or weaken them:
$accepted_json

Previously rejected hypotheses; do not repeat them in this discovery phase:
$rejected_json

Be as exhaustive as the semantic evidence supports. Submit all new candidates in one verifier call. The host stores accepted tool submissions exactly, so the final structured response contains only deliberately abandoned hypothesis IDs and reasons.
