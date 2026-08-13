Discovery phase $phase of at most $max_phases. Discover at most $max_constraints new equational constraints for this dataset. This phase uses a fresh sample that does not overlap earlier discovery samples.

Dataset description:
$dataset_description

Numerical columns and descriptions:
$columns_json

Representative real-data sample ($sample_rows rows, CSV):
~~~csv
$sample_csv
~~~

Already accepted hypotheses; do not repeat or algebraically rearrange them:
$accepted_json

Previously rejected hypotheses; do not repeat or revise them in this discovery phase:
$rejected_json

Be as exhaustive as the evidence supports while avoiding speculative formulas. Submit all new candidates in one verifier call. The host stores accepted tool submissions exactly, so the final structured response contains only abandoned hypothesis IDs and reasons.
