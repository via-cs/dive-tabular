Refinement round $refinement_round of $max_refinement_rounds for discovery phase $phase.

Dataset description:
$dataset_description

Metadata for the currently involved columns:
$columns_json

Random real-data sample of the involved columns ($sample_rows rows, CSV):
```csv
$sample_csv
```

Candidate and complete verification history, including violating samples from every previous attempt:
$candidate_history_json

Submit exactly one revision with the same constraint ID. If no principled revision remains, finish with rejected_hypotheses listing that ID and a concrete reason.
