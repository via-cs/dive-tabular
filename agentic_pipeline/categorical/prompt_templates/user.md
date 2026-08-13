Discovery run $run of $runs. Discover at most $max_constraints categorical dependency constraints. The authoritative acceptance rule is `support > 0` and `violation_rate < $violation_threshold`.

Dataset description:
$dataset_description

Categorical columns selected strictly from info.json entries whose type is `cat`:
$columns_json

Representative sample ($sample_rows rows, CSV):
~~~csv
$sample_csv
~~~

Inspect and analyze any promising relationships. Act on each tool result before deciding whether to drill down, submit, revise, or abandon a hypothesis. Finish with the structured completion when no new semantically defensible constraints remain.
