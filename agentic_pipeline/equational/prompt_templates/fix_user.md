Write a repair for target column `$target_column` using only the other involved columns.

Frozen constraint:
$constraint_json

Target description:
$target_description

Available source columns and descriptions:
$source_descriptions_json

Representative source-only sample ($sample_rows rows, CSV). The target column is intentionally absent:
~~~csv
$sample_csv
~~~

Return code for this target only. If the equation cannot be inverted unambiguously from these source columns, return code=null and explain why.
