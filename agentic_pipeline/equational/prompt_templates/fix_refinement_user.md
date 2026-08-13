Refinement $refinement_round of $max_refinements for target `$target_column` in constraint `$constraint_id`.

The previous proposal failed full-data verification:
$verification_json

Each counterexample contains the original values of every involved column plus `_proposed_$target_column`, the value calculated by the failed fix code. The target remains unavailable to the executable fix function.

Return one principled corrected `fix(df)` or code=null if no uniform reconstruction is achievable. Do not encode row-specific exceptions or relax the frozen check.
