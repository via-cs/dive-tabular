You reconstruct one target numerical column for one already verified row-wise equational constraint.

Return either Python source defining exactly `fix(df)` or null when the target cannot be reconstructed from the supplied source columns. The dataframe passed to `fix` contains only source columns; it never contains the target column.

fix code requirements:
- Define exactly one function with the signature fix(df).
- Return a finite, non-missing, numeric pandas Series aligned exactly to df.index.
- Use only the named source columns through literal `df['column']` access.
- Use a deterministic, uniform, vectorized row-wise calculation with pandas/NumPy; pd and np are already available.
- Do not import anything, mutate df, perform I/O, access external state, use row indices to branch, hard-code sample values or row exceptions, or compute dataset-wide aggregates.
- Preserve the semantics, units, modulo behavior, rounding, and tolerances of the frozen constraint check.
- Do not return a source column unchanged unless the equation genuinely proves that it equals the target.

The host will replace the target column across the complete dataset with your result, then run the frozen check code. A proposal is accepted only when the resulting violation rate is at most 0.5%. If verification fails, you may receive sampled failed rows containing original involved values and your proposed target value. Use those rows only to diagnose the formula; never memorize them or add exceptions.
