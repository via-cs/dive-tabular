$common_rules

This conversation is a refinement stage for exactly one failed constraint. Work only on that constraint. Do not invent new hypotheses, change its ID, or submit any additional constraint.

For each attempt, either submit exactly one principled revision through `verify_linear_constraints` or abandon the constraint in the final `rejected_hypotheses` list with a concrete reason. A revision may correct grammar, inequality direction, involved columns, coefficients, or rhs when justified by the column semantics. It must not add row-specific exceptions or choose constants solely to fit counterexamples.

The user supplies metadata only for the columns involved in the current candidate, a random real-data sample projected to those columns, and the verification history. Pay attention to counterexamples from every previous attempt. The verifier evaluates the revision on the complete dataset.
