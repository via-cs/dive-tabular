$common_rules

This conversation refines exactly one failed equational constraint. Work only on that constraint. Do not invent new hypotheses, change its ID, or submit any additional constraint. For each attempt, either submit exactly one principled revision through the verifier or abandon the constraint in the final rejected_hypotheses list with a concrete reason.

A revision may fix code, correct a semantically justified formula, or use a justified numerical tolerance. Preserve the original semantic relationship. Do not fit arbitrary constants from samples or counterexamples. Do not add conditionals, clipping, rounding, enlarged tolerances, or row-specific exceptions unless directly justified by the column definitions.

The user supplies metadata only for the columns involved in the current candidate, a random real-data sample projected to those columns, and the complete verification history. Pay attention to counterexamples from every previous attempt. The verifier evaluates the revision on the complete dataset.

If no principled revision remains, finish immediately.
