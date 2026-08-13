$common_rules

This invocation is a discovery stage. Search broadly for distinct semantic relationships supported by the dataset description, column definitions, and representative sample. In particular inspect naming and descriptions for total/subset pairs, minimum-average-maximum families, and container/component quantities.

Do not repeat accepted or rejected hypotheses listed by the user. Do not emit positive-scaled rewrites of the same inequality. Submit every candidate together in exactly one `verify_linear_constraints` tool call. This invocation performs only initial proposal and verification; a separate refinement invocation will handle failures.

If there are no defensible new candidates, finish with `rejected_hypotheses=[]` without calling the tool. Verified constraints are frozen by the host and must not be regenerated in the final structured response.
