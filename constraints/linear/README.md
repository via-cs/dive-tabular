# Linear convex postprocessing

This package implements the post-hoc projection used by CLAMP in Afonja et al.
(2026). For each violating generated row `x`, it solves

```text
minimize_z  sum_j ((z_j - x_j) / sigma_j)^2
subject to  A z >= b
```

with CVXPY. `sigma_j` is learned only from a reference training CSV. This
normalization prevents features with large numeric units from dominating which
values are repaired. Constraints remain in original units. Rows already
satisfying every constraint are copied exactly. Use `--linear-scale-mode none` to
recover the paper's literal unscaled `||z - x||_2^2` objective.

Constraint files are JSON lists modeled after the repository's equational
annotations. Every entry uses one canonical representation:

```json
{
  "id": "x_max_ge_x_min",
  "description": "Maximum X is no smaller than minimum X.",
  "formula": "X_Maximum >= X_Minimum",
  "columns": ["X_Maximum", "X_Minimum"],
  "coefficients": {"X_Maximum": 1, "X_Minimum": -1},
  "sense": ">=",
  "rhs": 0,
  "mutable_columns": ["X_Maximum", "X_Minimum"],
  "explanation": "Bounding-box extrema are ordered."
}
```

`mutable_columns` lets a rule refer to fixed numeric categorical columns while
only repairing continuous columns. The target is never changed unless it is
explicitly listed as mutable.

Repair every available constraint family through the unified fixer:

```bash
uv run python -m constraints.expert_constraints_fix \
  experiments/url/gaussian_copula/unconstrained \
  experiments/url/gaussian_copula/expert_fixed \
  dataset/url/constraints_expert
```

Use `--synthetic`, `--train`, `--test`, and `--metadata` to override defaults
with exact files. Evaluate raw and repaired experiments through the standard
evaluation entry point:

```bash
uv run python -m evaluation.evaluate_experiment \
  experiments/url/gaussian_copula/unconstrained \
  --constraints-expert dataset/url/constraints_expert \
  --output experiments/url/gaussian_copula/unconstrained/evaluation.json

uv run python -m evaluation.evaluate_experiment \
  experiments/url/gaussian_copula/expert_fixed \
  --constraints-expert dataset/url/constraints_expert \
  --output experiments/url/gaussian_copula/expert_fixed/evaluation.json
```
