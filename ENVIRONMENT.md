# Environment and numerical reproducibility

## Recorded software environment

The original multi-run command manifests recorded:

- Python 3.13.2, packaged by Anaconda, compiled with GCC 11.2.0.
- Generator device option: `auto`.

The artifact pins Python in `.python-version` and all Python packages in
`uv.lock`. Important resolved versions include:

| Component | Version |
| --- | --- |
| Python | 3.13.2 |
| NumPy | 2.5.0 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.0 |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime used by PyTorch | 12.4 |
| cuDNN | 9.1.0 |
| SDV | 1.32.1 |
| SDMetrics | 0.25.0 |
| CVXPY | 1.9.2 |
| XGBoost | 3.3.0 |

The packaging host ran Ubuntu 24.04.4 LTS with Linux 6.8.0 on x86-64. At
packaging time it exposed four NVIDIA RTX 6000 Ada Generation GPUs with driver
575.57.08. The original command manifests did not record the resolved device or
GPU identity, so this hardware record is context rather than proof that every
original training job used the same device.

Install the environment with:

```bash
uv sync --frozen
```

## Determinism boundary

The experiment configurations freeze split, training, and sampling seeds. The
runners seed Python, NumPy, and PyTorch, and Gaussian Copula uses an explicitly
seeded SDV sampling state. The neural runners do not enable PyTorch deterministic
algorithms or pin GPU kernel selection. Consequently:

- dataset splits are expected to be byte-identical, and packaged frozen
  constraint files should be reused unchanged;
- CPU/GPU changes can produce different neural checkpoints and synthetic rows;
- hosted constraint-discovery model responses can change over time; and
- downstream metric values should be compared with a numerical tolerance, not
  required to be bit-identical.

Use the included frozen constraints when validating synthesis,
postprocessing, and evaluation without hosted-model variability.

## Artifact validation tolerances

These are operational artifact checks, not claims of statistical equivalence.
Compute the same per-dataset, generator, and raw/constrained averages from the
included reference `evaluation.json` files and from the reproduced run.

- Dataset, split, configuration, and frozen-constraint file contents: exact.
- Status labels, row counts, and metric availability: exact.
- Metrics bounded to `[0, 1]`, including quality scores, ROC AUC, CVR, and
  SCVC: absolute mean difference no larger than the greater of `0.02` or one
  original sample standard deviation.
- R-squared: absolute mean difference no larger than the greater of `0.05` or
  one original sample standard deviation.
- RMSE and other scale-dependent errors: relative mean difference no larger
  than the greater of `5%` or one original sample standard deviation expressed
  relative to the original mean.

If a result falls outside these checks, first rerun on CPU or on the documented
CUDA/PyTorch stack, confirm that frozen constraints were restored, and verify
that the packaged datasets, configurations, and constraints were not modified.
