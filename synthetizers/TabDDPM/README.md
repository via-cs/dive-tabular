# TAB-DDPM runner

`run_TabDDPM.py` trains the official TAB-DDPM Gaussian/multinomial diffusion
core through this repository's standard experiment lifecycle.

## Train and sample

Run from the repository root:

```bash
uv run python run_TabDDPM.py train-sample \
  --data-dir dataset/steel \
  --output-dir experiments/steel/tabddpm/unconstrained \
  --steps 30000 \
  --num-timesteps 100 \
  --hidden-dims 256,512,512,256 \
  --batch-size 1024 \
  --num-files 3 \
  --device auto
```

When no command is given, `train-sample` is implied. The runner supports
separate `train` and `sample` commands matching the CTGAN and TVAE runners.

By default, every column's `cat` or `num` designation comes from
`<data-dir>/info.json`. The target comes from `utility_feature.json`. Exact
input files can be supplied with `--data-file`, `--info-file`, and
`--utility-feature-file`.

For deliberate type experiments, categorical and numerical feature lists can
be overridden separately:

```bash
uv run python run_TabDDPM.py train-sample \
  --data-dir dataset/example \
  --data-file /path/to/example.csv \
  --info-file /path/to/example-info.json \
  --utility-feature-file /path/to/example-utility.json \
  --output-dir experiments/example/tabddpm/unconstrained \
  --categorical-columns category_a,category_b \
  --numerical-columns value_a,value_b
```

The target is excluded from overrides and retains its `info.json` type. If
only one feature list is supplied, the other is inferred as its complement.

## Fast smoke test

```bash
uv run python run_TabDDPM.py train-sample \
  --data-dir dataset/steel \
  --output-dir /tmp/tabddpm-steel-smoke \
  --max-rows 128 \
  --steps 2 \
  --batch-size 64 \
  --hidden-dims 32,32 \
  --time-embedding-dim 16 \
  --num-timesteps 4 \
  --num-rows 8 \
  --sample-batch-size 4 \
  --device cpu
```

## Artifacts

The output directory contains:

- `tabddpm.pt`: raw and EMA denoiser weights plus model configuration;
- `tabddpm_preprocessor.pkl`: train-fitted numerical and categorical transforms;
- `train_loss.csv`: logged Gaussian, multinomial, and total losses;
- `train.csv`, `test.csv`, `metadata.json`, and `label_maps.json`;
- `train_config.json`; and
- `synthetic/synthetic_N.csv` plus its `sample_config.json`.

Standalone sampling defaults to EMA weights:

```bash
uv run python run_TabDDPM.py sample \
  --experiment-dir experiments/steel/tabddpm/unconstrained \
  --num-files 3 \
  --checkpoint-variant ema \
  --device auto
```
