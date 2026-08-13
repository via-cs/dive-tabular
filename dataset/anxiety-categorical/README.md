# Categorical Anxiety

## Description

This auxiliary dataset is a low-cardinality categorical view of a social
anxiety and lifestyle survey. Continuous measurements and ordinal scores are
grouped into interpretable bands for categorical-constraint discovery and
evaluation. It is not one of the seven datasets in the main result table.

## Source

- [Social Anxiety Dataset on Kaggle](https://www.kaggle.com/datasets/natezhang123/social-anxiety-dataset)
- License: CC0 Public Domain

The download may require local Kaggle credentials.

## Create `data.csv`

Download the source CSV from the repository root:

```bash
uv run python dataset/anxiety-categorical/raw/download_anxiety.py
```

The source is written to `dataset/anxiety-categorical/raw/data.csv`.

The provided preprocessing entry point is:

```bash
uv run python dataset/anxiety-categorical/raw/preprocess_anxiety.py --overwrite
```

This creates `dataset/anxiety-categorical/data.csv`, refreshes `info.json` and
`meta.json`, and writes the expert categorical constraints under
`constraints_expert/`. Pass `--overwrite` again whenever those generated files
already exist.
