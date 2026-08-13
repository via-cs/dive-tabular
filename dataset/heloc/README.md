# HELOC

## Description

The HELOC dataset comes from the FICO Explainable Machine Learning Challenge.
Each row represents an anonymized home-equity line-of-credit application. The
binary target, `is_at_risk`, describes repayment risk. Preprocessing converts
the original labels, removes rows containing only the `-9` sentinel, and
removes rows that violate selected total-trade consistency rules.

## Source

- [FICO HELOC dataset on Hugging Face](https://huggingface.co/datasets/mstz/heloc)

The download script uses a pinned revision of the public source.

## Create `data.csv`

Run from the repository root:

```bash
uv run python dataset/heloc/raw/download_heloc.py
uv run python dataset/heloc/raw/prepreocess_heloc.py --force
```

The result is written to `dataset/heloc/data.csv`. The preprocessing script
also regenerates `dataset/heloc/info.json`.

Note that the preprocessing filename intentionally follows the repository’s
existing spelling: `prepreocess_heloc.py`.
