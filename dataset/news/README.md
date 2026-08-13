# Online News Popularity

## Description

This binary-classification dataset contains extracted features for Mashable
articles. Preprocessing removes article URLs and raw share counts, combines the
weekday indicators, removes rows with undefined positive/negative word-rate
ratios, and defines `is_popular` as at least 1,400 shares.

## Source

- [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity)
- DOI: [10.24432/C5NS3V](https://doi.org/10.24432/C5NS3V)
- License: CC BY 4.0

## Create `data.csv`

Run from the repository root:

```bash
uv run python dataset/news/raw/download_news.py
uv run python dataset/news/raw/preprocess_news.py --force
```

The result is written to `dataset/news/data.csv`. Preprocessing also writes an
audit file under `dataset/news/raw/` and regenerates
`dataset/news/info.json`.
