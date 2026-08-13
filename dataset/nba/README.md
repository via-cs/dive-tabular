# NBA

## Description

This regression dataset contains NBA regular-season player scoring statistics
from 1996–97 through 2025–26. Statistics are normalized per 100 offensive
possessions. Identifiers and exposure columns are removed, leaving scoring
volume, accuracy, shot-mix, efficiency, blocked-attempt, and usage measures.

## Source

- [pbpstats](https://pbpstats.com/)
- [pbpstats totals API](https://api.pbpstats.com/get-totals/nba)

## Create `data.csv`

The NBA script downloads every configured season and preprocesses them in one
command. Run from the repository root:

```bash
uv run python dataset/nba/raw/download_all_seasons.py
```

Season files are stored under `dataset/nba/raw/all_seasons/`, the processed
table is written to `dataset/nba/data.csv`, and the preprocessing report is
written to `dataset/nba/raw/preprocessing.txt`.

Use `--force-download` to replace downloaded season files and
`--force-output` to replace existing outputs.
