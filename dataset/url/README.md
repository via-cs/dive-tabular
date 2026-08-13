# Web Page Phishing Detection

## Description

This balanced binary-classification dataset contains URL, page-content, and
third-party-service features for legitimate and phishing webpages. The target
is `status`. Preprocessing removes the near-unique raw URL and the
constant-zero `submit_email` field.

## Source

- [Mendeley Data: Web Page Phishing Detection](https://data.mendeley.com/datasets/c2gw7fy2j4/3)
- DOI: [10.17632/c2gw7fy2j4.3](https://doi.org/10.17632/c2gw7fy2j4.3)
- [Kaggle mirror used by the downloader](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)
- License: CC BY 4.0

The Kaggle download may require local Kaggle credentials.

## Create `data.csv`

Run from the repository root:

```bash
uv run python dataset/url/raw/download_url.py
uv run python dataset/url/raw/preprocess_url.py --force
```

The result is written to `dataset/url/data.csv`. Preprocessing also writes an
audit file under `dataset/url/raw/` and regenerates `dataset/url/info.json`.
