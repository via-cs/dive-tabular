# Steel Industry Energy Consumption

## Description

This regression dataset contains electricity-use observations from a steel
facility at 15-minute intervals. It includes active and reactive energy,
power-factor measurements, carbon-dioxide emissions, time and weekday context,
and load category. The target is `Usage_kWh`.

## Source

- [Steel Industry Energy Consumption on Kaggle](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption)

The download may require local Kaggle credentials.

## Create `data.csv`

Run from the repository root:

```bash
uv run python dataset/steel/raw/download_steel.py dataset/steel/raw

uv run python dataset/steel/preprocess_steel.py dataset/steel/raw \
  --output dataset/steel/data.csv \
  --metadata-output dataset/steel/raw/preprocessing_info.json \
  --info-output dataset/steel/info.json \
  --force
```

The result is written to `dataset/steel/data.csv`.
