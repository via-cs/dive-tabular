# Flights

## Description

This arrival-delay regression dataset contains U.S. flights from 2019–2023,
including airlines, routes, operational times, durations, distance, and delay
measurements. Preprocessing removes cancelled or diverted records, filters
invalid rows, converts local clock fields to UTC minute-of-day values, and
retains rows satisfying the configured time relationships.

## Source

- Flight data: [Flight Delay and Cancellation Dataset (2019–2023) on Kaggle](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
- Airport timezones: [OpenFlights Airports data](https://openflights.org/data.php),
  redistributed by the [MoET repository](https://github.com/mgdhooghe/MoET)

The Kaggle download may require local Kaggle credentials. The airport metadata
is an OpenFlights airport-database snapshot with a header row added by MoET.
OpenFlights makes the database available under the Open Database License
(ODbL) 1.0; see its data page for attribution and license details.

## Create `data.csv`

Run from the repository root:

```bash
uv run python dataset/flights/raw/download_flights_sample.py dataset/flights/raw

curl -L \
  https://raw.githubusercontent.com/mgdhooghe/MoET/b0940a2aad6400f44eab1796e54326fea2913e1c/data/flights/airports_labeled.dat \
  -o dataset/flights/raw/airports_labeled.dat

uv run python dataset/flights/preprocess_flights.py \
  dataset/flights/raw/flights_sample_3m.csv \
  --airports dataset/flights/raw/airports_labeled.dat \
  --output dataset/flights/raw/flights_preprocessed.csv \
  --force

uv run python dataset/flights/sample_preprocessed_flights.py \
  dataset/flights/raw/flights_preprocessed.csv \
  --output dataset/flights/data.csv \
  --rows 60000 \
  --seed 42 \
  --force
```

For reproducibility, the expected SHA-256 of `airports_labeled.dat` is
`6734d6a2c296befd983c91b1dd07b7ef07019f5bd6a24084a775884a7d771e9d`.

Preprocessing retains 2,391,146 valid rows and writes them to
`dataset/flights/raw/flights_preprocessed.csv`. It also writes the audit file
`dataset/flights/raw/preprocessing_info.json`. The sampling command then writes
the deterministic 60,000-row experiment table to `dataset/flights/data.csv`.
Its expected SHA-256 is
`733de150aab2eea0157b7ab3090f6c9da9957737e30b62a2458ac48d67a8e01c`.
