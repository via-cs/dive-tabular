# NYC Green Taxi

## Description

This total-amount regression dataset is based on a 60,000-row sample of 2015
NYC Green Taxi trips. It contains trip geography, distance, itemized charges,
clock and duration features, dispatch and fare codes, passenger count, and
payment type. Preprocessing removes four invalid-duration rows and rebuilds the
payment total from its components.

## Source

- Original records: [NYC Taxi & Limousine Commission Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Paper input: [60,000-row 2015 Green Taxi sample hosted by MoET](https://github.com/mgdhooghe/MoET/blob/b0940a2aad6400f44eab1796e54326fea2913e1c/dataset/2015_green_60000.csv)

## Create `data.csv`

Download the exact 60,000-row input used by the preprocessor from the pinned
MoET revision:

```bash
mkdir -p dataset/taxi/raw
curl -L \
  https://raw.githubusercontent.com/mgdhooghe/MoET/b0940a2aad6400f44eab1796e54326fea2913e1c/dataset/2015_green_60000.csv \
  -o dataset/taxi/raw/2015_green_60000.csv
```

For reproducibility, the expected SHA-256 is
`1f256e862eec3fb8c274c47f33b2a241f4be74e6905d5aeb34d2de7be623e2fc`.
The file originates from 2015 NYC TLC Green Taxi trip records and is
redistributed by MoET. MoET does not document the procedure used to select the
60,000 rows, so cite both NYC TLC as the original source and MoET as the source
of the exact sample.

Then run from the repository root:

```bash
uv run python dataset/taxi/preprocess_taxi.py \
  dataset/taxi/raw/2015_green_60000.csv \
  --output dataset/taxi/data.csv \
  --metadata-output dataset/taxi/raw/preprocessing_info.json \
  --info-output dataset/taxi/info.json \
  --force
```

The result is written to `dataset/taxi/data.csv`.
