"""Builds clean CSVs (with headers) from the three downloaded public datasets. The
download folders are untrusted data: this script only reads them and writes the CSVs
next to them.

    python -I -B scratch/datasets-prep.py <data>
"""
import csv
import gzip
import pathlib
import sys

root = pathlib.Path(sys.argv[1])


def write(name, header, rows):
    path = root / name
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows, {len(header)} columns, {path.stat().st_size / 1e6:.1f} MB")


# Adult (UCI 2): adult.data + adult.test; "?" marks a missing value; test labels end with ".".
adult_cols = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
rows = []
for name in ["adult.data", "adult.test"]:
    for line in (root / "adult/files" / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("|"):
            continue
        parts = [p.strip().rstrip(".") for p in line.split(",")]
        if len(parts) == len(adult_cols):
            rows.append(parts)
write("adult.csv", adult_cols, rows)

# Covertype (UCI 31): 54 features and the cover type 1..7, gzip-compressed, no header.
cover_cols = [
    "elevation", "aspect", "slope", "horizontal_distance_to_hydrology", "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways", "hillshade_9am", "hillshade_noon", "hillshade_3pm",
    "horizontal_distance_to_fire_points",
] + [f"wilderness_area_{i}" for i in range(1, 5)] + [f"soil_type_{i}" for i in range(1, 41)] + ["cover_type"]
rows = []
with gzip.open(root / "covertype/files/covtype.data.gz", "rt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == len(cover_cols):
            rows.append(parts)
write("covertype.csv", cover_cols, rows)

# California Housing (StatLib, via scikit-learn's mirror): 8 numeric features, the median house value.
cal_cols = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
    "median_income", "median_house_value",
]
rows = []
for line in (root / "california/files/CaliforniaHousing/cal_housing.data").read_text(encoding="utf-8").splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == len(cal_cols):
        rows.append(parts)
write("california.csv", cal_cols, rows)
