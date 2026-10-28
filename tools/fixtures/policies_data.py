"""Writes the data of the `policies` fixture bundle as Parquet: categories coded by
integers and by floats (which only a typed file keeps), a boolean, a text category
with missing values, numbers with missing values and -999 markers, and a binary
target. Deterministic.

    python3 tools/fixtures/policies_data.py <out.parquet>

Needs pyarrow.
"""
import math
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

rng = random.Random(7)
cols = {k: [] for k in ("zone", "grade", "flag", "mode", "temp", "load", "status")}
for _ in range(1500):
    zone = rng.choice([1, 2, 3, 4])
    grade = rng.choice([1.0, 2.0, 3.0])
    flag = rng.random() < 0.3
    mode = rng.choice(["auto", "manual", "service", None])
    temp = rng.gauss(60.0, 8.0)
    r = rng.random()
    if r < 0.05:
        temp = None
    elif r < 0.10:
        temp = -999.0
    load = rng.gauss(1.0, 0.3)
    t = 60.0 if temp in (None, -999.0) else temp
    score = 0.4 * (zone - 2.5) + 0.5 * (grade - 2.0) + (0.8 if flag else 0.0) + (0.6 if mode == "service" else 0.0)
    score += 0.05 * (t - 60.0) + 1.2 * (load - 1.0)
    status = "fail" if rng.random() < 1.0 / (1.0 + math.exp(-score)) else "ok"
    for k, v in zip(cols, (zone, grade, flag, mode, temp, load, status)):
        cols[k].append(v)

types = {"zone": pa.int64(), "grade": pa.float64(), "flag": pa.bool_(), "mode": pa.string(),
         "temp": pa.float64(), "load": pa.float64(), "status": pa.string()}
pq.write_table(pa.table({k: pa.array(v, types[k]) for k, v in cols.items()}), sys.argv[1])
print(sys.argv[1])
