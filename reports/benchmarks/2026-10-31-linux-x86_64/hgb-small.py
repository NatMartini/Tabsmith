"""Why HistGradientBoosting takes seconds on the README's small problem: the same fit at
several thread counts, with and without what the benchmark runs before it."""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

path = sys.argv[1]
print("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "default"), flush=True)
text = ["line", "shift", "supplier", "device_type", "firmware"]
df = pd.read_csv(path, dtype={c: str for c in text}, keep_default_na=False, na_values=["", "NA", "null"])
y = df["defect"].astype(int).to_numpy()
num = [c for c in df.columns if c not in text + ["defect", "unit_id", "device_id", "batch_id", "maintenance_flag"] and pd.api.types.is_numeric_dtype(df[c])]
cat = text
x = df[num + cat]
rng = np.random.default_rng(7)
tr = rng.permutation(len(df))[:7356]
pos = y[tr].sum()
weights = np.where(y[tr] == 1, len(tr) / (2 * pos), len(tr) / (2 * (len(tr) - pos)))
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="<missing>"), OneHotEncoder(handle_unknown="ignore")), cat),
])
t0 = time.perf_counter()
features = pre.fit_transform(x.iloc[tr])
print(f"preprocess {time.perf_counter() - t0:.3f} s, shape {features.shape}, sparse {hasattr(features, 'toarray')}", flush=True)
if hasattr(features, "toarray"):
    features = features.toarray()
for label, kw in (("with weights", {"sample_weight": weights}), ("no weights", {})):
    model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=54, max_leaf_nodes=31, max_depth=8, min_samples_leaf=20, max_bins=255, l2_regularization=1.0, early_stopping=False, random_state=7)
    t0 = time.perf_counter()
    model.fit(features, y[tr], **kw)
    print(f"HGB fit, {label}: {time.perf_counter() - t0:.3f} s", flush=True)
