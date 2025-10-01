import pandas as pd
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
# from src.features import extract_price_features, clean_price_user

sys.path.append(os.path.abspath("dealmonitor/backend/src"))
from dealmonitor.features.features import (
    clean_price,
    extract_price_features,
)

df = pd.read_json("data/raw/raw_data.jsonl", lines=True)

# df["target"] = df["price_user"].apply(clean_price_user)
df["target"] = df["price_user"].apply(clean_price)

df = df[df["target"].notnull()]

features = df.apply(extract_price_features, axis=1, result_type="expand")
dataset = pd.concat([features, df["target"]], axis=1)

dataset.to_parquet("data/dataset.parquet")
dataset.to_csv("data/dataset.csv", index=False)