import pandas as pd
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from src.features import extract_features, clean_price_user

df = pd.read_json("data/raw/raw_data.jsonl", lines=True)
df["target"] = df["price_user"].apply(clean_price_user)
df = df[df["target"].notnull()]

features = df.apply(extract_features, axis=1, result_type="expand")
dataset = pd.concat([features, df["target"]], axis=1)

dataset.to_parquet("data/dataset.parquet")
