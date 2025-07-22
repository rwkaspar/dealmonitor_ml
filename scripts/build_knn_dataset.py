import os
import sys
import json
import pandas as pd
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from src.knn_dataset_builder import build_knn_training_rows

rows = []
with open("data/raw/raw_data.jsonl", "r") as f:
    for line in f:
        raw = json.loads(line)
        rows.extend(build_knn_training_rows(raw))

df = pd.DataFrame(rows)
df.to_parquet("data/knn_training_set.parquet")
print(f"✅ {len(df)} Trainingspunkte gespeichert.")
