import sys, os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.knn_model import train_knn_model
from src.knn_predictor import predict_best_candidate

train_knn_model("data/knn_training_set.parquet", "models/knn_model.pkl")

with open("data/raw/raw_data.jsonl") as f:
    example = json.loads(f.readline())

print(f"🔍 Beispiel: {example.get("url")}")
predicted = predict_best_candidate(example)
print(f"🔍 Vorhergesagter Preis: {predicted} €, korrekter Preis: {example.get("price_user")}")