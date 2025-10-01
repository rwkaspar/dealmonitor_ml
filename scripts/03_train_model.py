import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from src.model import train_model, predict_price
from src.evaluate import evaluate_model
from src.candidate_extractor import extract_price_candidates

DATA_PATH = "data/dataset.parquet"
MODEL_PATH = "models/price_model.pkl"

os.makedirs("models", exist_ok=True)

train_model(data_path=DATA_PATH, model_path=MODEL_PATH)

evaluate_model(model_path=MODEL_PATH, data_path=DATA_PATH, max_points=1000)

example = {
    "content_html": "<html><span>€1.399,00</span></html>",
    "snapshot_dom": "<span>€1.399,00</span>",
    "xhrs": {},
    "user_language": "de",
    "user_agent": "Mozilla",
}

price = predict_price(example)
print(f"📦 Predicted Price: {price} €")