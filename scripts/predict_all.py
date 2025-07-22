import os, sys, json
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nn_predictor import predict_best_candidate_nn

# Load raw_data
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://dev_user:dev_password@localhost:5432/dev_db")
engine = create_engine(DATABASE_URL)

print("📥 Loading raw_data...")
df = pd.read_sql("SELECT * FROM raw_data", engine)
print(f"🔢 {len(df)} rows loaded.")

# Predict
results = []
for row in df.to_dict(orient="records"):
    pred, top = predict_best_candidate_nn(row)
    best_score = top[0][1] if top else 0
    results.append({
        "fingerprint": row.get("fingerprint"),
        "price_user": row.get("price_user"),
        "predicted_price": pred,
        "score": round(best_score, 3),
        "created_at": str(row.get("created_at")),
        "top_candidates": [(round(val, 2), round(prob, 3)) for val, prob, _ in top]
    })

# Save
os.makedirs("predictions", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"predictions/predictions_{timestamp}.jsonl"

with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"✅ Saved predictions to {out_path}")
