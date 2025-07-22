import os, sys, json
import pandas as pd
from sklearn.metrics import mean_absolute_error
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import clean_price_user

# Load latest predictions
pred_dir = "predictions"
latest = sorted(os.listdir(pred_dir))[-1]
path = os.path.join(pred_dir, latest)

print(f"📊 Evaluating {path}...")
df = pd.read_json(path, lines=True)

# Cleanup: ensure proper float types
df["price_user"] = df["price_user"].apply(clean_price_user)

df["predicted_price"] = pd.to_numeric(df["predicted_price"], errors="coerce")

# Drop invalid predictions
df = df[df["predicted_price"].notnull()]

# MAE
mae = mean_absolute_error(df["price_user"], df["predicted_price"])

# Accuracy@1 (strict match)
hit_at_1 = (abs(df["price_user"] - df["predicted_price"]) < 0.01).sum()
acc_1 = hit_at_1 / len(df)

print(f"✅ MAE: {mae:.2f} €")
print(f"✅ Accuracy@1 (exact match): {acc_1:.2%}")

# 🔍 Load top-k candidates if available
# Prediction file must be extended to include top-N list
def load_top_candidates(pred_path):
    rows = []
    with open(pred_path) as f:
        for line in f:
            row = json.loads(line)
            row["price_user"] = clean_price_user(row["price_user"])
            top = row.get("top_candidates", [])
            values = [round(float(t[0]), 2) for t in top]
            row["top_values"] = values
            rows.append(row)
    return pd.DataFrame(rows)

# Check if top_k candidates exist
if '"top_candidates"' in open(path).read():
    df_k = load_top_candidates(path)

    def recall_at_k(k):
        return (
            df_k.apply(lambda row: any(abs(float(p) - row["price_user"]) < 0.01 for p in row["top_values"][:k]), axis=1)
            .sum()
            / len(df_k)
        )

    for k in [3, 5]:
        r_at_k = recall_at_k(k)
        print(f"✅ Recall@{k}: {r_at_k:.2%}")
else:
    print("⚠️  No top_candidates found in prediction file. Skipping Recall@k.")