import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nn_predictor import predict_best_candidate_nn

RAW_DATA_PATH = "data/raw/raw_data.jsonl"
SEARCH_TERM = "ali"

with open(RAW_DATA_PATH) as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        if SEARCH_TERM in row["url"]:
            break
    else:
        raise ValueError(f"No entry found containing {SEARCH_TERM}")

predicted, top = predict_best_candidate_nn(row)

print(f"🔍 Predicted price from {row["url"]}: {predicted} €")
print("🔝 Top-Kandidats:")
for val, prob, meta in top:
    print(f"  → {val:.2f} € (score={prob:.3f}, tag={meta.get('tag')}, class={meta.get('css_class')})")
