import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nn_predictor import predict_best_candidate_nn

with open("data/raw/raw_data.jsonl") as f:
    row = json.loads(f.readline())

predicted, top = predict_best_candidate_nn(row)

print(f"🔍 Predicted price: {predicted} €")
print("🔝 Top-Kandidats:")
for val, prob, meta in top:
    print(f"  → {val:.2f} € (score={prob:.3f}, tag={meta.get('tag')}, class={meta.get('css_class')})")
