import os, sys, json
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from src.candidate_extractor import extract_all_candidates, extract_price_candidates
# from src.features import extract_price_features, clean_price_user
from src.model_nn_v3 import build_feature_df
from src.nn_predictor import load_model, predict_best_candidates_nn_from_row

sys.path.append(os.path.abspath("dealmonitor/backend/src"))
from dealmonitor.features.features import clean_price, extract_price_features
from dealmonitor.price_logic.candidate_extractor import (
    extract_all_candidates,
    extract_price_candidates,
)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://dev_user:dev_password@localhost:5432/dev_db")
engine = create_engine(DATABASE_URL)

print("📥 Loading raw_data...")
raw_df = pd.read_sql("SELECT * FROM raw_data", engine)
print(f"🔢 {len(raw_df)} rows loaded.")

# --- Prepare model ---
model = load_model()

# --- Prepare dataset ---
# raw_df["target"] = raw_df["price_user"].apply(clean_price_user)
raw_df["target"] = raw_df["price_user"].apply(clean_price)
raw_df = raw_df[raw_df["target"].notnull()]

results = []
df = pd.DataFrame()
# go through each row, extract features, prepare for prediction and do the prediction
for i, raw_row in enumerate(raw_df.itertuples()):
    print(type(raw_row))
    print(dir(raw_row))

    row = raw_row._asdict()  # convert namedtuple to dict
    # test = build_knn_training_rows(row)
    # extract candidates from the row
    updated_row = predict_best_candidates_nn_from_row(row)
    candidates = updated_row.get("candidates", [])
    if not candidates:
        continue  # skip ohne Kandidaten
    # df = pd.concat([df, pd.DataFrame([updated_row])], ignore_index=True)

    # Sortiere nach Prediction-Score (proba)
    sorted_cands = sorted(candidates, key=lambda c: c.get("proba", 0), reverse=True)
    top_candidates = sorted_cands[:3]
    pred_price = top_candidates[0]["value_clean"] if top_candidates else None
    expected_price = updated_row.get("target", None)
    is_hit = any(abs(expected_price - c["value_clean"]) < 0.01 for c in top_candidates if expected_price is not None)

    # Fingerprint, created_at, etc. holen (optional: nachbauen falls nötig)
    results.append({
        "raw_data_id": updated_row.get("id"),
        "url": updated_row.get("url"),
        "fingerprint": updated_row.get("fingerprint", ""),
        "price_user": expected_price,
        "predicted_price": pred_price,
        "score": round(float(top_candidates[0]["proba"]), 3) if top_candidates else None,
        "top_candidates": [
            (round(float(c["value_clean"]), 2), round(float(c["proba"]), 3)) for c in top_candidates
        ],
        "created_at": updated_row.get("created_at", ""),
        "is_hit": is_hit
    })

# --- Speichere Ergebnisse ---
import os, json
from datetime import datetime
os.makedirs("predictions", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = f"predictions/predictions_{timestamp}.jsonl"

with open(out_path, "w") as f:
    for r in results:
        # Zeitstempel ggf. korrekt zu String machen
        if hasattr(r["created_at"], "isoformat"):
            r["created_at"] = r["created_at"].isoformat()
        f.write(json.dumps(r) + "\n")

# latest
with open("predictions/predictions_latest.jsonl", "w") as f:
    for r in results:
        if hasattr(r["created_at"], "isoformat"):
            r["created_at"] = r["created_at"].isoformat()
        f.write(json.dumps(r) + "\n")

print(f"✅ Saved predictions to {out_path}")
    # df = df.loc[X.index].reset_index(drop=True)
    # X = X.reset_index(drop=True)
    # meta_cols = ["raw_data_id", "price_user", "value_clean"]
    # X_features = X.drop(columns=meta_cols, errors="ignore")

    # print("Features at prediction:", X_features.columns)
    # print("Model was trained on n_features_in_:", getattr(model, "n_features_in_", "??"))    
    # --- Extract features ---

# --- Generate candidates like during training ---
# candidates = extract_all_candidates(raw_df)  # returns rows with value_clean, raw_data_id, source, tag, etc.
# candidates_df = pd.DataFrame(candidates)
# candidates_df["price_user"] = candidates_df["price_user"].apply(clean_price_user)

# --- Extract features ---
# X = build_feature_df(df)
# X = X.replace([float("inf"), float("-inf")], pd.NA).dropna()
# df = df.loc[X.index].reset_index(drop=True)
# X = X.reset_index(drop=True)
# meta_cols = ["raw_data_id", "price_user", "value_clean"]
# X_features = X.drop(columns=meta_cols, errors="ignore")

# print(X_features.describe())
# print(X_features.head(20))




# # --- Predict ---
# probs = model.predict_proba(X_features)[:, 1]
# df["proba"] = probs



# # --- Predict ---
# print("Features for prediction:", list(X_features.columns))
# print("Model was trained on:", model.feature_names_in_, "features")

# probs = model.predict_proba(X_features)[:, 1]
# predictions = model.predict(X_features)
# df["proba"] = probs

# --- Group by raw_data_id & get top candidates ---
# results = []
# for raw_data_id, group in df.groupby("id"):
#     sorted_group = group.sort_values("proba", ascending=False)
#     expected_price = group["value_clean"].iloc[0]

#     top_candidates = sorted_group[["value_clean", "proba"]].values[:3].tolist()
#     pred_price = top_candidates[0][0] if top_candidates else None
#     is_hit = any(abs(expected_price - val) < 0.01 for val, _ in top_candidates)

#     results.append({
#         "raw_data_id": int(raw_data_id),
#         "fingerprint": group["fingerprint"].iloc[0],
#         "price_user": expected_price,
#         "predicted_price": pred_price,
#         "score": round(top_candidates[0][1], 3) if top_candidates else None,
#         "top_candidates": [(round(val, 2), round(prob, 3)) for val, prob in top_candidates],
#         "created_at": group["created_at"].iloc[0].isoformat() if hasattr(group["created_at"].iloc[0], "isoformat") else str(group["created_at"].iloc[0]),
#         "is_hit": is_hit
#     })

# # --- Save results ---
# os.makedirs("predictions", exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# out_path = f"predictions/predictions_{timestamp}.jsonl"

# with open(out_path, "w") as f:
#     for r in results:
#         f.write(json.dumps(r) + "\n")

# # latest
# with open("predictions/predictions_latest.jsonl", "w") as f:
#     for r in results:
#         f.write(json.dumps(r) + "\n")

# print(f"✅ Saved predictions to {out_path}")
