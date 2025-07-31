import os
import joblib
import pandas as pd
import logging
from .features import extract_price_features, clean_price_user
from .candidate_extractor import extract_price_candidates
from .model_nn_v3 import build_feature_df

logger = logging.getLogger(__name__)

_model = None  # Global cache
model_path = "models/nn_model_latest.pkl"
model = joblib.load(model_path)


def load_model(path: str = "models/model_best.pkl"):
    """
    Load the trained model from disk (with caching).
    """
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model not found at: {path}")
        _model = joblib.load(path)
    return _model

def predict_best_candidate_nn(row: dict):
    """
    Predict best price candidate from a raw_data row (with candidate list inside).
    """
    candidates = row.get("candidates", [])
    if not candidates:
        return None, []

    df = pd.DataFrame(candidates)
    df["price_user"] = clean_price_user(row.get("price_user"))

    # Apply feature extraction
    feature_df = df.apply(lambda r: extract_price_features(r.to_dict()), axis=1)
    X = pd.DataFrame(list(feature_df))

    # Ensure same feature order
    expected_cols = list(model.feature_names_in_)
    missing = set(expected_cols) - set(X.columns)
    if missing:
        raise ValueError(f"Missing features in input: {missing}")

    X = X[expected_cols]

    # Predict
    probs = model.predict_proba(X)[:, 1]
    df["proba"] = probs

    # Sort and return best
    top_candidates = sorted(
        zip(df["value_clean"], probs, df.get("outer_html", [""] * len(df))),
        key=lambda x: -x[1]
    )

    best_price = top_candidates[0][0] if top_candidates else None
    return best_price, top_candidates

def predict_best_candidates_nn_from_row(row: dict):
    logger.info(f"Predicting best candidates for row ID: {row.get('id', 'unknown')}")
    price_user = row.get("price_user", None)
    price_user_clean = clean_price_user(price_user)
    row['value_clean'] = price_user_clean

    candidates = extract_price_candidates(row['content_html'], row['xhrs'])
    results = []
    for cand in candidates:
        value_clean = clean_price_user(cand["value_raw"])
        if value_clean is None:
            continue
        match = abs(value_clean - price_user_clean) < 0.01

        cand_with_features = {
            "raw_data_id": row.get("id", ""),
            "source": cand.get("source"),
            "value_clean": value_clean,
            "match_with_user": int(match),
            "depth": cand.get("depth", -1),
            "tag": cand.get("tag", ""),
            "css_len": len(cand.get("css_class", "")),
            "has_currency": int("€" in cand["value_raw"] or "$" in cand["value_raw"]),
            "price_user": price_user_clean,
        }
        results.append(cand_with_features)
    row['candidates'] = results

    # --- Prediction ---
    X = build_feature_df(pd.DataFrame(row['candidates']))
    X = X.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if X.empty:
        logger.warning(f"No valid candidates for row ID: {row.get('id', 'unknown')}")
        row["proba"] = []
        return row

    meta_cols = ["raw_data_id", "price_user", "value_clean"]
    X_features = X.drop(columns=meta_cols, errors="ignore")

    logger.debug("Features for prediction: %s", list(X_features.columns))
    logger.debug("Model was trained on: %s", model.feature_names_in_)

    try:
        probs = model.predict_proba(X_features)[:, 1]
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        row["proba"] = []
        return row

    # Score zu jedem Kandidaten hinzufügen
    for i, cand in enumerate(row['candidates']):
        cand["proba"] = float(probs[i])

    # Besten Kandidaten bestimmen
    best_idx = probs.argmax()
    row["best_candidate"] = row['candidates'][best_idx]
    row["best_proba"] = float(probs[best_idx])

    return row
