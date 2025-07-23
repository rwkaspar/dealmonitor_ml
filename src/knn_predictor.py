import pandas as pd
import joblib
from .candidate_extractor import extract_price_candidates
from .features import clean_price_user


def predict_best_candidate(raw_row: dict, model_path: str = "models/knn_model.pkl") -> float | None:
    model = joblib.load(model_path)

    candidates = extract_price_candidates(
        raw_row.get("content_html", ""),
        raw_row.get("xhrs", {})
    )

    rows = []
    for cand in candidates:
        value_clean = clean_price_user(cand["value_raw"])
        if value_clean is None:
            continue

        row = {
            "value_clean": value_clean,
            "depth": cand.get("depth", -1),
            "css_len": len(cand.get("css_class", "")),
            "has_currency": int("€" in cand["value_raw"] or "$" in cand["value_raw"]),
        }
        rows.append((row, value_clean))  # tuple of features + original value

    if not rows:
        return None

    X = pd.DataFrame([r[0] for r in rows])
    values = [r[1] for r in rows]

    probs = model.predict_proba(X)[:, 1]  # Wahrscheinlichkeit für Klasse 1
    best_idx = probs.argmax()
    return round(values[best_idx], 2)

def predict_best_candidate_with_debug(raw_row: dict, model_path: str = "models/knn_model.pkl", top_n: int = 5):
    model = joblib.load(model_path)
    candidates = extract_price_candidates(raw_row.get("content_html", ""), raw_row.get("xhrs", {}))

    rows = []
    for cand in candidates:
        value_clean = clean_price_user(cand["value_raw"])
        if value_clean is None:
            continue

        row = {
            "value_clean": value_clean,
            "depth": cand.get("depth", -1),
            "css_len": len(cand.get("css_class", "")),
            "has_currency": int("€" in cand["value_raw"] or "$" in cand["value_raw"]),
            "source": cand.get("source", "")
        }
        rows.append((row, value_clean, cand))

    if not rows:
        return None, []

    X = pd.DataFrame([r[0] for r in rows])
    values = [r[1] for r in rows]
    metadata = [r[2] for r in rows]

    probs = model.predict_proba(X)[:, 1]
    ranked = sorted(zip(values, probs, metadata), key=lambda x: -x[1])

    best = ranked[0]
    top = ranked[:top_n]

    return round(best[0], 2), top  # predicted price, full list of top candidates

