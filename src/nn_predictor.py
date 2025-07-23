import pandas as pd
import joblib
from pathlib import Path
from .candidate_extractor import extract_price_candidates
from .features import clean_price_user

MODEL_PATH = Path(__file__).parent.parent / "models" / "nn_model.pkl"

def predict_best_candidate_nn(raw_row: dict, model_path: str = MODEL_PATH, top_n: int = 5):
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

    best_price = round(best[0], 2)
    best_meta = best[2]

    result = {
        "price": best_price,
        "score": round(best[1], 3),
        "outer_html": best_meta.get("outer_html"),
        "tag": best_meta.get("tag"),
        "css_class": best_meta.get("css_class"),
        "source": best_meta.get("source"),
        "selector": best_meta.get("selector"),
    }

    return result, top
