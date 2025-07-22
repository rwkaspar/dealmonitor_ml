from typing import List, Dict
from src.candidate_extractor import extract_price_candidates
from src.features import clean_price_user


def build_knn_training_rows(raw_row: dict) -> List[Dict]:
    """
    Nimmt einen Datensatz aus raw_data.jsonl und gibt Trainingsbeispiele für KNN zurück.
    """
    content_html = raw_row.get("content_html", "")
    xhrs = raw_row.get("xhrs", {})
    price_user = raw_row.get("price_user", None)
    price_user_clean = clean_price_user(price_user)

    if price_user_clean is None:
        return []

    candidates = extract_price_candidates(content_html, xhrs)
    result = []

    for cand in candidates:
        value_clean = clean_price_user(cand["value_raw"])
        if value_clean is None:
            continue

        match = abs(value_clean - price_user_clean) < 0.01  # Toleranz bei Float-Vergleich

        row = {
            "source": cand.get("source"),
            "value_clean": value_clean,
            "match_with_user": int(match),
            "depth": cand.get("depth", -1),
            "tag": cand.get("tag", ""),
            "css_len": len(cand.get("css_class", "")),
            "has_currency": int("€" in cand["value_raw"] or "$" in cand["value_raw"]),
            "price_user": price_user_clean,
        }

        result.append(row)

    return result
