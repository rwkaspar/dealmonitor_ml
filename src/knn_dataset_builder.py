import sys
import os
import logging
from typing import List, Dict
# from .candidate_extractor import extract_price_candidates

# from .features import clean_price_user

sys.path.append(os.path.abspath("dealmonitor/backend/src"))
from dealmonitor.features.features import clean_price
from dealmonitor.price_logic.candidate_extractor import extract_price_candidates
from dealmonitor.utils import extract_domain_from_url, get_shop_id_by_domain
from dealmonitor.database import get_db_session

logger = logging.getLogger(__name__)

def build_knn_training_rows(raw_row: dict) -> List[Dict]:
    """
    Nimmt einen Datensatz aus raw_data.jsonl und gibt Trainingsbeispiele für KNN zurück.
    """
    content_html = raw_row.get("content_html", "")
    xhrs = raw_row.get("xhrs", {})
    price_user = raw_row.get("price_user", None)
    url = raw_row.get("url", "")  # to get the domain
    domain = extract_domain_from_url(url)
    shop_id = get_shop_id_by_domain(get_db_session(), domain)
    # TODO use ID of shop instead of domain! This should make training easier.

    # price_user_clean = clean_price_user(price_user)
    price_user_clean = clean_price(price_user)

    if price_user_clean is None:
        return []

    candidates = extract_price_candidates(content_html, xhrs, url)
    result = []

    for cand in candidates:
        # value_clean = clean_price_user(cand["value_raw"])
        value_clean = clean_price(cand["value_raw"])
        if value_clean is None:
            continue

        match = abs(value_clean - price_user_clean) < 0.01  # Toleranz bei Float-Vergleich

        row = {
            "raw_data_id": raw_row.get("id", ""),
            "shop_id": shop_id,
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
