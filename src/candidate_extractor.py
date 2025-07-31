import re
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Union


def extract_all_candidates(raw_data: Union[List[Dict], "pd.DataFrame"]) -> List[Dict]:
    """
    Extract all price candidates from raw_data (either list of dicts or DataFrame).
    Each candidate will include: value_raw, source, outer_html, tag, css_class, depth, raw_data_id, created_at
    """
    if hasattr(raw_data, "to_dict"):
        raw_data = raw_data.to_dict(orient="records")

    candidates = []

    for row in raw_data:
        html = row.get("content_html", "")
        xhrs = row.get("xhrs")
        raw_data_id = row.get("id")
        created_at = row.get("created_at")
        fingerprint = row.get("fingerprint")
        price_user = row.get("price_user")

        found = extract_price_candidates(html, xhrs)

        for c in found:
            c["raw_data_id"] = raw_data_id
            c["created_at"] = created_at
            c["fingerprint"] = fingerprint
            c["price_user"] = price_user
            candidates.append(c)

    return candidates

def extract_price_candidates(content_html: str, xhrs: Any = None) -> List[Dict]:
    """
    Findet preisähnliche Werte in HTML und XHRs und gibt strukturierte Kandidaten zurück
    """
    candidates = []

    # 🟡 HTML-Kandidaten via BeautifulSoup
    if content_html:
        soup = BeautifulSoup(content_html, "html.parser")

        # All text nodes containing prices
        price_pattern = re.compile(
            r"""
            (
                (?:[A-Z]{3}\s*)?
                [€$£¥₹₽₩₪฿₺₫₴₦₲₵₡₢₳₭₮₰₱₲₳₤₥₧₯₶₸₺₻₼₽₾₿₠₡₢₣₤₥₦₧₨₩₪₫₭₮₯₰₱₲₳₴₵₶₷₸₺₻₼₽₾₿]?
                \s*[-]?
                (?:\d{1,3}([.,'\s]\d{3})*|\d+)
                ([.,]\d{2})?
                \s*[€$£¥₹₽₩₪฿₺₫₴₦₲₵₡₢₳₭₮₰₱₲₳₤₥₧₯₶₸₺₻₼₽₾₿]?
                (?:\s*[A-Z]{3})?
            )
            """,
            re.VERBOSE
        )

        for el in soup.find_all(text=price_pattern):
            text = el.strip() # type: ignore
            if not text:
                continue

            outer_html = str(el.parent)
            tag = el.parent.name # type: ignore
            css_class = " ".join(el.parent.get("class", [])) # type: ignore
            depth = len(list(el.parents))  # tiefe im DOM

            candidates.append({
                "source": "html",
                "value_raw": text,
                "tag": tag,
                "css_class": css_class,
                "outer_html": outer_html,
                "depth": depth,
            })

    # 🔵 XHR-Kandidaten aus JSON
    if isinstance(xhrs, str):
        try:
            xhrs = json.loads(xhrs)
        except Exception:
            xhrs = {}

    def extract_from_json(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                extract_from_json(v, path + "." + k if path else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                extract_from_json(item, f"{path}[{i}]")
        else:
            if isinstance(obj, (int, float)) or (
                isinstance(obj, str) and re.fullmatch(r"\d{1,4}([.,]\d{3})*([.,]\d{2})", obj)
            ):
                candidates.append({
                    "source": "xhrs",
                    "value_raw": str(obj),
                    "json_path": path,
                })

    extract_from_json(xhrs)

    return candidates
