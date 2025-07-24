import re
import json
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_price_candidates(content_html: str, xhrs: Any = None) -> List[Dict]:
    """
    Findet preisähnliche Werte in HTML und XHRs und gibt strukturierte Kandidaten zurück
    """
    candidates = []

    # 🟡 HTML-Kandidaten via BeautifulSoup
    if content_html:
        soup = BeautifulSoup(content_html, "html.parser")

        # Alle Textnodes mit Preisen
        price_pattern = re.compile(r"\d{1,4}([.,]\d{3})*([.,]\d{2})")  # z. B. 1.399,00

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
