# moved to backend

import json
import re
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

MAX_PRICE = 1e9  # This should be high enough for all currencies (e.g., VND, CRC)
MAX_DEPTH = 100  # Sensible upper bound for DOM depth
MAX_CSS_LEN = 1000  # Sensible upper bound for CSS class length

def clean_price_user(value):
    """
    Converts a price value (str, float, int) to float, handling various formats and invalid values.
    """
    # Accept numeric values directly
    if isinstance(value, (float, int)):
        if np.isnan(value) or not np.isfinite(value):
            return None
        return float(value)
    if not isinstance(value, str):
        return None

    # Remove all known non-numeric symbols, common currency, spaces, apostrophes, narrow spaces
    value = value.strip()
    value = value.replace("'", "").replace('\u202f', '').replace('\xa0', '')  # remove various spaces
    cleaned = re.sub(r"[^\d,.\-]", "", value)
    # Handle cases with both . and ,
    if ',' in cleaned and '.' in cleaned:
        if cleaned.rfind('.') < cleaned.rfind(','):
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '.')
    try:
        v = float(cleaned)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None

def extract_price_features(row: dict) -> dict:
    """
    Extract ML features from a raw_data candidate row.
    Assumes 'row' contains at least: value_clean, depth, tag, source, price_user
    """
    source = str(row.get("source", "") or "")
    tag = str(row.get("tag", "") or "")
    outer_html = str(row.get("outer_html", "") or "")

    value_clean = float(row.get("value_clean", 0.0))
    # price_user = float(row.get("price_user", 0.0))
    if row.get("target") is None:
        price_user = clean_price_user(row.get("price_user", 0.0))
    else:
        price_user = float(row.get("target", 0.0))

    # Textual features
    contains_currency = bool(re.search(
            r"[€$₫₩¥₹₽₴₺₦₱₲₵₸₡₲₽złkr₪₭₫៛₮₦₩₭₽₽₽₽₡A\$₣₧₨₤₥₦₧₨₩₭₮₯₰₱₲₳₴₵₸₺₼₽₾₿]|[A-Z]{3}", 
            str(row.get("value_clean", "")), 
            re.IGNORECASE))
    contains_comma = "," in source
    contains_dot = "." in source
    contains_discount_word = bool(re.search(r"(sale|angebot|rabatt|deal|reduziert)", source, flags=re.I))
    contains_strike_word = bool(re.search(r"(old|strike|durchgestrichen|uvp)", source, flags=re.I))

    # Structural / semantic
    tag_lower = tag.lower()
    tag_is_span = tag_lower == "span"
    tag_is_div = tag_lower == "div"
    tag_is_ins = tag_lower == "ins"
    tag_is_del = tag_lower == "del"

    # Relative price info
    # price_diff_abs = abs(value_clean - price_user)
    # price_diff_ratio = price_diff_abs / price_user if price_user > 0 else 0.0

    # Other
    selector_length = int(row.get("css_len", 0))
    depth_raw = row.get("depth", 0)
    if math.isnan(depth_raw):
        depth = 0
    else:
        depth = int(depth_raw)

    return {
        "value_clean": value_clean,
        "depth": depth,
        "css_len": selector_length,
        "has_currency": int(contains_currency),
        "contains_comma": int(contains_comma),
        "contains_dot": int(contains_dot),
        "discount_word": int(contains_discount_word),
        "strike_word": int(contains_strike_word),
        "tag_is_span": int(tag_is_span),
        "tag_is_div": int(tag_is_div),
        "tag_is_ins": int(tag_is_ins),
        "tag_is_del": int(tag_is_del),
        # "price_diff_abs": price_diff_abs,
        # "price_diff_ratio": price_diff_ratio
    }


def extract_price_features_old(cand):
    """
    Extracts robust features from a single price candidate dict (used for ML models).
    """
    val = clean_price_user(cand.get("value_clean"))
    features = {
        # Clamp value_clean to valid price range
        "value_clean": np.clip(val if val is not None else 0, 0, MAX_PRICE),
        # Clamp DOM depth and CSS class length
        "depth": np.clip(cand.get("depth", -1), 0, MAX_DEPTH),
        "css_len": np.clip(len(cand.get("css_class", "")), 0, MAX_CSS_LEN),
        # Check for any known currency symbol or code
        "has_currency": int(bool(re.search(
            r"[€$₫₩¥₹₽₴₺₦₱₲₵₸₡₲₽złkr₪₭₫៛₮₦₩₭₽₽₽₽₡A\$₣₧₨₤₥₦₧₨₩₭₮₯₰₱₲₳₴₵₸₺₼₽₾₿]|[A-Z]{3}", 
            str(cand.get("value_clean", "")), 
            re.IGNORECASE))),
        # Count number of digits (proxy for magnitude and potential currency type)
        # "digits_count": len(str(int(abs(val)))) if val and np.isfinite(val) else 0,
    }
    # Optionally, you can add more price-candidate-based features here
    return features

def extract_global_features(row):
    """
    Extracts features from the whole product context (HTML, DOM, XHR, language, UA).
    """
    html = row.get("content_html", "") or ""
    dom = row.get("snapshot_dom", "") or ""
    xhrs = row.get("xhrs") or {}
    lang = row.get("user_language", "") or ""
    ua = row.get("user_agent", "") or ""

    # Global statistics about the HTML content
    html_len = min(len(html), 1_000_000)
    num_currency = len(re.findall(r"(€|\$|USD|EUR|VND|CRC|JPY|[A-Z]{3})", html, re.IGNORECASE))
    num_prices = len(re.findall(r"\d{1,3}([.,]\d{3})*([.,]\d{2})?", html))
    dom_len = min(len(dom), 1_000_000)
    dom_prices = len(re.findall(r"\d{1,3}([.,]\d{3})*([.,]\d{2})?", dom))

    # Convert XHRs to dict if necessary
    if isinstance(xhrs, str):
        try:
            xhrs = json.loads(xhrs)
        except Exception:
            xhrs = {}
    xhr_str = json.dumps(xhrs)
    xhr_price_keys = sum(1 for k in xhr_str.split('"') if "price" in k.lower())

    # Language and user-agent signals
    lang_de = int(lang.startswith("de"))
    ua_mobile = int("mobile" in ua.lower())

    return {
        "html_len": html_len,
        "num_currency_tokens": num_currency,
        "num_price_patterns": num_prices,
        "dom_len": dom_len,
        "dom_price_patterns": dom_prices,
        "xhrs_price_keys": xhr_price_keys,
        "lang_de": lang_de,
        "ua_mobile": ua_mobile,
    }

def flatten_sample(raw, cand=None):
    """
    Flattens a raw product row (and optionally a price candidate) into a single feature vector.
    Suitable for both ML training and inference.
    """
    features = extract_global_features(raw)
    if cand:
        features.update(extract_price_features(cand))
    # Example: add page title length, URL length, etc.
    features['page_title_length'] = len(raw.get('page_title', ''))
    features['url_length'] = len(raw.get('url', ''))
    features['lang'] = raw.get('user_language', 'unknown')
    # Flatten product options if present (optional)
    product_options = raw.get('product_options', [])
    if isinstance(product_options, list):
        for option in product_options:
            key = f"option_{option.get('name', '').lower()}"
            val = option.get('value', '')
            if key:  # Avoid empty keys
                features[key] = val
    # Price element text length and contains-euro flag
    price_text = raw.get('price_element', {}).get('text', '')
    features['price_text_len'] = len(price_text)
    features['price_contains_eur'] = '€' in price_text
    # Add any further custom features as needed!
    return features
