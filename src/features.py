import json
import re

def clean_price_user(value):
    if isinstance(value, (float, int)):
        return float(value)

    if not isinstance(value, str):
        return None

    import re
    cleaned = re.sub(r"[^\d,.\-]", "", value.strip())

    if ',' in cleaned and '.' in cleaned:
        if cleaned.rfind('.') < cleaned.rfind(','):
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '.')

    try:
        return float(cleaned)
    except:
        return None
    
def extract_features(row):
    features = {}

    html = row.get("content_html") or ""
    dom = row.get("snapshot_dom") or ""
    xhrs = row.get("xhrs") or {}
    lang = row.get("user_language") or ""
    ua = row.get("user_agent") or ""

    # HTML-basiert
    features["html_len"] = len(html)
    features["num_currency_tokens"] = len(re.findall(r"(€|\$|USD|EUR)", html))
    features["num_price_patterns"] = len(re.findall(r"\d{1,3}([.,]\d{3})*([.,]\d{2})?", html))

    # DOM
    features["dom_len"] = len(dom)
    features["dom_price_patterns"] = len(re.findall(r"\d{1,3}([.,]\d{3})*([.,]\d{2})?", dom))

    # XHRs (string to dict if necessary)
    if isinstance(xhrs, str):
        try:
            xhrs = json.loads(xhrs)
        except:
            xhrs = {}
    xhr_str = json.dumps(xhrs)
    features["xhrs_price_keys"] = sum(1 for k in xhr_str.split('"') if "price" in k.lower())

    # Optionen
    features["lang_de"] = int(lang.startswith("de"))
    features["ua_mobile"] = int("mobile" in ua.lower())

    return features


def flatten_sample(raw):
    """
    Wandelt einen einzelnen Rohdatensatz (Dict) in einen Feature-Vektor (Dict) um.
    """
    features = {}

    # Beispiel: Seiten-Features
    features['page_title_length'] = len(raw.get('page_title', ''))
    features['url_length'] = len(raw.get('url', ''))
    features['lang'] = raw.get('user_language', 'unknown')

    # Produktoptionen flach machen
    product_options = raw.get('product_options', [])
    if isinstance(product_options, list):
        for option in product_options:
            key = f"option_{option.get('name', '').lower()}"
            val = option.get('value', '')
            if key:  # Avoid empty keys
                features[key] = val

    # Preis-Text als String (kannst du später zum Parsen oder als Feature nehmen)
    price_text = raw.get('price_element', {}).get('text', '')
    features['price_text_len'] = len(price_text)
    features['price_contains_eur'] = '€' in price_text

    # etc... alles was als Feature taugt, reinbauen!

    return features

def flatten_jsonl_file(input_path, output_path_features, output_path_labels):
    """
    Liest eine JSONL-Datei mit Rohdaten ein und speichert Features und Labels als CSV.
    """
    import pandas as pd

    rows = []
    labels = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw = json.loads(line)
            features = flatten_sample(raw)
            rows.append(features)
            # Target: price_confirmed (sofern vorhanden)
            labels.append(raw.get('price_confirmed', None))

    df = pd.DataFrame(rows)
    df.to_csv(output_path_features, index=False)
    pd.Series(labels, name="price").to_csv(output_path_labels, index=False)
