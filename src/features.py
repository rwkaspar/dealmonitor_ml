import json

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
