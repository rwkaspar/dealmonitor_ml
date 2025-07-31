import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import shutil
import logging
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler



logger = logging.getLogger(__name__)

def add_advanced_features(df):
    # Regex for currency symbols and 3-letter codes
    currency_symbols = r"[€$£¥₫₡₹₽₴₺₦₱₲₵₭₮₩₪₸₽]"
    currency_codes = r"\b([A-Z]{3})\b"

    # Clean up value_clean before feature creation
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["value_clean", "depth", "css_len"])

    # Number of digits (helps with currencies with large numbers)
    df["digits_count"] = df["value_clean"].apply(lambda x: len(str(int(abs(x)))) if pd.notnull(x) else 0)

    # Has 3-letter currency code (USD, VND, CRC etc.)
    df["currency_code_len3"] = df["tag"].str.upper().str.contains(currency_codes).astype(int) | \
                               df["css_len"].astype(str).str.upper().str.contains(currency_codes).astype(int)

    # Currency symbol left/right of the number
    def has_symbol_left(s):
        return int(bool(re.search(rf"{currency_symbols}\s*\d", str(s))))
    def has_symbol_right(s):
        return int(bool(re.search(rf"\d+\s*{currency_symbols}", str(s))))

    df["has_symbol_left"] = df["tag"].apply(has_symbol_left)
    df["has_symbol_right"] = df["tag"].apply(has_symbol_right)

    # Has decimal separator
    df["has_decimal_separator"] = df["value_clean"].apply(lambda x: int(('.' in str(x)) or (',' in str(x))) if pd.notnull(x) else 0)

    # Is integer
    df["is_integer"] = df["value_clean"].apply(lambda x: int(float(x).is_integer()) if pd.notnull(x) else 0)

    # Is <span> or <price> tag
    df["tag_span_price"] = df["tag"].apply(lambda x: int(str(x).lower() in ['span', 'price']))

    # Outlier flag (for very large/small values)
    df["outlier_flag"] = df["value_clean"].apply(lambda x: int(x < 0.1 or x > 2_000_000_000) if pd.notnull(x) else 0)

    return df

def train_nn_model(
    data_path: str = "data/knn_training_set.parquet",
    model_path: str = "models/nn_model.pkl"
):
    df = pd.read_parquet(data_path)
    logger.info(df.info())
    
    # Add extra features
    df = add_advanced_features(df)

    # Pick your final features here!
    feature_cols = [
        "value_clean",         # normalized price value
        "depth",               # DOM depth
        "css_len",             # Length of CSS-Class
        "has_currency",        # Contains € or $
        "currency_code_len3",  # Contains a 3-letter currency code
        "digits_count",        # Number of digits
        "has_symbol_left",     # Symbol on left of number
        "has_symbol_right",    # Symbol on right of number
        "has_decimal_separator", # Has decimal separator
        "is_integer",          # Is the value an integer
        "tag_span_price",      # Is tag span or price
        "outlier_flag",        # Outlier flag
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X["value_clean"] = X["value_clean"].clip(lower=0, upper=2_147_483_647)
    X["depth"] = X["depth"].clip(lower=0, upper=50)
    X["css_len"] = X["css_len"].clip(lower=0, upper=200)
    X["digits_count"] = X["digits_count"].clip(lower=0, upper=20)
    X = X.apply(pd.to_numeric, errors="coerce")
    logger.debug("Any NaN? %s", X.isnull().any())
    logger.debug("Any inf? %s", np.isinf(X.values).any())
    logger.debug(X.describe())
    
    y = df["match_with_user"].astype(int)
    logger.debug(y.value_counts())

    X = X.dropna()
    y = y.loc[X.index]

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y) # type: ignore

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42
    )

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64, 32),
        activation="tanh",
        solver="adam",
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.001,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ F1 Score: {f1:.3f}")

    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}")

    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    versioned_model_path = model_path.replace(".pkl", f"_{timestamp}.pkl")
    shutil.copy(model_path, versioned_model_path)

    # Update symlink to latest
    symlink_path = os.path.join(os.path.dirname(model_path), "nn_model_latest.pkl")
    try:
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(os.path.abspath(model_path), symlink_path)
    except Exception as e:
        print(f"⚠️ Could not update symlink: {e}")

    print(f"✅ Versioned model saved as {versioned_model_path}")
    print(f"🔗 Symlink updated: {symlink_path}")
