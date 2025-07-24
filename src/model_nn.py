import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import shutil
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler

logger = logging.getLogger(__name__)

def train_nn_model(
    data_path: str = "data/knn_training_set.parquet",
    model_path: str = "models/nn_model.pkl"
):
    df = pd.read_parquet(data_path)

    # logger.info(df.info())
    
    feature_cols = [
        "value_clean",       # price value
        "depth",             # DOM depth
        "css_len",           # lenth of CSS-Class
        "has_currency",      # contains € or $
    ]

    for col in feature_cols:
        # cast all values to numeric, coercing errors
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X["value_clean"] = X["value_clean"].clip(lower=0, upper=2147483647)
    X["depth"] = X["depth"].clip(lower=0, upper=50)
    X["css_len"] = X["css_len"].clip(lower=0, upper=200)
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
        hidden_layer_sizes=(128, 64, 32),
        activation="tanh",  # for more data relu, alternative: logistic
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
    print(f"✅ Modell gespeichert unter {model_path}")

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