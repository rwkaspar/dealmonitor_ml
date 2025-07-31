import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import shutil
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler

logger = logging.getLogger(__name__)

def train_nn_model(
    data_path: str = "data/knn_training_set.parquet",
    model_path: str = "models/nn_model.pkl"
):
    # Load feature set
    df = pd.read_parquet(data_path)
    logger.info(df.info())

    # Define feature columns
    feature_cols = [
        "value_clean",       # Extracted price value
        "depth",             # DOM depth
        "css_len",           # Length of CSS selector
        "has_currency",      # Whether value contains currency symbols
    ]

    # Ensure all feature columns are numeric and clip outliers
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["value_clean"] = df["value_clean"].clip(lower=0, upper=2147483647)
    df["depth"] = df["depth"].clip(lower=0, upper=50)
    df["css_len"] = df["css_len"].clip(lower=0, upper=200)

    # Drop rows with invalid feature values
    df = df.dropna(subset=feature_cols)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    logger.debug("Any NaN? %s", df[feature_cols].isnull().any())
    logger.debug("Any inf? %s", np.isinf(df[feature_cols].values).any())
    logger.debug(df[feature_cols].describe())

    # Convert target column to binary integers
    df["match_with_user"] = df["match_with_user"].astype(int)
    logger.debug(df["match_with_user"].value_counts())

    # Perform train/test split at the raw_data_id level to avoid data leakage
    raw_data_ids = df["raw_data_id"].unique()
    train_ids, test_ids = train_test_split(raw_data_ids, test_size=0.2, random_state=42)

    df_train = df[df["raw_data_id"].isin(train_ids)]
    df_test = df[df["raw_data_id"].isin(test_ids)]

    # Prepare training features and target
    X_train = df_train[feature_cols]
    y_train = df_train["match_with_user"]

    # Apply oversampling only to training data
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # Prepare test data (without oversampling)
    X_test = df_test[feature_cols]
    y_test = df_test["match_with_user"]

    # Define and train the neural network model
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="tanh",           # Alternatives: relu, logistic
        solver="adam",
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.001,
        verbose=True,
        # early_stopping=True,
        # n_iter_no_change=10,
    )

    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate model performance on unseen test data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ F1 Score: {f1:.3f}")

    # Get predicted probabilities for class 1 (is correct candidate)
    probs = model.predict_proba(X_test)[:, 1]

    # Build DataFrame to combine features and predictions
    # Merge proba into original test DataFrame
    df_eval = df_test.copy().reset_index(drop=True)
    df_eval["proba"] = probs

    # Sanity check: Ensure 'price_user' is still present and consistent per group
    assert df_eval.groupby("raw_data_id")["price_user"].nunique().max() == 1, "Inconsistent price_user per group"


    # Top-K evaluation per raw_data_id
    top1_correct = 0
    top3_correct = 0
    total = 0

    for raw_data_id, group in df_eval.groupby("raw_data_id"):
        # Sort candidates by predicted probability (descending)
        sorted_group = group.sort_values("proba", ascending=False)

        expected_price = sorted_group["price_user"].iloc[0]  # same for all rows in group

        # Top-1
        top1_value = sorted_group["value_clean"].iloc[0]
        if np.isclose(top1_value, expected_price, atol=0.01):
            top1_correct += 1

        # Top-3
        top3_values = sorted_group["value_clean"].iloc[:3].values
        if np.any(np.isclose(top3_values, expected_price, atol=0.01)):
            top3_correct += 1

        total += 1

    # Final scores
    top1_acc = top1_correct / total
    top3_acc = top3_correct / total

    print(f"🎯 Top-1 Accuracy: {top1_acc:.3f} ({top1_correct} of {total})")
    print(f"🎯 Top-3 Accuracy: {top3_acc:.3f} ({top3_correct} of {total})")

    # Save final model
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # Save a timestamped copy
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    versioned_model_path = model_path.replace(".pkl", f"_{timestamp}.pkl")
    shutil.copy(model_path, versioned_model_path)

    # Update symlink to latest model
    symlink_path = os.path.join(os.path.dirname(model_path), "nn_model_latest.pkl")
    try:
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(os.path.abspath(model_path), symlink_path)
    except Exception as e:
        print(f"⚠️ Could not update symlink: {e}")

    print(f"✅ Versioned model saved as {versioned_model_path}")
    print(f"🔗 Symlink updated: {symlink_path}")
