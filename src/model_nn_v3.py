import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import shutil
import logging
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler

from .features import extract_price_features  # use your features.py

logger = logging.getLogger(__name__)

MODEL_LATEST = "models/model_latest.pkl"
MODEL_BEST = "models/model_best.pkl"


def build_feature_df(df=pd.DataFrame()) -> pd.DataFrame:
    """
    Apply extract_price_features from features.py row-wise to each candidate to get final ML feature DataFrame.
    """
    features = df.apply(lambda row: extract_price_features(row.to_dict()), axis=1)
    features_df = pd.DataFrame(list(features))
    return features_df


def train_nn_model(
    data_path: str = "data/knn_training_set.parquet",
    model_path: str = "models/nn_model.pkl"
):
    df = pd.read_parquet(data_path)
    logger.info(df.info())

    # Extract features and keep only valid rows
    X_full = build_feature_df(df)
    X_full = X_full.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.loc[X_full.index].reset_index(drop=True)
    X_full = X_full.reset_index(drop=True)
    y = df["match_with_user"].astype(int)

    # Combine metadata for later evaluation
    X_full["raw_data_id"] = df["raw_data_id"].values
    X_full["price_user"] = df["price_user"].values
    X_full["value_clean"] = df["value_clean"].values

    # Save meta columns and drop before training
    meta_cols = ["raw_data_id", "price_user", "value_clean"]
    meta_full = X_full[meta_cols]
    X_features = X_full.drop(columns=meta_cols)

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_features, y)

    # ✅ Fix: use sample_indices_ to resample metadata safely
    meta_resampled = meta_full.iloc[ros.sample_indices_].reset_index(drop=True)

    X_resampled = X_resampled.reset_index(drop=True)
    y_resampled = y_resampled.reset_index(drop=True)

    # Train/test split by raw_data_id to avoid leakage
    unique_ids = meta_resampled["raw_data_id"].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    train_mask = meta_resampled["raw_data_id"].isin(train_ids)
    test_mask = meta_resampled["raw_data_id"].isin(test_ids)

    X_train, y_train = X_resampled[train_mask], y_resampled[train_mask]
    print(X_train.describe())
    X_test, y_test = X_resampled[test_mask], y_resampled[test_mask]
    meta_test = meta_resampled[test_mask].copy()

    # Train the model
    # model = MLPClassifier(
    #     hidden_layer_sizes=(128, 64, 32),
    #     activation="tanh",
    #     solver="adam",
    #     max_iter=1000,
    #     random_state=42,
    #     learning_rate_init=0.001,
    #     verbose=True
    # )

    # model = RandomForestClassifier(
    #     n_estimators=2000,
    #     max_depth=50,
    #     class_weight="balanced",
    #     random_state=42
    # )

    model = HistGradientBoostingClassifier(
        max_iter=5000,
        early_stopping=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Eval: Accuracy & F1
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ F1 Score: {f1:.3f}")

    # Eval: Top-K Accuracy
    probs = model.predict_proba(X_test)[:, 1]
    meta_test = meta_test.reset_index(drop=True)
    meta_test["proba"] = probs

    top1_correct = 0
    top3_correct = 0
    total = 0

    for raw_data_id, group in meta_test.groupby("raw_data_id"):
        sorted_group = group.sort_values("proba", ascending=False)
        expected_price = sorted_group["price_user"].iloc[0]

        # Top-1
        top1_value = sorted_group["value_clean"].iloc[0]
        if np.isclose(top1_value, expected_price, atol=0.01):
            top1_correct += 1

        # Top-3
        top3_values = sorted_group["value_clean"].iloc[:3].values
        if np.any(np.isclose(top3_values, expected_price, atol=0.01)): # type: ignore
            top3_correct += 1

        total += 1

    print(f"🎯 Top-1 Accuracy: {top1_correct / total:.3f} ({top1_correct} of {total})")
    print(f"🎯 Top-3 Accuracy: {top3_correct / total:.3f} ({top3_correct} of {total})")

    # Save base model
    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}")

    # Check for best model so far
    best_model_path = os.path.join(os.path.dirname(model_path), "model_best.pkl")
    update_best = True

    if os.path.exists(best_model_path):
        try:
            best_model = joblib.load(best_model_path)
            # Dummy X to get f1 of old model
            y_pred_best = best_model.predict(X_test)
            f1_best = f1_score(y_test, y_pred_best)
            if f1 <= f1_best:
                update_best = False
                print(f"ℹ️ Best model kept (f1 {f1_best:.3f} ≥ {f1:.3f})")
        except Exception as e:
            print(f"⚠️ Could not evaluate existing best model: {e}")

    if update_best:
        joblib.dump(model, best_model_path)
        print(f"🏆 New best model saved at {best_model_path}")
        subprocess.run(["dvc", "add", "models/model_best.pkl"])
        subprocess.run(["git", "add", "models/model_best.pkl.dvc", ".gitignore"])
        commit = 'Add/update best and latest model'
    else:
        commit = 'Add/update latest model'

    # Save versioned model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    versioned_model_path = model_path.replace(".pkl", f"_{timestamp}.pkl")
    shutil.copy(model_path, versioned_model_path)

    # Update symlink
    symlink_path = os.path.join(os.path.dirname(model_path), "model_latest.pkl")
    try:
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(os.path.abspath(model_path), symlink_path)
    except Exception as e:
        print(f"⚠️ Could not update symlink: {e}")
    subprocess.run(["dvc", "add", "models/model_latest.pkl"])
    subprocess.run(["git", "add", "models/model_latest.pkl.dvc", ".gitignore"])
    subprocess.run(["git", "commit", "-m", commit])
    subprocess.run(["dvc", "push"])

    print(f"✅ Versioned model saved as {versioned_model_path}")
    print(f"🔗 Symlink updated: {symlink_path}")

    # Append training result to log CSV
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": type(model).__name__,
        "params": str(model.get_params()),
        "accuracy": round(acc, 3),
        "f1_score": round(f1, 3),
        "top1_acc": round(top1_correct / total, 3),
        "top3_acc": round(top3_correct / total, 3),
        "raw_data_count": df["raw_data_id"].nunique(),
        "model_path": versioned_model_path
    }

    log_path = os.path.join(os.path.dirname(model_path), "training_log.csv")
    log_df = pd.DataFrame([log_entry])

    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_path, mode='w', header=True, index=False)

    print(f"📝 Training log updated: {log_path}")


if __name__ == "__main__":
    train_nn_model()
