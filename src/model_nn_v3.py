import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tqdm import tqdm
import shutil
import logging
import subprocess
import dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
# from imbalance-learn import 
from imblearn.over_sampling import RandomOverSampler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import dagshub

# from .features import extract_price_features  # use your features.py

sys.path.append(os.path.abspath("dealmonitor/backend/src"))
from dealmonitor.features.features import extract_price_features

logger = logging.getLogger(__name__)

dotenv.load_dotenv()  # Load environment variables from .env file

MODEL_LATEST = "models/model_latest.pkl"
MODEL_BEST = "models/model_best.pkl"

# 1️⃣ MLflow / DagsHub Setup (hier kannst du auch dotenv nehmen)
os.environ["MLFLOW_TRACKING_URI"] = os.environ["DAGSHUB_REPO_URL"]
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
dagshub.init(repo_owner=os.environ["MLFLOW_TRACKING_USERNAME"], repo_name='dealmonitor_ml', mlflow=True)


def build_feature_df(df=pd.DataFrame()) -> pd.DataFrame:
    features = df.apply(lambda row: extract_price_features(row.to_dict()), axis=1)
    features_df = pd.DataFrame(list(features))
    return features_df


def train_nn_model(
    data_path: str = "data/knn_training_set.parquet",
    model_path: str = "models/nn_model.pkl"
):
    with mlflow.start_run(run_name="Train NN Model"):
        # Track Input-Data as Artefact
        mlflow.log_artifact(data_path)

        df = pd.read_parquet(data_path)
        # logger.info(df.info())

        # Feature Engineering
        X_full = build_feature_df(df)
        X_full = X_full.replace([np.inf, -np.inf], np.nan).dropna()
        df = df.loc[X_full.index].reset_index(drop=True)
        X_full = X_full.reset_index(drop=True)
        y = df["match_with_user"].astype(int)

        # # One-Hot-Encoding domain feature
        # if "domain" in X_full.columns:
        #     domain_dummies = pd.get_dummies(X_full["domain"], prefix="domain")
        #     X_full = X_full.drop(columns=["domain"])
        #     X_full = pd.concat([X_full, domain_dummies], axis=1)

        meta_cols = ["raw_data_id", "price_user", "value_clean"]
        X_full["raw_data_id"] = df["raw_data_id"].values
        X_full["price_user"] = df["price_user"].values
        X_full["value_clean"] = df["value_clean"].values
        meta_full = X_full[meta_cols]
        X_features = X_full.drop(columns=meta_cols)

        # Oversampling
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_features, y)
        meta_resampled = meta_full.iloc[ros.sample_indices_].reset_index(drop=True)
        X_resampled = X_resampled.reset_index(drop=True)
        y_resampled = y_resampled.reset_index(drop=True)

        # Train/Test Split
        unique_ids = meta_resampled["raw_data_id"].unique()
        train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
        train_mask = meta_resampled["raw_data_id"].isin(train_ids)
        test_mask = meta_resampled["raw_data_id"].isin(test_ids)
        X_train, y_train = X_resampled[train_mask], y_resampled[train_mask]
        X_test, y_test = X_resampled[test_mask], y_resampled[test_mask]
        meta_test = meta_resampled[test_mask].copy()

        X_test = X_test.reset_index(drop=True)
        meta_test = meta_test.reset_index(drop=True)

        # Modell & Params als Param loggen
        # model = HistGradientBoostingClassifier(
        #     max_iter=5000,
        #     early_stopping=True,
        #     random_state=42
        # )

# Try with many models and many options

        # Models
        from sklearn.neural_network import MLPClassifier
        from xgboost import XGBClassifier
        # from lightgbm import LGBMClassifier

        models = {
            "mlp": MLPClassifier(),
            "xgb": XGBClassifier(),
            # "lgbm": LGBMClassifier()
        }

        param_grids = {
            # Model: mlp, Best F1: 0.8441, Params: {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (128, 64), 'learning_rate_init': 0.001, 'max_iter': 5000, 'random_state': 42, 'solver': 'adam'}
            "mlp": {
                "hidden_layer_sizes": [(64,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
                # (32,)
                'activation': ['relu', 'tanh', 'logistic'],  # 'logistic'
                'solver': ['adam', 'sgd'],  # 'lbfgs', 'sgd'
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.0001, 0.001],  # 0.01
                "max_iter": [10000, 5000, 1000, 100],  # seams to have no impact
                "random_state": [42],
            },
            # Model: xgb, Best F1: 0.8267, Params: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 500, 'random_state': 42, 'subsample': 0.6}
            "xgb": {
                "n_estimators": [100, 250, 500, 1000],
                "learning_rate": [0.001, 0.01, 0.1],
                "max_depth": [5, 7, 10, 12, 15],
                "subsample": [0.4, 0.6, 0.8],
                "colsample_bytree": [0.8, 1.0, 1.2],
                "random_state": [42]
            },
            # "lgbm": {
            #     "n_estimators": [250],# 500],
            #     "learning_rate": [0.001],# 0.01, 0.05, 0.1],
            #     "max_depth": [-1, 3],# 5, 7, 10]
            #     "num_leaves": [31],
            #     "min_data_in_leaf": [20],
            #     "feature_fraction": [0.8],
            #     "bagging_fraction": [0.8],
            #     "bagging_freq": [1],
            #     "early_stopping_rounds": [50],
            #     "random_state": [42]
            # }
        }

        # ## TEST
        # param_grids = {
        #     "mlp": {
        #         "hidden_layer_sizes": [(128, 64)],
        #         'activation': ['tanh'],
        #         'solver': ['adam'],
        #         'alpha': [0.001],
        #         'learning_rate_init': [0.001],
        #         "max_iter": [5000],
        #         "random_state": [42],
        #     },
        #     "xgb": {
        #         "n_estimators": [500],
        #         "learning_rate": [0.1],
        #         "max_depth": [10],
        #         "subsample": [0.6],
        #         "colsample_bytree": [1.0],
        #         "random_state": [42]
        #     },
        # }

        from sklearn.model_selection import ParameterGrid

        results = {}
        for name, model in models.items():
            best_model = None
            # best_score_1 = 0
            best_score = 0
            best_params = None
            for params in tqdm(ParameterGrid(param_grids[name]), desc=f"Tuning {name}"):
                clf = clone(model).set_params(**params)
                clf.fit(X_train, y_train)

                # score_1 = top_k_accuracy(clf, X_test, meta_test, k=1)
                score = top_k_accuracy(clf, X_test, meta_test, k=3)

                print(f"✅ Model: {name}, Params: {params}")
                print(f"▶ Top-3 Accuracy: {score:.3f}")
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                print(f"▶ Accuracy: {acc:.3f}")
                print(f"▶ F1 Score: {f1:.3f}")

                # signature and input example for MLflow
                input_example = X_train.iloc[:5].copy()  # oder nimm eine Zeile, je nach Bedarf
                signature = infer_signature(input_example, clf.predict(input_example))

                if score > best_score:
                    best_score = score
                    best_params = params
                    # best_model = joblib.loads(joblib.dumps(clf))  # Deep copy
                    best_model = clf

            print(f"▶ Best {name} Top-3 Accuracy: {best_score:.3f} with Params: {best_params}")

            # Log best model and params to MLflow
            mlflow.log_params({f"{name}_"+k: v for k, v in best_params.items()})
            mlflow.log_metric(f"{name}_top3_accuracy", best_score)
            model_path = f"models/{name}_best.pkl"
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(model_path)

            results[name] = {"model": best_model, "score": best_score, "params": best_params}

            sorted_results = sorted(
                ((name, res["score"], res["params"]) for name, res in results.items()),
                key=lambda x: x[1],
                reverse=True
            )

        top_n = 10
        print(f"\nTop {top_n} Modelle nach Top-K Accuracy:")
        for rank, (name, score, params) in enumerate(sorted_results[:top_n], start=1):
            print(f"{rank:2d}. {name:8s} – Top-K Accuracy: {score:.3f} – Params: {params}")

        #     gs = GridSearchCV(
        #         model,
        #         param_grids[name],
        #         scoring="f1",
        #         cv=3,
        #         n_jobs=-1,
        #         verbose=2,
        #         refit=False  # We only want the best params, not the refitted model
        #     )
        #     gs.fit(X_train, y_train)
        #     results[name] = {
        #         "best_score": gs.best_score_,
        #         "best_params": gs.best_params_,
        #         "best_estimator": gs.best_estimator_
        #     }

        #     results_df = pd.DataFrame(gs.cv_results_)
        #     print(f"Full CV results for model {name}:")
        #     print(results_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]])

        #     print(f"▶ Model trained: {name}")
        #     print(f"  - Best Score (F1): {gs.best_score_:.4f}")
        #     print(f"  - Best Parameters: {gs.best_params_}")

        # print("\nTraining Summary:")
        # for name, res in results.items():
        #     print(f"- Model: {name}, Best F1: {res['best_score']:.4f}, Params: {res['best_params']}")

# Try with more options to one model

        # # Definiere das Basis-Modell
        # model = MLPClassifier(random_state=42, max_iter=10000)

        # # Definiere den Suchraum für Hyperparameter
        # param_grid = {
        #     'hidden_layer_sizes': [(64,), (128, 64), (256, 128, 64)],
        #     'activation': ['relu', 'tanh', 'logistic'],
        #     'solver': ['adam', 'sgd'],
        #     'alpha': [0.0001, 0.001],
        #     'learning_rate_init': [0.001, 0.01]
        # }

        # # Initialisiere Grid Search mit Cross-Validation (z.B. 3-fach CV)
        # grid_search = GridSearchCV(
        #     estimator=model,
        #     param_grid=param_grid,
        #     scoring='f1',
        #     n_jobs=-1,
        #     cv=3,
        #     verbose=2
        # )

        # # X_train, y_train sind deine Trainingsdaten (bereits vorbereitet)
        # grid_search.fit(X_train, y_train)

        # print("Best parameters set found on development set:")
        # print(grid_search.best_params_)

        # print("Best CV score:")
        # print(grid_search.best_score_)

        # best_model = grid_search.best_estimator_

# Models: NN, XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, AdaBoost, HistGradientBoosting
        
        # model = MLPClassifier(
        #     hidden_layer_sizes=(256, 128, 64, 32),
        #     activation="tanh",
        #     solver="adam",
        #     max_iter=10000,
        #     random_state=42,
        #     learning_rate_init=0.001,
        # )

        # from xgboost import XGBClassifier
        # model = XGBClassifier(
        #     n_estimators=1000,
        #     learning_rate=0.01,
        #     max_depth=6,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     reg_alpha=0.1,
        #     reg_lambda=1,
        #     random_state=42,
        #     # early_stopping_rounds=50
        # )


        # from xgboost import XGBClassifier
        # model = XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=42)

        # from lightgbm import LGBMClassifier
        # model = LGBMClassifier(n_estimators=500, random_state=42)

        # from catboost import CatBoostClassifier
        # model = CatBoostClassifier(iterations=500, random_state=42, verbose=0)

        # from sklearn.ensemble import RandomForestClassifier
        # model = RandomForestClassifier(n_estimators=500, random_state=42)

        # from sklearn.ensemble import ExtraTreesClassifier
        # model = ExtraTreesClassifier(n_estimators=500, random_state=42)

        # from sklearn.ensemble import AdaBoostClassifier
        # model = AdaBoostClassifier(n_estimators=500, random_state=42)

        # mlflow.log_params(model.get_params())

        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # acc = accuracy_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        # print(f"✅ Accuracy: {acc:.3f}")
        # print(f"✅ F1 Score: {f1:.3f}")
        # mlflow.log_metric("accuracy", acc)
        # mlflow.log_metric("f1_score", f1)

        # # signature and input example for MLflow
        # input_example = X_train.iloc[:5].copy()  # oder nimm eine Zeile, je nach Bedarf
        # signature = infer_signature(input_example, model.predict(input_example))

        # # Top-K Accuracy
        # probs = model.predict_proba(X_test)[:, 1]
        # meta_test = meta_test.reset_index(drop=True)
        # meta_test["proba"] = probs
        # top1_correct = 0
        # top3_correct = 0
        # total = 0
        # for raw_data_id, group in meta_test.groupby("raw_data_id"):
        #     sorted_group = group.sort_values("proba", ascending=False)
        #     expected_price = sorted_group["price_user"].iloc[0]
        #     top1_value = sorted_group["value_clean"].iloc[0]
        #     if np.isclose(top1_value, expected_price, atol=0.01):
        #         top1_correct += 1
        #     top3_values = sorted_group["value_clean"].iloc[:3].values
        #     if np.any(np.isclose(top3_values, expected_price, atol=0.01)):
        #         top3_correct += 1
        #     total += 1
        # top1_acc = top1_correct / total
        # top3_acc = top3_correct / total
        # mlflow.log_metric("top1_accuracy", top1_acc)
        # mlflow.log_metric("top3_accuracy", top3_acc)
        # print(f"🎯 Top-1 Accuracy: {top1_acc:.3f} ({top1_correct} of {total})")
        # print(f"🎯 Top-3 Accuracy: {top3_acc:.3f} ({top3_correct} of {total})")

        # # Modell speichern & als Artefakt loggen
        # joblib.dump(model, model_path)
        # mlflow.set_tag("model_version", "latest")
        # mlflow.set_tag("model_path", "models/model_latest.pkl")
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     registered_model_name="DealMonitorNN",
        #     artifact_path="sklearn-model",
        #     signature=signature,
        #     input_example=input_example
        # )
        # mlflow.log_artifact(model_path)
        # print(f"✅ Model saved at {model_path}")

        # # DVC-Bestmodell verwalten wie gehabt
        # best_model_path = os.path.join(os.path.dirname(model_path), "model_best.pkl")
        # update_best = True
        # if os.path.exists(best_model_path):
        #     try:
        #         best_model = joblib.load(best_model_path)
        #         y_pred_best = best_model.predict(X_test)
        #         f1_best = f1_score(y_test, y_pred_best)
        #         if f1 <= f1_best:
        #             update_best = False
        #             print(f"ℹ️  Best model kept (f1 {f1_best:.3f} ≥ {f1:.3f})")
        #     except Exception as e:
        #         print(f"⚠️ Could not evaluate existing best model: {e}")

        # if update_best:
        #     joblib.dump(model, best_model_path)
        #     print(f"🏆 New best model saved at {best_model_path}")
        #     subprocess.run(["dvc", "add", "models/model_best.pkl"])
        #     subprocess.run(["git", "add", "models/model_best.pkl.dvc", ".gitignore"])
        #     commit = 'Add/update best and latest model'
        # else:
        #     commit = 'Add/update latest model'

        # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # versioned_model_path = model_path.replace(".pkl", f"_{timestamp}.pkl")
        # shutil.copy(model_path, versioned_model_path)
        # symlink_path = os.path.join(os.path.dirname(model_path), "model_latest.pkl")
        # try:
        #     if os.path.exists(symlink_path):
        #         os.remove(symlink_path)
        #     os.symlink(os.path.abspath(model_path), symlink_path)
        # except Exception as e:
        #     print(f"⚠️ Could not update symlink: {e}")
        # subprocess.run(["dvc", "add", "models/model_latest.pkl"])
        # subprocess.run(["git", "add", "models/model_latest.pkl.dvc", ".gitignore"])
        # subprocess.run(["git", "commit", "-m", commit])
        # subprocess.run(["dvc", "push"])

        # print(f"✅ Versioned model saved as {versioned_model_path}")
        # print(f"🔗 Symlink updated: {symlink_path}")

        # # Training log
        # log_entry = {
        #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #     "model_type": type(model).__name__,
        #     "params": str(model.get_params()),
        #     "accuracy": round(acc, 3),
        #     "f1_score": round(f1, 3),
        #     "top1_acc": round(top1_acc, 3),
        #     "top3_acc": round(top3_acc, 3),
        #     "raw_data_count": df["raw_data_id"].nunique(),
        #     "model_path": versioned_model_path
        # }
        # log_path = os.path.join(os.path.dirname(model_path), "training_log.csv")
        # log_df = pd.DataFrame([log_entry])
        # if os.path.exists(log_path):
        #     log_df.to_csv(log_path, mode='a', header=False, index=False)
        # else:
        #     log_df.to_csv(log_path, mode='w', header=True, index=False)
        # print(f"📝 Training log updated: {log_path}")


def top_k_accuracy(model, X, meta_test, k=3):
    probs = model.predict_proba(X)
    correct = 0
    total = 0
    for raw_id, group in meta_test.groupby("raw_data_id"):
        idx = group.index
        topk = np.argsort(probs[idx, 1])[::-1][:k]
        if group["value_clean"].iloc[topk].isin([group["price_user"].iloc[0]]).any():
            correct += 1
        total += 1
    return correct / total


if __name__ == "__main__":
    train_nn_model(data_path="data/knn_training_set.parquet", model_path="models/nn_model.pkl")
