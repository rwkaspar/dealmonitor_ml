import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def train_knn_model(data_path: str, model_path: str = "models/knn_model.pkl"):
    df = pd.read_parquet(data_path)

    feature_cols = [
        "value_clean", "depth", "css_len", "has_currency"
    ]

    X = df[feature_cols]
    y = df["match_with_user"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"✅ F1 Score: {f1_score(y_test, y_pred):.3f}")

    joblib.dump(model, model_path)
    print(f"✅ Modell gespeichert unter {model_path}")
