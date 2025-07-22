import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from src.features import extract_features, clean_price_user


def train_model(data_path: str, model_path: str = "models/price_model.pkl"):
    # Lade vorbereitete Daten
    df = pd.read_parquet(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Train/Test-Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modell initialisieren und trainieren
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Vorhersage
    y_pred = model.predict(X_test)

    # Metriken ausgeben
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Modell speichern
    joblib.dump(model, model_path)
    print(f"✅ Modell gespeichert unter {model_path}")


def predict_price(sample_row: dict, model_path: str = "models/price_model.pkl") -> float:
    """
    Erwartet ein Dictionary mit denselben Keys wie raw_data.jsonl
    Gibt den vorhergesagten Preis zurück (float)
    """
    model = joblib.load(model_path)
    features = extract_features(sample_row)
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    return round(prediction, 2)
