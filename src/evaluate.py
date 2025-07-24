import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, data_path: str, max_points: int = 1000):
    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📊 MAE:  {mae:.2f}")
    print(f"📊 R²:   {r2:.3f}")

    # Plot (optional)
    plt.figure(figsize=(6, 6))
    plt.scatter(y[:max_points], y_pred[:max_points], alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Ideal")
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("Model Prediction vs True Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
