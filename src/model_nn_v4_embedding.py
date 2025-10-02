import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from dealmonitor.features.features import extract_price_features

def build_feature_df(df):
    features = df.apply(lambda row: extract_price_features(row.to_dict()), axis=1)
    feats = pd.DataFrame(list(features))
    return feats

def train_shop_emb_model(data_path="data/knn_training_set.parquet"):
    df = pd.read_parquet(data_path)
    X_feats = build_feature_df(df).fillna(0)
    df = df.reset_index(drop=True); X_feats = X_feats.reset_index(drop=True)

    # Shop-ID encoden
    shop_le = LabelEncoder()
    shops = shop_le.fit_transform(df["domain"])
    num_shops = len(shop_le.classes_)
    
    # Meta und Target
    raw_ids = df["raw_data_id"].values
    y = df["match_with_user"].astype(int).values

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_feats, y)
    shops_res = np.repeat(shops, ros.sample_indices_.size // len(y))[:len(y_res)]  # passend anpassen

    # Train/Test Split
    X_tr, X_te, y_tr, y_te, shop_tr, shop_te = train_test_split(
        X_res, y_res, shops_res, test_size=0.2, random_state=42
    )

    # Keras Modell mit Shop-Embedding
    price_input = layers.Input(shape=(X_tr.shape[1],), name="price_features")
    shop_input = layers.Input(shape=(), dtype="int32", name="shop_id")

    emb = layers.Embedding(input_dim=num_shops, output_dim=16, name="shop_embedding")(shop_input)
    shop_vec = layers.Flatten()(emb)

    x = layers.Concatenate()([price_input, shop_vec])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=[price_input, shop_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Training
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        {"price_features": X_tr, "shop_id": shop_tr},
        y_tr,
        validation_split=0.1,
        epochs=100,
        batch_size=256,
        callbacks=[es]
    )

    # Evaluation
    loss, acc = model.evaluate({"price_features": X_te, "shop_id": shop_te}, y_te)
    print(f"Test Accuracy: {acc:.3f}")

    # Speichern
    model.save("models/shop_emb_model")
    joblib.dump(shop_le, "models/shop_label_encoder.pkl")
    print("Model und Encoder gespeichert.")

if __name__ == "__main__":
    train_shop_emb_model()
