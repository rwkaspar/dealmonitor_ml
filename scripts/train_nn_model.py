import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_nn_v3 import train_nn_model

if __name__ == "__main__":
    train_nn_model(data_path="data/knn_training_set.parquet", model_path="models/nn_model.pkl")
