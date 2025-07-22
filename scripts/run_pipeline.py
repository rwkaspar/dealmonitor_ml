import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sqlalchemy import create_engine
from src.features import clean_price_user
from src.candidate_extractor import extract_price_candidates
from src.knn_dataset_builder import build_knn_training_rows
from src.model_nn import train_nn_model


# 1. Connect to the PostgreSQL database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://dev_user:dev_password@localhost:5432/dev_db")
engine = create_engine(DATABASE_URL)

# 2. Load raw_data entries
print("📥 Loading raw_data from database...")
df = pd.read_sql("SELECT * FROM raw_data", engine)
print(f"🔢 Loaded {len(df)} rows.")

# 3. Build feature rows for each price candidate
print("🧠 Extracting candidate features...")
rows = []
for row in df.to_dict(orient="records"):
    rows.extend(build_knn_training_rows(row))

# 4. Save training data to file
train_df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
train_df.to_parquet("data/knn_training_set.parquet")
print(f"✅ Saved {len(train_df)} training rows to data/knn_training_set.parquet")

# 5. Train the neural network model on the extracted features
train_nn_model(data_path="data/knn_training_set.parquet", model_path="models/nn_model.pkl")
