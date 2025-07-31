# scripts/export_rawdata.py
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from src.features import clean_price_user


# load_dotenv()  # lädt .env aus aktuellem Verzeichnis
load_dotenv(dotenv_path="/home/dev/projects/dealmonitor_ml_clean/.env")
print("Database URL:", os.getenv("DATABASE_URL"))
db_url = os.getenv("DATABASE_URL")

db_user = os.getenv("db_user")
db_pass = os.getenv("db_pass")
db_host = os.getenv("db_host")
db_port = os.getenv("db_port")
db_name = os.getenv("db_name")
print(f"Connecting to database {db_name} at {db_host}:{db_port} as user {db_user}")
# DATABASE_URL=postgresql+psycopg2://dev_user:dev_password@db-dev:5432/dev_db
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

# Beispiel: alle URLs + Tracker + Preisverlauf joinen
query = """
SELECT
    t.id AS tracker_id,
    t.name,
    t.target_price,
    u.url,
    u.selector,
    u.outer_html,
    p.price,
    p.checked_at
FROM trackers t
JOIN urls u ON u.tracker_id = t.id
LEFT JOIN price_history p ON p.tracker_id = t.id
"""

df = pd.read_sql("SELECT * FROM raw_data ORDER BY created_at", engine)

# convert 
df["price_user_clean"] = df["price_user"].apply(clean_price_user)
df = df[df["price_user_clean"].notnull()]

# JSONL (für LLMs etc.)
df.to_json("data/raw/raw_data.jsonl", orient="records", lines=True)
print(f"Exported {len(df)} rows to data/raw/raw_data.jsonl")

# def export_rawdata_jsonl(out_path="data/raw/rawdata_export.jsonl"):
#     session = SessionLocal()
#     try:
#         # Fetch all entries from the table
#         all_entries = session.query(RawData).all()
#         with open(out_path, "w", encoding="utf-8") as f:
#             for entry in all_entries:
#                 # Build dict for JSON export
#                 row = {c.name: getattr(entry, c.name) for c in entry.__table__.columns}
#                 f.write(json.dumps(row, ensure_ascii=False) + "\n")
#         print(f"Exported {len(all_entries)} rows to {out_path}")
#     finally:
#         session.close()

# if __name__ == "__main__":
#     export_rawdata_jsonl()
