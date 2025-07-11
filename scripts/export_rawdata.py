# scripts/export_rawdata.py
import os
from dotenv import load_dotenv
import json
from dealmonitor.database import SessionLocal, RawData

# load_dotenv()  # lädt .env aus aktuellem Verzeichnis
load_dotenv(dotenv_path="/home/dev/projects/dealmonitor_ml/.env")
print("Database URL:", os.getenv("DATABASE_URL"))

def export_rawdata_jsonl(out_path="data/raw/rawdata_export.jsonl"):
    session = SessionLocal()
    try:
        # Fetch all entries from the table
        all_entries = session.query(RawData).all()
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in all_entries:
                # Build dict for JSON export
                row = {c.name: getattr(entry, c.name) for c in entry.__table__.columns}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Exported {len(all_entries)} rows to {out_path}")
    finally:
        session.close()

if __name__ == "__main__":
    export_rawdata_jsonl()
