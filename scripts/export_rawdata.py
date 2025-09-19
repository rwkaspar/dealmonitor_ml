import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

sys.path.append(os.path.abspath("dealmonitor/backend/src"))
from dealmonitor.features.features import clean_price

# Load environment variables from .env file in project root directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path=env_path)

# Get database connection parameters from environment variables
db_user = os.getenv("db_user")
db_pass = os.getenv("db_pass")
db_host = os.getenv("db_host")
db_port = os.getenv("db_port")
db_name = os.getenv("db_name")

print(f"Connecting to database {db_name} at {db_host}:{db_port} as user {db_user}")

# Create SQLAlchemy engine for PostgreSQL connection
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

# Define SQL query (example: joining trackers, urls, price history)
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

# Read the raw_data table ordered by creation timestamp into a DataFrame
df = pd.read_sql("SELECT * FROM raw_data ORDER BY created_at", engine)

# Clean up price_user column using the clean_price function
df["price_user_clean"] = df["price_user"].apply(clean_price)

# Filter out rows where price_user_clean is null
df = df[df["price_user_clean"].notnull()]

# Export the cleaned data as JSON Lines for downstream use (e.g. LLMs)
output_path = os.path.join("data", "raw", "raw_data.jsonl")
df.to_json(output_path, orient="records", lines=True)
print(f"Exported {len(df)} rows to {output_path}")
