import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# 1. Define Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(script_dir, "snowflake_key.p8")
parquet_path = os.path.join(
    script_dir, "pitcher_fatigue_predictions_FULL.parquet"
)

# 2. Load RSA Private Key
print("Reading private key...")
with open(key_path, "rb") as key_file:
  p_key = serialization.load_pem_private_key(
      key_file.read(), password=None, backend=default_backend()
  )

pkb = p_key.private_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)

# 3. Read Local Parquet File
print(f"Loading local Parquet file: {parquet_path}")
df = pd.read_parquet(parquet_path)

# Ensure all column names are UPPERCASE (Snowflake standard)
df.columns = [col.upper() for col in df.columns]
print(f"Loaded {len(df):,} rows with {len(df.columns)} columns.")

# 4. Connect to Snowflake
print("Connecting to Snowflake...")
conn = snowflake.connector.connect(
    user="AARONKIM",
    account="hmpmxku-fa92135",
    private_key=pkb,
    warehouse="COMPUTE_WH",
    database="BASEBALL_DB",
    schema="DBT_AKIM",
)

# 5. Bulk Upload Parquet Data to Snowflake Table
TABLE_NAME = "PITCHER_FATIGUE_PREDICTIONS"

print(f"Uploading data to Snowflake table: BASEBALL_DB.DBT_AKIM.{TABLE_NAME}...")

success, nchunks, nrows, _ = write_pandas(
    conn=conn,
    df=df,
    table_name=TABLE_NAME,
    database="BASEBALL_DB",
    schema="DBT_AKIM",
    auto_create_table=True,  # Automatically creates table schema if missing
    overwrite=True,  # Overwrites existing table if re-run
    chunk_size=100000,
)

conn.close()

if success:
  print(
      f"🎉 Success! Uploaded {nrows:,} rows into"
      f" BASEBALL_DB.DBT_AKIM.{TABLE_NAME}"
  )
else:
  print("❌ Upload failed.")