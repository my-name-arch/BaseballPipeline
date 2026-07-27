from pybaseball import chadwick_register
import pandas as pd

print("1. Downloading player lookup database from Chadwick Bureau...")
df = chadwick_register()

print("2. Formatting player names...")
# Clean and combine first + last names
df['name_first'] = df['name_first'].fillna('').str.title()
df['name_last'] = df['name_last'].fillna('').str.title()

df['BATTER_NAME'] = (df['name_first'] + ' ' + df['name_last']).str.strip()

# Select MLB ID (key_mlbam) and full name, removing duplicates/nulls
dim_batters = df[['key_mlbam', 'BATTER_NAME']].dropna().drop_duplicates(subset=['key_mlbam'])
dim_batters = dim_batters.rename(columns={'key_mlbam': 'BATTER_ID'})
dim_batters['BATTER_ID'] = dim_batters['BATTER_ID'].astype(int)

# Save into your dbt seeds folder (or current directory)
output_path = 'seeds/dim_batters.csv' # Adjust path if your seeds folder is elsewhere
dim_batters.to_csv(output_path, index=False)

print(f"Success! Saved {len(dim_batters)} player records to {output_path}") 