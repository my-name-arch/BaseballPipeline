import glob
import os
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

warnings.filterwarnings('ignore')

# Disable OpenMP multi-threading deadlocks on macOS
os.environ["OMP_NUM_THREADS"] = "1"

# ==========================================
# 0. CONFIG & SETUP
# ==========================================
RANDOM_STATE = 42
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

def get_optimal_precision_thresh(y_true, y_probs, target_precision=0.45):
    """
    Finds the optimal probability threshold satisfying a strict PRECISION floor
    (default: 45%+ precision) to aggressively suppress false alarms.
    """
    p, r, t = precision_recall_curve(y_true, y_probs)
    valid_idx = np.where(p[:-1] >= target_precision)[0]
    
    if len(valid_idx) > 0:
        f1 = 2 * (p[valid_idx] * r[valid_idx]) / (p[valid_idx] + r[valid_idx] + 1e-10)
        best_idx = valid_idx[np.argmax(f1)]
        return t[best_idx]
    else:
        return t[np.argmax(p[:-1])]


def get_snowflake_connection():
    key_candidates = (
        glob.glob(os.path.join(WORKING_DIR, "*.p8")) +
        glob.glob(os.path.join(WORKING_DIR, "*.pem")) +
        glob.glob(os.path.join(WORKING_DIR, "*.pub"))
    )

    if not key_candidates:
        raise FileNotFoundError("Could not find RSA key file (.p8 / .pem / .pub) in working directory.")

    KEY_PATH = key_candidates[0]
    with open(KEY_PATH, "rb") as key_file:
        p_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    return snowflake.connector.connect(
        user='AARONKIM',
        account='HMPMXKU-FA92135',
        private_key=pkb,
        warehouse='COMPUTE_WH',
        database='BASEBALL_DB',
        schema='DBT_AKIM',
        role='ACCOUNTADMIN'
    )


# ==========================================
# 1. DATA LOADING & ADVANCED FEATURE ENGINEERING
# ==========================================
print("Loading dataset & engineering false-positive resistant features...")
file_path = os.path.join(WORKING_DIR, "models_staging_PitchFact_FULL.csv")
if not os.path.exists(file_path):
    candidates = glob.glob(os.path.join(WORKING_DIR, "*PitchFact*.csv"))
    file_path = candidates[0] if candidates else glob.glob(os.path.join(WORKING_DIR, "*.csv"))[0]

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.upper()
df = df.loc[:, ~df.columns.duplicated()].copy()

# Sort pitch sequence strictly
sort_cols = [c for c in ['GAME_PK', 'PITCHER_ID', 'AT_BAT_NUMBER', 'PITCH_NUMBER'] if c in df.columns]
df = df.sort_values(sort_cols).reset_index(drop=True)

# ------------------------------------------
# ENRICHMENT: FETCH TEAMS FROM DIM_GAMES
# ------------------------------------------
print("Fetching HOME_TEAM and AWAY_TEAM from BASEBALL_DB.DBT_AKIM.DIM_GAMES...")
try:
    conn = get_snowflake_connection()
    games_query = "SELECT GAME_PK, HOME_TEAM, AWAY_TEAM FROM BASEBALL_DB.DBT_AKIM.DIM_GAMES"
    dim_games_df = pd.read_sql(games_query, conn)
    df = df.merge(dim_games_df, on="GAME_PK", how="left")
except Exception as e:
    print(f"Warning: Could not fetch dim_games from Snowflake ({e}). Proceeding with local dataset.")

# ------------------------------------------
# FASTBALL BASELINES & PERSONALIZED Z-SCORES
# ------------------------------------------
fastballs = ["FF", "SI", "FC", "FA", "4-SEAM FASTBALL", "SINKER", "CUTTER"]
df["IS_FASTBALL"] = df["PITCH_TYPE"].astype(str).str.upper().isin(fastballs).astype(int)

# Inning 1 Fastball Baseline (First 8 fastballs in Inning 1)
fb_only = df[(df["IS_FASTBALL"] == 1) & (df["INNING"] == df["INNING"].min())].copy()
fb_baselines = (
    fb_only.groupby(["GAME_PK", "PITCHER_ID"])
    .apply(lambda g: g.head(8)[["RELEASE_SPEED", "RELEASE_SPIN_RATE", "RELEASE_POS_Z", "RELEASE_EXTENSION"]].mean())
    .reset_index()
)
fb_baselines.columns = [
    "GAME_PK", "PITCHER_ID", 
    "FB_BASE_SPEED", "FB_BASE_SPIN", "FB_BASE_POS_Z", "FB_BASE_EXTENSION"
]

df = df.merge(fb_baselines, on=["GAME_PK", "PITCHER_ID"], how="left")

# Calculate Personal Standard Deviations per Pitcher
p_stats = df.groupby('PITCHER_ID').agg(
    P_VELO_STD=('RELEASE_SPEED', lambda x: max(x.std(), 0.5)),
    P_SLOT_STD=('RELEASE_POS_Z', lambda x: max(x.std(), 0.04))
).reset_index()
df = df.merge(p_stats, on='PITCHER_ID', how='left')

# MULTI-SIGNAL GROUND TRUTH TARGET
df["VELO_DROP_RAW"] = df["FB_BASE_SPEED"] - df["RELEASE_SPEED"]
df["ARM_DROP_RAW"] = df["FB_BASE_POS_Z"] - df["RELEASE_POS_Z"]
df["SPIN_DROP_RAW"] = df["FB_BASE_SPIN"] - df["RELEASE_SPIN_RATE"]

df["IS_FATIGUED"] = (
    (df["IS_FASTBALL"] == 1) & 
    (df["VELO_DROP_RAW"] > 1.5) & 
    ((df["ARM_DROP_RAW"] > 0.12) | (df["SPIN_DROP_RAW"] > 60))
).astype(int)

if "BAT_SCORE_DIFF" in df.columns:
    df["RUNS_IN_INNING"] = df.groupby(["GAME_PK", "INNING", "INNING_TOPBOT"])["BAT_SCORE_DIFF"].transform(
        lambda x: x.max() - x.min()
    )
    df["IS_BIG_INNING"] = (df["RUNS_IN_INNING"] >= 3).astype(int)
else:
    df["IS_BIG_INNING"] = 0

# ------------------------------------------
# ROLLING MECHANICAL DECAY & WORKLOAD METRICS
# ------------------------------------------
pg = df.groupby(["GAME_PK", "PITCHER_ID"])

# Exponential Weighted Moving Decay Signals (span=8)
df["VELO_DECAY_RECENT"] = (df["FB_BASE_SPEED"] - pg["RELEASE_SPEED"].transform(
    lambda x: x.shift(1).ewm(span=8, min_periods=3).mean()
)).fillna(0)

df["SPIN_DECAY_RECENT"] = (df["FB_BASE_SPIN"] - pg["RELEASE_SPIN_RATE"].transform(
    lambda x: x.shift(1).ewm(span=8, min_periods=3).mean()
)).fillna(0)

df["ARM_SLOT_DROP_Z"] = (df["FB_BASE_POS_Z"] - pg["RELEASE_POS_Z"].transform(
    lambda x: x.shift(1).ewm(span=8, min_periods=3).mean()
)).fillna(0)

# Pitch Count Workload Interactions
df["PITCH_COUNT"] = pg.cumcount() + 1
df["INNING_PITCH_COUNT"] = df.groupby(["GAME_PK", "PITCHER_ID", "INNING"])["PITCH_NUMBER"].transform("count")
df["WORKLOAD_X_VELO_DECAY"] = df["PITCH_COUNT"] * df["VELO_DECAY_RECENT"]
df["WORKLOAD_X_SLOT_DROP"] = df["PITCH_COUNT"] * df["ARM_SLOT_DROP_Z"]

if "N_THRUORDER_PITCHER" in df.columns:
    df["ORDER_X_SPIN_DECAY"] = df["N_THRUORDER_PITCHER"] * df["SPIN_DECAY_RECENT"]

# ==========================================
# 2. FEATURE SELECTION & CALIBRATED MODEL TRAINING
# ==========================================
base_exclude = [
    # Identifiers & Metadata
    "PITCH_SK", "GAME_PK", "PITCHER_ID", "BATTER_ID", "GAME_DATE", "PITCH_TYPE", "PITCH_RESULT",
    "PITCH_CATEGORY", "PLAY_DESCRIPTION", "IF_FIELDING_ALIGNMENT", "OF_FIELDING_ALIGNMENT",
    "HOME_TEAM", "AWAY_TEAM",
    
    # Target Definitions & Downstream Logic
    "IS_FATIGUED", "IS_BIG_INNING", "RUNS_IN_INNING", "BAT_SCORE_DIFF", 
    "VELO_DROP_RAW", "ARM_DROP_RAW", "SPIN_DROP_RAW", "DUGOUT_ACTION", "PULL_RECOMMENDED",
    
    # Baselines
    "FB_BASE_SPEED", "FB_BASE_SPIN", "FB_BASE_POS_Z", "FB_BASE_EXTENSION", "IS_FASTBALL",
    "GAME_SPEED", "GAME_SPIN", "GAME_AVG_SPEED", "GAME_AVG_SPIN", 
    "SEASON_SPEED", "SEASON_SPIN", "SEASON_AVG_SPEED", "SEASON_AVG_SPIN", "PITCHER_AVG_MAX",
    
    # Trajectory Vectors
    "VX0", "VY0", "VZ0", "AX", "AY", "AZ", "API_BREAK_Z_WITH_GRAVITY", "API_BREAK_X_ARM", 
    "API_BREAK_X_BATTER_IN", "SPIN_AXIS", "RELEASE_POS_X", "RELEASE_POS_Y", "RELEASE_EXTENSION",
    "PAST_5_MAX_RELEASE_SPEED", "NEXT_5_MAX_RELEASE_SPEED",
    
    # Post-Contact Batted-Ball Leakage
    "LAUNCH_SPEED", "LAUNCH_ANGLE", "LAUNCH_SPEED_ANGLE", "HIT_DISTANCE_SC", 
    "HIT_LOCATION", "XBA", "XWOBA", "XSLG", "WOBA_VALUE", "BABIP_VALUE"
]

# Model 1 Exclusions (Includes RELEASE_SPEED, RELEASE_SPIN_RATE, RELEASE_POS_Z)
m1_exclude = base_exclude + [
    "M1_FATIGUE_RISK_SCORE", "M2_BIG_INNING_RISK_SCORE", "FATIGUE_STREAK_3", "DELTA_PITCHER_RUN_EXP"
]

m1_feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in m1_exclude]

# Model 2 Exclusions
m2_exclude = base_exclude + [
    "M2_BIG_INNING_RISK_SCORE"
]

m2_feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in m2_exclude]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# n_jobs set to 1 to prevent macOS OpenMP deadlock hangs
M1_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.02,
    "num_leaves": 25,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 2.5,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "verbose": -1,
}

M2_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "scale_pos_weight": 2.0,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "verbose": -1,
}

def fit_calibrated_model(params, X, y, cv_obj, name, target_precision=0.45):
    print(f"\n--- Fitting {name} (Calibrated Isotonic Classifier) ---")
    print(f"Features included ({len(X.columns)}): {list(X.columns)}")
    
    base_model = LGBMClassifier(**params)
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
    
    oof_probs = cross_val_predict(calibrated_model, X, y, cv=cv_obj, method="predict_proba")[:, 1]
    thresh = get_optimal_precision_thresh(y, oof_probs, target_precision=target_precision)
    
    print(f"OOF ROC-AUC: {roc_auc_score(y, oof_probs):.4f} | Optimal High-Precision Threshold: {thresh:.4f}")
    print(classification_report(y, oof_probs >= thresh))
    return oof_probs, thresh

# Train Model 1 (M1 Fatigue Risk Engine)
df["M1_FATIGUE_RISK_SCORE"], m1_thresh = fit_calibrated_model(
    M1_PARAMS, df[m1_feature_cols], df["IS_FATIGUED"], cv, "Model 1: Fatigue Risk Engine", target_precision=0.45
)

# Calculate 3-Pitch Rolling Streak Signal
pg = df.groupby(["GAME_PK", "PITCHER_ID"])
df["FATIGUE_STREAK_3"] = pg["M1_FATIGUE_RISK_SCORE"].transform(
    lambda x: (x >= m1_thresh).rolling(3, min_periods=3).sum() == 3
).fillna(0).astype(int)

# Train Model 2 (M2 Big Inning Risk Engine)
df["M2_BIG_INNING_RISK_SCORE"], m2_thresh = fit_calibrated_model(
    M2_PARAMS, df[m2_feature_cols], df["IS_BIG_INNING"], cv, "Model 2: Big Inning Risk Engine", target_precision=0.40
)

# ==========================================
# 3. DUGOUT ACTION ENGINE & EXPORTS
# ==========================================
df["PULL_RECOMMENDED"] = ((df["M2_BIG_INNING_RISK_SCORE"] >= m2_thresh) & (df["FATIGUE_STREAK_3"] == 1)).astype(int)

conditions = [
    (df["M2_BIG_INNING_RISK_SCORE"] >= m2_thresh) & (df["FATIGUE_STREAK_3"] == 1),
    (df["M2_BIG_INNING_RISK_SCORE"] >= (m2_thresh * 0.85)),
    (df["M1_FATIGUE_RISK_SCORE"] >= m1_thresh)
]
choices = [
    "RED: PULL_IMMEDIATE",
    "ORANGE: WARMUP_BULLPEN",
    "YELLOW: MONITOR_CLOSELY"
]

df["DUGOUT_ACTION"] = np.select(conditions, choices, default="GREEN: NORMAL")

# Local File Exports
parquet_path = os.path.join(WORKING_DIR, "pitcher_fatigue_predictions_FULL.parquet")
csv_path = os.path.join(WORKING_DIR, "pitcher_fatigue_and_big_inning_predictions.csv")

df.to_parquet(parquet_path, index=False)
df.to_csv(csv_path, index=False)

print("\n✅ LOCAL EXPORT SUCCESS!")
print(f"1. Parquet dataset exported to: {parquet_path}")
print(f"2. FULL Predictions CSV exported to: {csv_path}")

# ==========================================
# 4. EXPORT PREDICTIONS TO SNOWFLAKE
# ==========================================
try:
    print("\nReusing Snowflake connection for data upload...")
    snowflake_df = df.copy()
    snowflake_df.columns = (
        snowflake_df.columns
        .str.strip()
        .str.upper()
        .str.replace(' ', '_')
        .str.replace('[^A-Z0-9_]', '', regex=True)
    )
    snowflake_df = snowflake_df.loc[:, ~snowflake_df.columns.duplicated()].copy()

    TABLE_NAME = 'PITCHER_FATIGUE_PREDICTIONS'

    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS BASEBALL_DB.DBT_AKIM.{TABLE_NAME}")
    cursor.close()

    success, nchunks, nrows, _ = write_pandas(
        conn=conn,
        df=snowflake_df,
        table_name=TABLE_NAME,
        auto_create_table=True,
        overwrite=True,
        quote_identifiers=True,
        chunk_size=50000,
        compression='gzip'
    )
    print(f"✅ SNOWFLAKE UPLOAD SUCCESS! Uploaded {nrows} rows into BASEBALL_DB.DBT_AKIM.{TABLE_NAME}")
    conn.close()
except Exception as e:
    print(f"Snowflake upload bypassed or failed: {e}")
