import glob, os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

# ==========================================
# 0. CONFIG & SETUP
# ==========================================
RANDOM_STATE = 42
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

M1_PARAMS = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 63, "scale_pos_weight": 2.5, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
M2_PARAMS = {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 31, "scale_pos_weight": 2.0, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}

def get_best_thresh(y_true, y_probs):
    p, r, t = precision_recall_curve(y_true, y_probs)
    return t[np.argmax(2 * (p * r) / (p + r + 1e-10))]

# ==========================================
# 1. LOAD DATA & FEATURE ENGINEERING
# ==========================================
print("Loading dataset & engineering features...")
file_path = os.path.join(WORKING_DIR, "models_staging_PitchFact_FULL.csv")
if not os.path.exists(file_path):
    candidates = glob.glob(os.path.join(WORKING_DIR, "*PitchFact*.csv"))
    file_path = candidates[0] if candidates else glob.glob(os.path.join(WORKING_DIR, "*.csv"))[0]

df = pd.read_csv(file_path)
df.columns = df.columns.str.upper()
df = df.sort_values(by=["GAME_PK", "PITCHER_ID", "AT_BAT_NUMBER", "PITCH_NUMBER"]).reset_index(drop=True)

# Baselines & Target Labels
fastballs = ["FF", "SI", "FC", "FA", "4-SEAM FASTBALL", "SINKER", "CUTTER"]
df["IS_FASTBALL"] = df["PITCH_TYPE"].astype(str).str.upper().isin(fastballs).astype(int)

fb_df = df[df["IS_FASTBALL"] == 1]
baselines = fb_df.groupby(["GAME_PK", "PITCHER_ID"])[["RELEASE_SPEED", "RELEASE_SPIN_RATE", "RELEASE_POS_Z", "RELEASE_EXTENSION"]].transform(lambda x: x.head(10).mean())
df[["FB_BASE_SPEED", "FB_BASE_SPIN", "FB_BASE_POS_Z", "FB_BASE_EXTENSION"]] = baselines

df["IS_FATIGUED"] = ((df["IS_FASTBALL"] == 1) & ((df["FB_BASE_SPEED"] - df["RELEASE_SPEED"]) > 1.5)).astype(int)
df["RUNS_IN_INNING"] = df.groupby(["GAME_PK", "INNING", "INNING_TOPBOT"])["BAT_SCORE_DIFF"].transform(lambda x: x.max() - x.min())
df["IS_BIG_INNING"] = (df["RUNS_IN_INNING"] >= 3).astype(int)

# Rolling Mechanical Decay & Workload Stress
pg = df.groupby(["GAME_PK", "PITCHER_ID"])
df["VELO_DECAY_RECENT"] = df["FB_BASE_SPEED"] - pg["RELEASE_SPEED"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
df["SPIN_DECAY_RECENT"] = df["FB_BASE_SPIN"] - pg["RELEASE_SPIN_RATE"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
df["ARM_SLOT_DROP_Z"] = df["FB_BASE_POS_Z"] - pg["RELEASE_POS_Z"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
df["EXTENSION_DROP"] = df["FB_BASE_EXTENSION"] - pg["RELEASE_EXTENSION"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())

pitcher_avg = df.groupby(["PITCHER_ID", "GAME_PK"])["PITCH_COUNT"].max().groupby("PITCHER_ID").mean()
df["PITCHER_AVG_MAX"] = df["PITCHER_ID"].map(pitcher_avg)
df["WORKLOAD_CAPACITY_RATIO"] = df["PITCH_COUNT"] / (df["PITCHER_AVG_MAX"] + 1e-5)
rest = df["PITCHER_DAYS_SINCE_PREV_GAME"].fillna(4) if "PITCHER_DAYS_SINCE_PREV_GAME" in df.columns else 4
df["REST_ADJUSTED_WORKLOAD"] = df["PITCH_COUNT"] / (rest + 1.0)
df["INNING_PITCH_COUNT"] = df.groupby(["GAME_PK", "PITCHER_ID", "INNING"])["PITCH_NUMBER"].transform("count")

df["WORKLOAD_X_VELO_DECAY"] = df["PITCH_COUNT"] * df["VELO_DECAY_RECENT"].fillna(0)
df["WORKLOAD_X_SLOT_DROP"] = df["PITCH_COUNT"] * df["ARM_SLOT_DROP_Z"].fillna(0)
df["ORDER_X_SPIN_DECAY"] = df["N_THRUORDER_PITCHER"] * df["SPIN_DECAY_RECENT"].fillna(0)
df["NEXT_5_MAX_RELEASE_SPEED"] = pg["RELEASE_SPEED"].transform(lambda x: x.iloc[::-1].rolling(5, min_periods=1).max().iloc[::-1])

# ==========================================
# 2. FEATURE SELECTION & MODEL TRAINING
# ==========================================
exclude = [
    "PITCH_SK", "GAME_PK", "PITCHER_ID", "BATTER_ID", "GAME_DATE", "PITCH_TYPE", "PITCH_RESULT", 
    "PITCH_CATEGORY", "PLAY_DESCRIPTION", "IF_FIELDING_ALIGNMENT", "OF_FIELDING_ALIGNMENT", 
    "IS_FATIGUED", "IS_BIG_INNING", "RUNS_IN_INNING", "FB_BASE_SPEED", "FB_BASE_SPIN", "FB_BASE_POS_Z", 
    "FB_BASE_EXTENSION", "IS_FASTBALL", "RELEASE_SPEED", "NEXT_5_MAX_RELEASE_SPEED", "RELEASE_SPIN_RATE", 
    "RELEASE_POS_X", "RELEASE_POS_Y", "RELEASE_POS_Z", "RELEASE_EXTENSION", "VX0", "VY0", "VZ0", "AX", 
    "AY", "AZ", "API_BREAK_Z_WITH_GRAVITY", "API_BREAK_X_ARM", "API_BREAK_X_BATTER_IN", "SPIN_AXIS", 
    "ZONE", "LAUNCH_SPEED", "LAUNCH_ANGLE", "LAUNCH_SPEED_ANGLE", "HIT_DISTANCE_SC", "HIT_LOCATION", 
    "XBA", "XWOBA", "XSLG", "WOBA_VALUE", "BABIP_VALUE", "DELTA_PITCHER_RUN_EXP", "GAME_SPEED", "GAME_SPIN", 
    "GAME_AVG_SPEED", "GAME_AVG_SPIN", "SEASON_SPEED", "SEASON_SPIN", "SEASON_AVG_SPEED", "SEASON_AVG_SPIN", 
    "PITCHER_AVG_MAX", "M1_FATIGUE_RISK_SCORE", "M2_BIG_INNING_RISK_SCORE", "FATIGUE_STREAK_3", 
    "PULL_RECOMMENDED", "DUGOUT_ACTION"
]

feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def fit_model(params, X, y, name):
    print(f"\n--- {name} ---")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model = LGBMClassifier(**params).fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    thresh = get_best_thresh(y_te, probs)
    print(f"ROC-AUC: {roc_auc_score(y_te, probs):.4f} | Optimal Threshold: {thresh:.4f}")
    print(classification_report(y_te, probs >= thresh))
    oof_probs = cross_val_predict(LGBMClassifier(**params), X, y, cv=cv, method="predict_proba")[:, 1]
    return oof_probs, thresh

# Train Models
df["M1_FATIGUE_RISK_SCORE"], m1_thresh = fit_model(M1_PARAMS, df[feature_cols], df["IS_FATIGUED"], "Model 1: Fatigue Risk")
df["FATIGUE_STREAK_3"] = pg["M1_FATIGUE_RISK_SCORE"].transform(lambda x: (x >= m1_thresh).rolling(3, min_periods=3).sum() == 3).astype(int)

m2_features = feature_cols + ["M1_FATIGUE_RISK_SCORE", "FATIGUE_STREAK_3"]
df["M2_BIG_INNING_RISK_SCORE"], m2_thresh = fit_model(M2_PARAMS, df[m2_features], df["IS_BIG_INNING"], "Model 2: Big Inning Risk")

# ==========================================
# 3. DUGOUT ACTION ENGINE & EXPORTS
# ==========================================
df["PULL_RECOMMENDED"] = (df["M2_BIG_INNING_RISK_SCORE"] >= m2_thresh) & (df["FATIGUE_STREAK_3"] == 1)

def assign_dugout_status(row):
    if (row["M2_BIG_INNING_RISK_SCORE"] >= m2_thresh) and (row["FATIGUE_STREAK_3"] == 1):
        return "RED: PULL_IMMEDIATE"
    elif row["M2_BIG_INNING_RISK_SCORE"] >= (m2_thresh * 0.80):
        return "ORANGE: WARMUP_BULLPEN"
    elif row["M1_FATIGUE_RISK_SCORE"] >= m1_thresh:
        return "YELLOW: MONITOR_CLOSELY"
    return "GREEN: NORMAL"

df["DUGOUT_ACTION"] = df.apply(assign_dugout_status, axis=1)

# Save Predictions
df.to_parquet(os.path.join(WORKING_DIR, "pitcher_fatigue_predictions_FULL.parquet"))
df.to_csv(os.path.join(WORKING_DIR, "pitcher_fatigue_and_big_inning_predictions.csv"), index=False)
print("\n✅ PIPELINE SUCCESS! Saved Parquet and CSV outputs.")