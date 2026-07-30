import os
import glob
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="MLB Dugout Fatigue Engine", page_icon="⚾", layout="wide")

COLOR_MAP = {
    'FF': '#d62728', 'SI': '#ff7f0e', 'FC': '#bcbd22', 'FA': '#e377c2',
    'SL': '#1f77b4', 'CU': '#9467bd', 'KC': '#8c564b', 'CH': '#2ca02c',
    'FS': '#17becf', 'ST': '#7f7f7f', 'SV': '#bcbd22'
}

@st.cache_data
def load_data():
    """Loads and preprocesses Statcast pitcher fatigue dataset with game dimension metadata."""
    try:
        conn = st.connection("snowflake")
        df = conn.query("""
            SELECT f.*, g.HOME_TEAM, g.AWAY_TEAM, g.GAME_DATE, g.GAME_YEAR, g.GAME_TYPE
            FROM BASEBALL_DB.DBT_AKIM.PITCHER_FATIGUE_PREDICTIONS f
            LEFT JOIN BASEBALL_DB.DBT_AKIM.DIM_GAMES g ON f.GAME_PK = g.GAME_PK
        """)
        df.columns = df.columns.str.strip().str.upper()
    except Exception:
        primary_path = "/Users/Dooley/MLB Project/pitcher_fatigue/pitcher_fatigue_and_big_inning_predictions.csv"
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        
        candidate_paths = [
            primary_path,
            os.path.join(root_dir, "pitcher_fatigue_and_big_inning_predictions.csv"),
            os.path.join(script_dir, "pitcher_fatigue_and_big_inning_predictions.csv"),
            os.path.join(os.getcwd(), "pitcher_fatigue_and_big_inning_predictions.csv")
        ]
        
        pred_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        
        if not pred_path:
            csv_matches = glob.glob(os.path.join(root_dir, "*.csv")) + glob.glob(os.path.join(script_dir, "*.csv"))
            if csv_matches:
                pred_path = csv_matches[0]

        if not pred_path or not os.path.exists(pred_path):
            st.error(f"Dataset not found. Checked path: {primary_path}")
            st.stop()
            
        df = pd.read_csv(pred_path)
        df.columns = df.columns.str.strip().str.upper()

        games_path = next((p for p in [os.path.join(root_dir, f) for f in ["dim_games.csv", "dim_game.csv", "game_dim.csv"]]
                          + [os.path.join(script_dir, f) for f in ["dim_games.csv", "dim_game.csv", "game_dim.csv"]] if os.path.exists(p)), None)
        if games_path:
            dim_games = pd.read_csv(games_path)
            dim_games.columns = dim_games.columns.str.strip().str.upper()
            df['GAME_PK'] = df['GAME_PK'].astype(int)
            dim_games['GAME_PK'] = dim_games['GAME_PK'].astype(int)
            cols = ['GAME_PK'] + [c for c in ['HOME_TEAM', 'AWAY_TEAM', 'GAME_DATE', 'GAME_YEAR', 'GAME_TYPE'] if c in dim_games.columns]
            df = df.merge(dim_games[cols], on='GAME_PK', how='left')

    # Filter out rare pitch types (<0.5% usage)
    if 'PITCH_TYPE' in df.columns:
        pitch_counts = df['PITCH_TYPE'].value_counts(normalize=True)
        valid_pitch_types = pitch_counts[pitch_counts >= 0.005].index.tolist()
        df = df[df['PITCH_TYPE'].isin(valid_pitch_types)].copy()

    sort_cols = [c for c in ['GAME_PK', 'PITCHER_ID', 'GAME_DATE', 'INNING', 'AT_BAT_NUMBER', 'PITCH_NUMBER', 'PITCH_COUNT'] if c in df.columns]
    df = df.sort_values(sort_cols)
    
    # Recalculate PITCH_COUNT per outing to guarantee clean 1-N sequence
    df['PITCH_COUNT'] = df.groupby(['GAME_PK', 'PITCHER_ID']).cumcount() + 1

    outing_roles = df.groupby(['GAME_PK', 'PITCHER_ID']).agg(
        START_INNING=('INNING', 'min'),
        MAX_PITCH_COUNT=('PITCH_COUNT', 'max')
    ).reset_index()

    # Outing Workload Classification:
    # Starters: Outings with 50+ pitches OR starting in Inning 1 with 30+ pitches
    # Relievers: Traditional bullpen workloads (< 50 pitches)
    outing_roles['PITCHER_ROLE'] = np.where(
        (outing_roles['MAX_PITCH_COUNT'] >= 50) | ((outing_roles['START_INNING'] == 1) & (outing_roles['MAX_PITCH_COUNT'] >= 30)),
        'Starter', 'Reliever'
    )
    df = df.merge(outing_roles[['GAME_PK', 'PITCHER_ID', 'PITCHER_ROLE']], on=['GAME_PK', 'PITCHER_ID'], how='left')

    for b_col in ["ON_1B", "ON_2B", "ON_3B"]:
        if b_col not in df.columns:
            df[b_col] = np.nan

    df["HAS_RUNNER_1B"] = df["ON_1B"].notna().astype(int)
    df["HAS_RUNNER_2B"] = df["ON_2B"].notna().astype(int)
    df["HAS_RUNNER_3B"] = df["ON_3B"].notna().astype(int)
    df["RUNNERS_ON_BASE_COUNT"] = df["HAS_RUNNER_1B"] + df["HAS_RUNNER_2B"] + df["HAS_RUNNER_3B"]
    df["IS_RISP"] = ((df["HAS_RUNNER_2B"] == 1) | (df["HAS_RUNNER_3B"] == 1)).astype(int)

    des_cols = [c for c in ['PITCH_RESULT', 'DESCRIPTION', 'EVENTS', 'PITCH_DES', 'TYPE', 'DES', 'RESULT'] if c in df.columns]
    if des_cols:
        combined_des = df[des_cols].fillna('').astype(str).agg(' '.join, axis=1).str.replace('_', ' ', regex=False).str.upper()
        
        # Keywords for Whiffs
        whiff_keywords = r'SWINGING STRIKE|SWUNG ON AND MISSED|MISS|FOUL TIP|WHIFF'
        # Keywords for Swings
        swing_keywords = r'SWINGING STRIKE|SWUNG ON AND MISSED|MISS|FOUL TIP|FOUL|IN PLAY|HIT INTO PLAY|STRIKE BLOCKED|HIT IN PLAY|WHIFF'
        
        df['IS_WHIFF'] = combined_des.str.contains(whiff_keywords, regex=True).astype(int)
        df['IS_SWING'] = combined_des.str.contains(swing_keywords, regex=True).astype(int)
    else:
        df['IS_SWING'] = 0
        df['IS_WHIFF'] = 0

    # Dynamic xBA mapping
    xba_col = next((c for c in ['ESTIMATED_BA_USING_SPEEDANGLE', 'XBA', 'EST_BA', 'EXPECTED_BA', 'ESTIMATED_BA'] if c in df.columns), None)
    if xba_col:
        df['DERIVED_XBA'] = pd.to_numeric(df[xba_col], errors='coerce')
    else:
        df['DERIVED_XBA'] = np.nan

    # Dynamic Run Expectancy Delta mapping (Strictly prioritize explicit DELTA columns)
    re_candidates = [c for c in df.columns if c in ['DELTA_RUN_EXP', 'DELTA_PITCHER_RUN_EXP', 'DELTA_RE', 'RE_DELTA', 'RUN_EXP_DELTA']]
    if not re_candidates:
        re_candidates = [c for c in df.columns if any(k in c for k in ['DELTA_RE', 'RE_DELTA', 'DELTA_RUN_EXP', 'RUN_EXP_DELTA', 'DELTA_PITCHER'])]
    if not re_candidates:
        re_candidates = [c for c in df.columns if 'RUN_EXP' in c and 'BEFORE' not in c and 'AFTER' not in c]

    re_col = re_candidates[0] if re_candidates else None
    if re_col:
        df['DERIVED_RUN_EXP_DELTA'] = pd.to_numeric(df[re_col], errors='coerce').fillna(0.0)
    else:
        df['DERIVED_RUN_EXP_DELTA'] = 0.0

    # Flexible outcome event scanning for Outs, Hits, Walks, and Runs
    event_candidate_cols = [c for c in df.columns if any(k in c for k in ['EVENT', 'DES', 'RESULT', 'PA_', 'BAT_', 'OUT'])]
    if not event_candidate_cols:
        event_candidate_cols = [c for c in ['EVENTS', 'EVENT', 'DES', 'DESCRIPTION', 'PITCH_DES', 'PITCH_RESULT', 'TYPE'] if c in df.columns]

    if event_candidate_cols:
        combined_events = df[event_candidate_cols].fillna('').astype(str).agg(' '.join, axis=1).str.replace('_', ' ', regex=False).str.lower()
        
        # Outs
        outs_3 = combined_events.str.contains(r'triple play', regex=True).astype(int) * 3
        outs_2 = combined_events.str.contains(r'double play|grounded into double play|strikeout double play', regex=True).astype(int) * 2
        outs_1 = combined_events.str.contains(r'strikeout|field out|force out|sac fly|sac bunt|fielders choice|caught stealing|pickoff|flyout|groundout|lineout|pop out|\bout\b', regex=True).astype(int) * 1
        
        df['DERIVED_OUTS'] = np.where(outs_3 > 0, 3, np.where(outs_2 > 0, 2, outs_1))
        
        # Hits and Walks
        df['DERIVED_HITS'] = combined_events.str.contains(r'single|double|triple|home run', regex=True).astype(int)
        df['DERIVED_WALKS'] = combined_events.str.contains(r'walk|intent walk|base on balls|hit by pitch', regex=True).astype(int)
    else:
        df['DERIVED_OUTS'] = 0
        df['DERIVED_HITS'] = 0
        df['DERIVED_WALKS'] = 0

    # Derive Runs Allowed
    bat_col = next((c for c in df.columns if 'BAT_SCORE' in c and 'POST' not in c), None)
    post_bat_col = next((c for c in df.columns if 'POST_BAT_SCORE' in c), None)
    er_col = next((c for c in df.columns if c in ['EARNED_RUNS', 'ER', 'RUNS_ALLOWED', 'R', 'RUNS']), None)

    if er_col:
        df['DERIVED_RUNS'] = pd.to_numeric(df[er_col], errors='coerce').fillna(0.0)
    elif bat_col and post_bat_col:
        score_diff = pd.to_numeric(df[post_bat_col], errors='coerce') - pd.to_numeric(df[bat_col], errors='coerce')
        df['DERIVED_RUNS'] = score_diff.clip(lower=0).fillna(0.0)
    elif event_candidate_cols:
        df['DERIVED_RUNS'] = combined_events.str.contains(r'home run|run|scores|sac fly', regex=True).astype(int)
    else:
        df['DERIVED_RUNS'] = 0.0

    p_stats = df.groupby('PITCHER_ID').agg(
        P_VELO_STD=('RELEASE_SPEED', lambda x: max(x.std(), 0.5)),
        P_SLOT_STD=('RELEASE_POS_Z', lambda x: max(x.std(), 0.04))
    ).reset_index()
    df = df.merge(p_stats, on='PITCHER_ID', how='left')

    if 'M1_FATIGUE_RISK_SCORE' in df.columns and 'FATIGUE_STREAK_3' not in df.columns:
        pg = df.groupby(['GAME_PK', 'PITCHER_ID'])
        df['FATIGUE_STREAK_3'] = pg['M1_FATIGUE_RISK_SCORE'].transform(
            lambda x: (x >= 0.2879).rolling(3, min_periods=3).sum() == 3
        ).fillna(0).astype(int)

    return df

df = load_data()

def format_player_name(val):
    v = str(val).strip()
    return f"{v.split(',')[1].strip()} {v.split(',')[0].strip()}" if ',' in v else v

pitcher_team_col = next((c for c in ['PITCHER_TEAM', 'TEAM', 'PITCH_TEAM'] if c in df.columns), None)
if pitcher_team_col:
    df['DERIVED_PITCHER_TEAM'] = df[pitcher_team_col].astype(str).str.strip()
elif 'INNING_TOPBOT' in df.columns and 'HOME_TEAM' in df.columns and 'AWAY_TEAM' in df.columns:
    df['DERIVED_PITCHER_TEAM'] = df.apply(
        lambda r: str(r['HOME_TEAM']).strip() if str(r.get('INNING_TOPBOT', '')).upper().startswith('TOP') 
        else str(r['AWAY_TEAM']).strip(), axis=1
    )
else:
    df['DERIVED_PITCHER_TEAM'] = df['HOME_TEAM'].astype(str).str.strip()

st.sidebar.header("⚾ Dugout Command Navigation")

# 1. Team Filter
teams = sorted([t for t in df['DERIVED_PITCHER_TEAM'].unique() if t not in ['nan', 'None', '', '<NA>']])
selected_team = st.sidebar.selectbox("Select Team", ["All Teams"] + teams) if teams else "All Teams"
team_df = df.copy() if selected_team == "All Teams" else df[df['DERIVED_PITCHER_TEAM'] == selected_team]

# 2. Starter vs Reliever Filter
role_choice = st.sidebar.radio("Pitcher Role Filter:", ["All Roles", "Starters", "Relievers"], horizontal=True)
if role_choice == "Starters":
    role_df = team_df[team_df['PITCHER_ROLE'] == 'Starter'].copy()
elif role_choice == "Relievers":
    role_df = team_df[team_df['PITCHER_ROLE'] == 'Reliever'].copy()
else:
    role_df = team_df.copy()

# 3. Pitcher Filter
available_pitchers = sorted(role_df['PITCHER_ID'].dropna().unique().astype(str).tolist())
pitcher_map = {format_player_name(p): p for p in available_pitchers}

search = st.sidebar.text_input("🔍 Search Pitcher:", "").strip().lower()
filtered_names = [n for n in pitcher_map.keys() if search in n.lower()] if search else list(pitcher_map.keys())

selected_pitcher_display = st.sidebar.selectbox("Select Pitcher", ["All Pitchers"] + filtered_names)
is_all_pitchers = (selected_pitcher_display == "All Pitchers")

if is_all_pitchers:
    player_df = role_df.copy()
else:
    selected_pitcher = pitcher_map[selected_pitcher_display]
    player_df = role_df[role_df['PITCHER_ID'].astype(str) == selected_pitcher].copy()

def make_game_label(r):
    gp, dt, h, a = int(r['GAME_PK']), str(r.get('GAME_DATE', '')).strip(), str(r.get('HOME_TEAM', '')).strip(), str(r.get('AWAY_TEAM', '')).strip()
    return f"{dt} | {a} @ {h} ({gp})" if h and a else f"{dt} Game PK: {gp}"

game_labels = player_df.drop_duplicates('GAME_PK').apply(make_game_label, axis=1)
game_options = dict(zip(game_labels, player_df.drop_duplicates('GAME_PK')['GAME_PK']))

selected_game_label = st.sidebar.selectbox("Select Game Outing", ["All Games"] + list(game_options.keys()))
is_all_games = (selected_game_label == "All Games")

if is_all_games:
    base_pitcher_df = player_df.sort_values(["PITCH_COUNT"]).copy()
else:
    selected_game = game_options[selected_game_label]
    base_pitcher_df = player_df[player_df['GAME_PK'] == selected_game].sort_values("PITCH_COUNT").copy()

st.sidebar.divider()
st.sidebar.header("⏱️ Pitch Interval Controls")

max_p_count = int(base_pitcher_df['PITCH_COUNT'].max()) if not base_pitcher_df.empty else 120
pitch_range = st.sidebar.slider(
    "Pitch Sequence Range:",
    min_value=1,
    max_value=max(max_p_count, 10),
    value=(1, max(max_p_count, 10)),
    help="Filter focus to a specific pitch sequence range (e.g., pitches 40 to 80)."
)

interval_bin_size = st.sidebar.selectbox(
    "Sequence Binning Interval:",
    options=[1, 5, 10, 15],
    index=0,
    format_func=lambda x: "Continuous (Pitch-by-Pitch)" if x == 1 else f"{x}-Pitch Bins",
    help="Aggregate continuous stream into sequence bins to smooth out noise."
)

pitcher_df = base_pitcher_df[
    (base_pitcher_df['PITCH_COUNT'] >= pitch_range[0]) & 
    (base_pitcher_df['PITCH_COUNT'] <= pitch_range[1])
].copy()

# Action Tier Filter
st.sidebar.divider()
st.sidebar.header("🏷️ Action Tier Filter")
if 'DUGOUT_ACTION' in df.columns:
    available_actions = sorted(df['DUGOUT_ACTION'].dropna().unique().tolist())
    selected_actions = st.sidebar.multiselect("Recommendation Category:", options=available_actions, default=available_actions)
    pitcher_df = pitcher_df[pitcher_df['DUGOUT_ACTION'].isin(selected_actions)].copy()

st.sidebar.divider()
st.sidebar.header("⚡ Risk Overlay Controls")

fatigue_alert_threshold = st.sidebar.slider("Highlight Risk Threshold:", 0, 100, 60, 1, format="%d%%") / 100.0

target_60_raw = st.sidebar.slider(
    "60% UI Fatigue Anchor (Raw Score):", 
    0.05, 0.40, 0.15, 0.01, 
    help="Determines what raw model risk score maps to 60% on the UI scale."
)

if 'M1_FATIGUE_RISK_SCORE' in pitcher_df.columns:
    pitcher_df['UI_FATIGUE_SCORE'] = (pitcher_df['M1_FATIGUE_RISK_SCORE'] / target_60_raw) * 0.60
    pitcher_df['UI_FATIGUE_SCORE'] = pitcher_df['UI_FATIGUE_SCORE'].clip(upper=0.99)
else:
    pitcher_df['UI_FATIGUE_SCORE'] = 0.0

pitcher_df['ACTIVE_FATIGUE_METRIC'] = pitcher_df['UI_FATIGUE_SCORE']

if interval_bin_size > 1:
    pitcher_df['PITCH_BIN'] = ((pitcher_df['PITCH_COUNT'] - 1) // interval_bin_size) * interval_bin_size + (interval_bin_size // 2) + 1
else:
    pitcher_df['PITCH_BIN'] = pitcher_df['PITCH_COUNT']

st.title("⚾ MLB Pitch-by-Pitch Dugout Fatigue Engine")

context_str = f"**Team:** `{selected_team}` | **Role:** `{role_choice}` | **Pitcher:** `{selected_pitcher_display}` | **Game:** `{selected_game_label}`"
st.markdown(context_str)

# Outing Peak and Recent Trend calculations
recent_10_pitches = pitcher_df.tail(10)
recent_max_m1 = recent_10_pitches['M1_FATIGUE_RISK_SCORE'].max() if 'M1_FATIGUE_RISK_SCORE' in pitcher_df.columns else 0.0
outing_peak_m1 = pitcher_df['M1_FATIGUE_RISK_SCORE'].max() if 'M1_FATIGUE_RISK_SCORE' in pitcher_df.columns else 0.0
outing_peak_scaled = pitcher_df['UI_FATIGUE_SCORE'].max() if 'UI_FATIGUE_SCORE' in pitcher_df.columns else 0.0
recent_streak_active = (recent_10_pitches['FATIGUE_STREAK_3'].sum() >= 1) if 'FATIGUE_STREAK_3' in pitcher_df.columns else False

latest_pitch = pitcher_df.iloc[-1] if not pitcher_df.empty else {}
dugout_action = latest_pitch.get("DUGOUT_ACTION", "GREEN: NORMAL")

# Velo and Arm Slot Delta calculations
primary_pt = pitcher_df['PITCH_TYPE'].mode()[0] if not pitcher_df['PITCH_TYPE'].empty else 'FF'

if is_all_games or is_all_pitchers:
    early_df = pitcher_df[pitcher_df['PITCH_COUNT'] <= 15]
    late_df = pitcher_df[pitcher_df['PITCH_COUNT'] >= 60]
    
    early_velo = early_df[early_df['PITCH_TYPE'] == primary_pt]['RELEASE_SPEED'].mean()
    late_velo = late_df[late_df['PITCH_TYPE'] == primary_pt]['RELEASE_SPEED'].mean() if not late_df.empty else pitcher_df[pitcher_df['PITCH_TYPE'] == primary_pt]['RELEASE_SPEED'].mean()
    velo_delta = (late_velo - early_velo) if pd.notnull(late_velo) and pd.notnull(early_velo) else 0.0

    early_rel_z = early_df['RELEASE_POS_Z'].mean()
    late_rel_z = late_df['RELEASE_POS_Z'].mean() if not late_df.empty else pitcher_df['RELEASE_POS_Z'].mean()
    rel_z_delta = (late_rel_z - early_rel_z) if pd.notnull(late_rel_z) and pd.notnull(early_rel_z) else 0.0
else:
    inn1_df = pitcher_df[pitcher_df['INNING'] == pitcher_df['INNING'].min()]
    inn1_velo = inn1_df[inn1_df['PITCH_TYPE'] == primary_pt]['RELEASE_SPEED'].mean()
    recent_velo = recent_10_pitches[recent_10_pitches['PITCH_TYPE'] == primary_pt]['RELEASE_SPEED'].mean()
    velo_delta = (recent_velo - inn1_velo) if pd.notnull(recent_velo) and pd.notnull(inn1_velo) else 0.0

    inn1_rel_z = inn1_df['RELEASE_POS_Z'].mean()
    recent_rel_z = recent_10_pitches['RELEASE_POS_Z'].mean()
    rel_z_delta = (recent_rel_z - inn1_rel_z) if pd.notnull(recent_rel_z) and pd.notnull(inn1_rel_z) else 0.0

# Calculate Run Expectancy Delta per Outing
num_outings = max(pitcher_df.groupby(['GAME_PK', 'PITCHER_ID']).ngroups, 1)
total_run_exp = pitcher_df['DERIVED_RUN_EXP_DELTA'].sum() if 'DERIVED_RUN_EXP_DELTA' in pitcher_df.columns else 0.0
outing_run_exp = total_run_exp / num_outings

avg_xba = pitcher_df['DERIVED_XBA'].mean() if 'DERIVED_XBA' in pitcher_df.columns else np.nan

total_outs = pitcher_df['DERIVED_OUTS'].sum() if 'DERIVED_OUTS' in pitcher_df.columns else 0
total_runs = pitcher_df['DERIVED_RUNS'].sum() if 'DERIVED_RUNS' in pitcher_df.columns else 0
total_hits = pitcher_df['DERIVED_HITS'].sum() if 'DERIVED_HITS' in pitcher_df.columns else 0
total_walks = pitcher_df['DERIVED_WALKS'].sum() if 'DERIVED_WALKS' in pitcher_df.columns else 0

# Fallback for outs if event text was missing or sparse
if total_outs == 0 and not pitcher_df.empty:
    total_outs = len(pitcher_df) / 5.0  # Approx 5 pitches per out baseline

innings_pitched = total_outs / 3.0
calculated_era = (total_runs / max(innings_pitched, 0.33)) * 9.0 if innings_pitched > 0 else 0.0
calculated_whip = ((total_hits + total_walks) / max(innings_pitched, 0.33)) if innings_pitched > 0 else 0.0

total_swings = pitcher_df['IS_SWING'].sum() if 'IS_SWING' in pitcher_df.columns else 0
total_whiffs = pitcher_df['IS_WHIFF'].sum() if 'IS_WHIFF' in pitcher_df.columns else 0
whiff_rate = (total_whiffs / max(total_swings, 1)) if total_swings > 0 else 0.0

max_tto = pitcher_df['N_THRUORDER_PITCHER'].max() if 'N_THRUORDER_PITCHER' in pitcher_df.columns else 1

# Decision Banner
if is_all_games or is_all_pitchers:
    st.info(f"📊 **AGGREGATE PROFILE:** Analyzing **{len(pitcher_df):,}** pitches across **{pitcher_df.groupby(['GAME_PK', 'PITCHER_ID']).ngroups:,}** outings for `{selected_team} ({role_choice}) / {selected_pitcher_display}`.")
else:
    if "RED" in dugout_action or (recent_streak_active and recent_max_m1 >= 0.45) or outing_peak_m1 >= 0.70:
        st.error(f"🚨 **DUGOUT ACTION REQUIRED: PULL PITCHER IMMEDIATELY** | Sustained High Risk (3-Pitch Streak) | Peak Fatigue: `{outing_peak_scaled:.1%}`")
    elif "ORANGE" in dugout_action or recent_max_m1 >= 0.35:
        st.warning(f"⚠️ **DUGOUT ACTION RECOMMENDED: WARM UP RELIEVER** | Recent Fatigue Spike: `{outing_peak_scaled:.1%}`")
    elif "YELLOW" in dugout_action or recent_max_m1 >= 0.28:
        st.info(f"🟡 **MONITOR CLOSELY: PROACTIVE WARNING ZONE** | Peak Fatigue: `{outing_peak_scaled:.1%}` | Pitch Count: {int(latest_pitch.get('PITCH_COUNT', 0))}")
    else:
        st.success(f"🟢 **STATUS NORMAL: PITCHER FRESH** | Peak Fatigue: `{outing_peak_scaled:.1%}` | Baseline Signals Stable")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Pitches Analyzed", f"{len(pitcher_df):,}")
k2.metric("Peak Fatigue Score", f"{outing_peak_scaled:.1%}")
k3.metric(f"Velo Delta ({primary_pt})", f"{velo_delta:+.1f} mph", delta_color="normal" if velo_delta >= -1.0 else "inverse")
k4.metric("Arm Slot Drop", f"{rel_z_delta:+.2f} ft", delta_color="normal" if rel_z_delta >= -0.1 else "inverse")
k5.metric("Calculated ERA", f"{calculated_era:.2f}" if innings_pitched > 0 else "N/A")
k6.metric("Calculated WHIP", f"{calculated_whip:.2f}" if innings_pitched > 0 else "N/A")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Whiff %", f"{whiff_rate:.1%}")
m2.metric("Avg xBA", f"{avg_xba:.3f}" if pd.notnull(avg_xba) else "N/A")
m3.metric("Run Exp Delta / Outing", f"{outing_run_exp:+.3f}" if pd.notnull(outing_run_exp) else "+0.000")
m4.metric("Times Thru Order", f"{int(max_tto)}x" if pd.notnull(max_tto) else "1x")
m5.metric("Outings Count", f"{pitcher_df.groupby(['GAME_PK', 'PITCHER_ID']).ngroups:,}")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "⚡ Live Pitch-by-Pitch & Interval Stream", 
    "🎯 In-Game Tactical & Mechanical Drift", 
    "📊 Repertoire Health & Outcome Metrics"
])

with tab1:
    st.subheader("📉 Pitch-by-Pitch Fatigue Decay & Physical Signals Stream")

    fig_stream = make_subplots(specs=[[{"secondary_y": True}]])
    x_col = 'PITCH_BIN' if interval_bin_size > 1 else 'PITCH_COUNT'
    
    if is_all_games or is_all_pitchers or interval_bin_size > 1:
        agg_cols = {'RELEASE_SPEED': 'mean', 'UI_FATIGUE_SCORE': 'mean', 'GAME_PK': 'count'}
        plot_df = pitcher_df.groupby([x_col, 'PITCH_TYPE']).agg(agg_cols).reset_index()
        fatigue_plot_df = pitcher_df.groupby(x_col).agg({'UI_FATIGUE_SCORE': 'mean', 'GAME_PK': 'count'}).reset_index()
    else:
        plot_df = pitcher_df.copy()
        fatigue_plot_df = pitcher_df.copy()

    for pt in sorted(plot_df['PITCH_TYPE'].unique()):
        pt_data = plot_df[plot_df['PITCH_TYPE'] == pt].sort_values(x_col)
        fig_stream.add_trace(
            go.Scatter(
                x=pt_data[x_col], y=pt_data['RELEASE_SPEED'],
                name=f"{pt} Velo",
                mode="lines+markers",
                connectgaps=True,
                line=dict(color=COLOR_MAP.get(pt, "#7f7f7f"), width=2),
                marker=dict(size=5),
                hovertemplate="Pitch %{x}<br>Release Speed: %{y:.1f} mph<br>Type: " + pt
            ), secondary_y=False
        )

    # Primary pitch velocity trend slope
    primary_pt_data = plot_df[plot_df['PITCH_TYPE'] == primary_pt]
    if len(primary_pt_data) >= 3:
        x_pt = primary_pt_data[x_col].values
        y_pt = primary_pt_data['RELEASE_SPEED'].values
        slope_pt, intercept_pt = np.polyfit(x_pt, y_pt, 1)
        trend_y_pt = slope_pt * x_pt + intercept_pt
        
        fig_stream.add_trace(
            go.Scatter(
                x=x_pt, y=trend_y_pt,
                name=f"{primary_pt} Slope ({slope_pt:+.3f} mph/pitch)",
                mode="lines",
                line=dict(color="darkred", width=2, dash="dash"),
                hovertemplate=f"{primary_pt} Slope: {slope_pt:+.3f} mph/pitch"
            ), secondary_y=False
        )

    if 'UI_FATIGUE_SCORE' in fatigue_plot_df.columns and not fatigue_plot_df.empty:
        x_all = fatigue_plot_df[x_col].values
        y_fatigue = fatigue_plot_df['UI_FATIGUE_SCORE'].values
        
        if len(x_all) >= 3:
            slope_f, intercept_f = np.polyfit(x_all, y_fatigue, 1)
        else:
            slope_f, intercept_f = (0.002, 0.05)

        trend_y_f = slope_f * x_all + intercept_f

        slope_10p = slope_f * 10 * 100
        rate_label = f"Fatigue Rate ({slope_10p:+.1f}% / 10 pitches)"

        fig_stream.add_trace(
            go.Scatter(
                x=x_all, y=trend_y_f,
                name=rate_label,
                mode="lines",
                line=dict(color="black", width=2, dash="dashdot"),
                hovertemplate=f"{rate_label}"
            ), secondary_y=True
        )

    # High-risk alert markers in single game view
    if not (is_all_games or is_all_pitchers) and 'FATIGUE_STREAK_3' in pitcher_df.columns:
        high_risk_pitches = pitcher_df[(pitcher_df['UI_FATIGUE_SCORE'] >= fatigue_alert_threshold) & (pitcher_df['FATIGUE_STREAK_3'] == 1)]
        if not high_risk_pitches.empty:
            fig_stream.add_trace(
                go.Scatter(
                    x=high_risk_pitches[x_col], y=high_risk_pitches['RELEASE_SPEED'],
                    name=f"Confirmed High Risk (≥{fatigue_alert_threshold:.0%})", mode="markers",
                    marker=dict(color="red", size=11, symbol="x", line=dict(width=2, color="darkred")),
                    hovertemplate="Pitch %{x}<br>Risk: %{customdata:.1%}",
                    customdata=high_risk_pitches['UI_FATIGUE_SCORE']
                ), secondary_y=False
            )

    # Scaled Fatigue Risk Curve Trace
    if 'UI_FATIGUE_SCORE' in fatigue_plot_df.columns:
        fig_stream.add_trace(
            go.Scatter(
                x=fatigue_plot_df[x_col], y=fatigue_plot_df['UI_FATIGUE_SCORE'],
                name="Scaled Fatigue Risk",
                mode="lines",
                line=dict(color="black", width=2.5),
                opacity=0.85,
                hovertemplate="Pitch %{x}<br>Scaled Fatigue Risk: %{y:.1%}"
            ), secondary_y=True
        )

    fig_stream.update_layout(
        title=f"<b>Pitch Progression ({selected_pitcher_display} | Role: {role_choice} | {selected_game_label})</b>",
        template="plotly_white", height=520, hovermode="x unified"
    )
    fig_stream.update_xaxes(title_text="Outing Pitch Count Sequence" if interval_bin_size == 1 else f"Sequence Bins ({interval_bin_size} Pitches)")
    fig_stream.update_yaxes(title_text="Release Speed (mph)", secondary_y=False)
    fig_stream.update_yaxes(title_text="Scaled Fatigue Risk", secondary_y=True, range=[0, 1.0])
    
    st.plotly_chart(fig_stream, use_container_width=True)

    st.subheader("📜 Pitch Stream Feed Data")
    feed_cols = ['PITCH_COUNT', 'INNING', 'PITCHER_ID', 'PITCHER_ROLE', 'PITCH_TYPE', 'RELEASE_SPEED', 'RELEASE_SPIN_RATE', 'RELEASE_POS_Z', 'DERIVED_XBA', 'UI_FATIGUE_SCORE', 'DUGOUT_ACTION']
    
    display_feed = pitcher_df[[c for c in feed_cols if c in pitcher_df.columns]].sort_values("PITCH_COUNT", ascending=False)
    st.dataframe(display_feed, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🎯 In-Game Leverage & Mechanical Drift Analysis")
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_z = px.scatter(
            pitcher_df, x="PITCH_COUNT", y="RELEASE_POS_Z", color="PITCH_TYPE",
            size="UI_FATIGUE_SCORE" if 'UI_FATIGUE_SCORE' in pitcher_df.columns else None,
            color_discrete_map=COLOR_MAP,
            title="<b>Arm Slot Height Drift (Release Z)</b>", template="plotly_white"
        )
        st.plotly_chart(fig_z, use_container_width=True)

    with col_b:
        if 'N_THRUORDER_PITCHER' in pitcher_df.columns:
            fig_tto = px.box(
                pitcher_df, x="N_THRUORDER_PITCHER", y="ACTIVE_FATIGUE_METRIC", color="N_THRUORDER_PITCHER",
                title="<b>Fatigue Accumulation by Times Through Order (TTO)</b>", template="plotly_white",
                labels={'N_THRUORDER_PITCHER': 'Times Through Lineup', 'ACTIVE_FATIGUE_METRIC': 'Fatigue Metric'}
            )
            st.plotly_chart(fig_tto, use_container_width=True)
        else:
            fig_release = px.scatter(
                pitcher_df, x="RELEASE_POS_X", y="RELEASE_POS_Z", color="INNING",
                size="UI_FATIGUE_SCORE" if 'UI_FATIGUE_SCORE' in pitcher_df.columns else None,
                color_continuous_scale="Reds",
                title="<b>Arm Slot Drift Scatter (Release X vs Z)</b>", template="plotly_white"
            )
            st.plotly_chart(fig_release, use_container_width=True)

with tab3:
    st.subheader("📊 Pitch Repertoire & In-Game Outcome Summary")
    rep_summary = []
    
    for pt in pitcher_df['PITCH_TYPE'].unique():
        pt_p = pitcher_df[pitcher_df['PITCH_TYPE'] == pt]
        usage_pct = len(pt_p) / max(len(pitcher_df), 1)
        
        e_v = pt_p["RELEASE_SPEED"].iloc[:3].mean() if len(pt_p) >= 3 else pt_p["RELEASE_SPEED"].mean()
        l_v = pt_p["RELEASE_SPEED"].iloc[-3:].mean() if len(pt_p) >= 3 else pt_p["RELEASE_SPEED"].mean()
        e_s = pt_p["RELEASE_SPIN_RATE"].iloc[:3].mean() if len(pt_p) >= 3 else pt_p["RELEASE_SPIN_RATE"].mean()
        l_s = pt_p["RELEASE_SPIN_RATE"].iloc[-3:].mean() if len(pt_p) >= 3 else pt_p["RELEASE_SPIN_RATE"].mean()
        
        pt_swings = pt_p['IS_SWING'].sum() if 'IS_SWING' in pt_p.columns else 0
        pt_whiffs = pt_p['IS_WHIFF'].sum() if 'IS_WHIFF' in pt_p.columns else 0
        pt_whiff_rate = (pt_whiffs / max(pt_swings, 1)) if pt_swings > 0 else 0.0
        
        pt_xba = pt_p['DERIVED_XBA'].mean() if 'DERIVED_XBA' in pt_p.columns else np.nan
        
        rep_summary.append({
            "Pitch Type": pt, "Count": len(pt_p), "Usage %": f"{usage_pct:.1%}",
            "Early Velo": f"{e_v:.1f}" if pd.notnull(e_v) else "N/A", 
            "Current Velo": f"{l_v:.1f}" if pd.notnull(l_v) else "N/A", 
            "Velo Delta": f"{l_v - e_v:+.1f} mph" if pd.notnull(l_v) and pd.notnull(e_v) else "N/A",
            "Whiff %": f"{pt_whiff_rate:.1%}",
            "Avg xBA": f"{pt_xba:.3f}" if pd.notnull(pt_xba) else "N/A",
            "Early Spin": f"{int(e_s)}" if pd.notnull(e_s) else "N/A", 
            "Current Spin": f"{int(l_s)}" if pd.notnull(l_s) else "N/A"
        })

    if rep_summary:
        st.dataframe(pd.DataFrame(rep_summary), use_container_width=True, hide_index=True)