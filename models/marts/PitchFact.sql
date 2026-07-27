{{ config(
    materialized='table'
) }}

WITH parsed AS (
    SELECT * 
    FROM {{ ref('stg_statcast_pitches') }}
)

SELECT
    -- 🔑 1. Primary Keys & Unique Identifiers
    MD5(CONCAT(
        COALESCE(p.GAME_DATE, ''), '_', 
        COALESCE(p.PITCHER_NAME, ''), '_', 
        COALESCE(CAST(p.AT_BAT_NUMBER AS VARCHAR), ''), '_', 
        COALESCE(CAST(p.PITCH_NUMBER AS VARCHAR), '')
    )) AS PITCH_SK,

    p.GAME_PK,
    p.GAME_DATE,
    p.PITCHER_NAME AS PITCHER_ID,
    p.BATTER_NAME AS BATTER_ID,

    --  2. Game Context & Sequence
    p.INNING,
    p.INNING_TOPBOT,
    p.AT_BAT_NUMBER,
    p.PITCH_NUMBER,
    p.PITCH_COUNT,
    p.OUTS_WHEN_UP,
    p.BALLS,
    p.STRIKES,
    p.N_THRUORDER_PITCHER,
    p.N_PRIORPA_THISGAME_PLAYER_AT_BAT,

    --  3. Game State & Defense
    p.HOME_SCORE,
    p.AWAY_SCORE,
    p.BAT_SCORE_DIFF,
    p.IF_FIELDING_ALIGNMENT,
    p.OF_FIELDING_ALIGNMENT,
    p.PITCHER_DAYS_SINCE_PREV_GAME,

    --  4. Pitch Characteristics & Movement (Kinematics)
    p.PITCH_TYPE,
    p.RELEASE_SPEED,
    p.RELEASE_SPIN_RATE,
    p.SPIN_AXIS,
    p.RELEASE_EXTENSION,
    p.RELEASE_POS_X,
    p.RELEASE_POS_Y,
    p.RELEASE_POS_Z,
    p.ZONE,
    
    -- Velocity & Acceleration Vectors
    p.INITIAL_VELOCITY_HORIZONTAL AS VX0,
    p.INITIAL_VELOCITY_VERTICAL AS VZ0,
    p.ACCELERATION_HORIZONTAL AS AX,
    p.ACCELERATION_VERTICAL AS AZ,
    
    -- API Break Profile
    p.API_BREAK_Z_WITH_GRAVITY,
    p.API_BREAK_X_ARM,
    p.API_BREAK_X_BATTER_IN,

    --  5. Pitch Outcome & Contact Metrics
    p.PITCH_RESULT,
    p.TYPE AS PITCH_CATEGORY,
    p.PLAY_DESCRIPTION,
    p.HIT_LOCATION,
    p.LAUNCH_SPEED,
    p.LAUNCH_ANGLE,
    p.HIT_DISTANCE_SC,
    p.LAUNCH_SPEED_ANGLE,

    --  6. Advanced Value & Expected Metrics (xStats)
    p.ESTIMATED_BA_USING_SPEEDANGLE AS XBA,
    p.ESTIMATED_WOBA_USING_SPEEDANGLE AS XWOBA,
    p.ESTIMATED_SLG_USING_SPEEDANGLE AS XSLG,
    p.WOBA_VALUE,
    p.BABIP_VALUE,
    p.DELTA_PITCHER_RUN_EXP,

    --  7. Macro Averages & Benchmarks
    p.GAME_SPEED,
    p.GAME_SPIN,
    p.GAME_AVG_SPEED,
    p.GAME_AVG_SPIN,
    p.SEASON_SPEED,
    p.SEASON_SPIN,
    p.SEASON_AVG_SPEED,
    p.SEASON_AVG_SPIN

FROM parsed p
ORDER BY p.API_BREAK_X_BATTER_IN DESC