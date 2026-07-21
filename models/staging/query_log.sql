WITH raw_pitches AS (
    SELECT * 
    FROM BASEBALL_DB.RAW.STATCAST_PITCHES
),

parsed AS (
    SELECT 
        -- Batter & Pitcher
        COALESCE(batter.BATTER_NAME, 'Unknown Batter') AS BATTER_NAME,
        p.PLAYER_NAME AS PITCHER_NAME,

        -- Simplified & cleaned pitch result logic
        REPLACE(
            CASE 
                -- Specific field out short-names
                WHEN NULLIF(p.EVENTS, 'None') = 'field_out' AND p.BB_TYPE = 'ground_ball' THEN 'ground_out'
                WHEN NULLIF(p.EVENTS, 'None') = 'field_out' AND p.BB_TYPE = 'line_drive' THEN 'line_out'
                WHEN NULLIF(p.EVENTS, 'None') = 'field_out' AND p.BB_TYPE = 'fly_ball' THEN 'fly_out'
                WHEN NULLIF(p.EVENTS, 'None') = 'field_out' AND p.BB_TYPE = 'popup' THEN 'pop_out'

                -- Any other generic field outs
                WHEN NULLIF(p.EVENTS, 'None') = 'field_out' AND NULLIF(p.BB_TYPE, 'None') IS NOT NULL THEN 
                    REPLACE(p.BB_TYPE, '_ball', '') || ' out'

                -- Direct outcomes (single, double, triple, home_run, walk, strikeout, etc.)
                WHEN NULLIF(p.EVENTS, 'None') IS NOT NULL THEN 
                    p.EVENTS

                -- Fallback if EVENTS is NULL/None (batted ball type or pitch description)
                ELSE COALESCE(
                    REPLACE(NULLIF(p.BB_TYPE, 'None'), '_ball', ''),
                    NULLIF(p.DESCRIPTION, 'None')
                )
            END,
            '_', ' '
        ) AS PITCH_RESULT,
        
        p.DES AS PLAY_DESCRIPTION,
        
        -- Exclude raw ID & outcome columns to keep table clean
        p.* EXCLUDE (
            BATTER, FIELDER_2, FIELDER_3, FIELDER_4, FIELDER_5, 
            FIELDER_6, FIELDER_7, FIELDER_8, FIELDER_9,
            ON_1B, ON_2B, ON_3B, PITCHER, PFX_X, PFX_Z, PLATE_X, PLATE_Z,
            HC_X, HC_Y, TFS_DEPRECATED, TFS_ZULU_DEPRECATED, UMPIRE, PLAYER_NAME,
            SV_ID, iso_value, delta_home_win_exp, delta_run_exp, age_pit_legacy, age_bat_legacy,
            batter_days_since_prev_game, batter_days_until_next_game, pitcher_days_until_next_game,
            attack_angle, attack_direction, swing_path_tilt, intercept_ball_minus_batter_pos_x_inches,
            intercept_ball_minus_batter_pos_y_inches, BAT_SCORE, FLD_SCORE, POST_BAT_SCORE, POST_FLD_SCORE,DES,BB_TYPE,EVENTS,description,SPIN_DIR,SPIN_RATE_DEPRECATED,BREAK_ANGLE_DEPRECATED,BREAK_LENGTH_DEPRECATED
        )
    FROM raw_pitches p
    
    -- Batter
    LEFT JOIN BASEBALL_DB.RAW.DIM_BATTERS batter
        ON p.BATTER = batter.BATTER_ID
)

SELECT *
FROM parsed