WITH raw_pitches AS (
    SELECT * 
    FROM BASEBALL_DB.RAW.STATCAST_PITCHES
    WHERE GAME_TYPE = 'R'
      AND RELEASE_SPEED IS NOT NULL
      AND RELEASE_SPIN_RATE IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY GAME_PK, AT_BAT_NUMBER, PITCH_NUMBER 
        ORDER BY NULL
    ) = 1
),

dim_batters_unique AS (
    SELECT 
        BATTER_ID,
        MAX(BATTER_NAME) AS BATTER_NAME
    FROM BASEBALL_DB.RAW.DIM_BATTERS
    GROUP BY BATTER_ID
),

parsed AS (
    SELECT 
        -- Batter & Pitcher
        COALESCE(batter.BATTER_NAME, 'Unknown Batter') AS BATTER_NAME,
        p.PLAYER_NAME AS PITCHER_NAME,

        TO_VARCHAR(TRY_TO_DATE(p.GAME_DATE::VARCHAR), 'MM/DD/YYYY') AS GAME_DATE,
        
        -- Pitcher Game Pitch Count
        {{ pitcher_pitch_count('p.GAME_PK', 'p.PITCHER', 'p.AT_BAT_NUMBER', 'p.PITCH_NUMBER') }} AS PITCH_COUNT,

        -- Clean Pitch Result
        {{ clean_pitch_result('p.EVENTS', 'p.BB_TYPE', 'p.DESCRIPTION') }} AS PITCH_RESULT,
        
        -- Game & Season Averages (By Pitch Type)
        {{ pitcher_game_avg('p.RELEASE_SPEED', 'p.GAME_PK', 'p.PITCHER', 'p.PITCH_TYPE', 1) }} AS GAME_AVG_SPEED,
        {{ pitcher_game_avg('p.RELEASE_SPIN_RATE', 'p.GAME_PK', 'p.PITCHER', 'p.PITCH_TYPE', 0) }} AS GAME_AVG_SPIN,
        {{ pitcher_season_avg('p.RELEASE_SPEED', 'p.GAME_YEAR', 'p.PITCHER', 'p.PITCH_TYPE', 1) }} AS SEASON_AVG_SPEED,
        {{ pitcher_season_avg('p.RELEASE_SPIN_RATE', 'p.GAME_YEAR', 'p.PITCHER', 'p.PITCH_TYPE', 0) }} AS SEASON_AVG_SPIN,  
        
        -- Overall Game & Season Averages (All Pitches Combined)
        {{ pitcher_game_avg('p.RELEASE_SPEED', 'p.GAME_PK', 'p.PITCHER', 1) }} AS GAME_SPEED,
        {{ pitcher_game_avg('p.RELEASE_SPIN_RATE', 'p.GAME_PK', 'p.PITCHER', 0) }} AS GAME_SPIN,
        {{ pitcher_season_avg('p.RELEASE_SPEED', 'p.GAME_YEAR', 'p.PITCHER', 1) }} AS SEASON_SPEED,
        {{ pitcher_season_avg('p.RELEASE_SPIN_RATE', 'p.GAME_YEAR', 'p.PITCHER', 0) }} AS SEASON_SPIN,  

        {{ rename_kinematics('p.VX0', 'p.VZ0', 'p.AX', 'p.AZ') }},

        p.DES AS PLAY_DESCRIPTION,
        
        p.* EXCLUDE (
            {{ statcast_excluded_columns() }}
        )
    FROM raw_pitches p
    LEFT JOIN dim_batters_unique batter
        ON p.BATTER = batter.BATTER_ID
)

SELECT *
FROM parsed
ORDER BY GAME_SPIN DESC