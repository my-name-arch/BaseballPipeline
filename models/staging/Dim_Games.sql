SELECT
    GAME_PK,
    TRY_TO_DATE(GAME_DATE::VARCHAR) AS GAME_DATE,
    GAME_YEAR,
    GAME_TYPE,
    HOME_TEAM,
    AWAY_TEAM
FROM {{ ref('stg_statcast_pitches') }}
GROUP BY 1, 2, 3, 4, 5, 6