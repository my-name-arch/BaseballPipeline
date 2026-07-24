SELECT
    BATTER_NAME,
    STAND AS BATTER_STAND,
    AGE_BAT AS BATTER_AGE
FROM {{ ref('stg_statcast_pitches') }}
GROUP BY 1, 2, 3