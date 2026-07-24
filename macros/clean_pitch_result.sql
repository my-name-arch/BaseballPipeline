{% macro clean_pitch_result(events, bb_type, description) %}
    REPLACE(
        CASE 
            -- Specific field out short-names
            WHEN NULLIF({{ events }}, 'None') = 'field_out' AND {{ bb_type }} = 'ground_ball' THEN 'ground_out'
            WHEN NULLIF({{ events }}, 'None') = 'field_out' AND {{ bb_type }} = 'line_drive'  THEN 'line_out'
            WHEN NULLIF({{ events }}, 'None') = 'field_out' AND {{ bb_type }} = 'fly_ball'    THEN 'fly_out'
            WHEN NULLIF({{ events }}, 'None') = 'field_out' AND {{ bb_type }} = 'popup'       THEN 'pop_out'

            -- Any other generic field outs
            WHEN NULLIF({{ events }}, 'None') = 'field_out' AND NULLIF({{ bb_type }}, 'None') IS NOT NULL 
                THEN REPLACE({{ bb_type }}, '_ball', '') || '_out'

            -- Direct outcomes (single, double, triple, home_run, walk, strikeout, etc.)
            WHEN NULLIF({{ events }}, 'None') IS NOT NULL 
                THEN {{ events }}

            -- Fallback if EVENTS is NULL/None (batted ball type or pitch description)
            ELSE COALESCE(
                REPLACE(NULLIF({{ bb_type }}, 'None'), '_ball', ''),
                NULLIF({{ description }}, 'None')
            )
        END,
        '_', ' '
    )
{% endmacro %}


{% macro pitcher_pitch_count(game_pk, pitcher_id, at_bat_num, pitch_num) %}
    ROW_NUMBER() OVER (
        PARTITION BY {{ game_pk }}, {{ pitcher_id }} 
        ORDER BY {{ at_bat_num }} ASC, {{ pitch_num }} ASC
    )
{% endmacro %}


{% macro pitcher_game_avg(metric, game_pk, pitcher, pitch_type=None, round_decimal=1) %}
    ROUND(
        AVG({{ metric }}) OVER (
            PARTITION BY {{ game_pk }}, {{ pitcher }}
            {% if pitch_type is not none %}
                , {{ pitch_type }}
            {% endif %}
        ),
        {{ round_decimal }}
    )
{% endmacro %}


{% macro pitcher_season_avg(metric_column, game_year, pitcher_id, pitch_type=None, precision=1) %}
    ROUND(
        AVG({{ metric_column }}) OVER (
            PARTITION BY {{ game_year }}, {{ pitcher_id }}
            {% if pitch_type is not none %}
                , {{ pitch_type }}
            {% endif %}
        ),
        {{ precision }}
    )
{% endmacro %}