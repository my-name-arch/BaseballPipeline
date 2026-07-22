{% macro pitcher_pitch_count(game_pk, pitcher_id, at_bat_num, pitch_num) %}
    ROW_NUMBER() OVER (
        PARTITION BY {{ game_pk }}, {{ pitcher_id }} 
        ORDER BY {{ at_bat_num }} ASC, {{ pitch_num }} ASC
    )
{% endmacro %}