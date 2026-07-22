{% macro pitcher_game_avg(metric_column, game_pk, pitcher_id,pitch_type, precision=1) %}
    ROUND(
        AVG({{ metric_column }}) OVER (
            PARTITION BY {{ game_pk }}, {{ pitcher_id }},{{pitch_type}}
        ),
        {{ precision }}
    )
{% endmacro %}