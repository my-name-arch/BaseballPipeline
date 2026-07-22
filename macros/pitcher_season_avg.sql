{% macro pitcher_season_avg(metric_column, game_year, pitcher_id,pitch_type, precision=1) %}
    ROUND(
        AVG({{ metric_column }}) OVER (
            PARTITION BY {{ game_year }}, {{ pitcher_id }},{{pitch_type}}
        ),
        {{ precision }}
    )
{% endmacro %}