{% macro pitcher_game_avg(metric_column, game_pk, pitcher_id, pitch_type=None, precision=1) %}
    ROUND(
        AVG({{ metric_column }}) OVER (
            PARTITION BY {{ game_pk }}, {{ pitcher_id }}
            {%- if pitch_type is not none and pitch_type != '' %}
                , {{ pitch_type }}
            {%- endif %}
        ),
        {{ precision }}
    )
{% endmacro %}