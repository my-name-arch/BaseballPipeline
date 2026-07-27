{% macro clean_pitch_result(events, bb_type, description) %}
    REPLACE(
        CASE 
            -- Specific field out short-names
            WHEN {{ events }} = 'field_out' AND {{ bb_type }} = 'ground_ball' THEN 'ground_out'
            WHEN {{ events }} = 'field_out' AND {{ bb_type }} = 'line_drive'  THEN 'line_out'
            WHEN {{ events }} = 'field_out' AND {{ bb_type }} = 'fly_ball'    THEN 'fly_out'
            WHEN {{ events }} = 'field_out' AND {{ bb_type }} = 'popup'       THEN 'pop_out'


            -- Fallback if EVENTS is NULL/None (batted ball type or pitch description)
            ELSE COALESCE(
                REPLACE(NULLIF({{ bb_type }}, 'None'), '_ball', ''),
                NULLIF({{ description }}, 'None')
            )
        END,
        '_', ' '
    )
{% endmacro %}

