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
                THEN REPLACE({{ bb_type }}, '_ball', '') || ' out'

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