{% macro rename_kinematics(vx,vz,ax, az) %}
    {{ vx }} AS INITIAL_VELOCITY_HORIZONTAL,
    {{ vz }} AS INITIAL_VELOCITY_VERTICAL,
    {{ ax }} AS ACCELERATION_HORIZONTAL,
    {{ az }} AS ACCELERATION_VERTICAL
{% endmacro %}