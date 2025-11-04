#version 420 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 colour;
layout(location = 2) in vec3 normal;

out vec4 f_base_colour;
out vec3 f_position;
out vec3 f_light_direction;
out vec3 f_normal;

uniform mat4 model, view, projection;
uniform mat3 normalmatrix;
uniform vec4 lightpos;

vec3 global_ambient = vec3(0.2, 0.2, 0.2);
int shininess = 8;

void main()
{
    vec4 position_h = vec4(position, 1.0);
    vec4 diffuse_albedo = colour;

    mat4 mv_matrix = view * model;
    vec4 P = mv_matrix * position_h;
    vec3 N = normalize(normalmatrix * normal);
    vec3 light_pos3 = lightpos.xyz;
    vec3 L = light_pos3 - P.xyz;

    f_base_colour = diffuse_albedo;

    f_position = P.xyz;
    f_light_direction = L;
    f_normal = N;

    gl_Position = projection * mv_matrix * position_h;
}