#version 420 core

in vec4 f_base_colour;
in vec3 f_position;
in vec3 f_light_direction;
in vec3 f_normal;

out vec4 outputColor;

int shininess = 8;

const vec3 global_ambient = vec3(0.25, 0.25, 0.25);
vec3 specular_albedo = vec3(1.0, 0.8, 0.6);

uniform uint emitmode;
uniform uint attenuationmode;

void main()
{
    vec3 N = normalize(f_normal);
    vec3 L = normalize(f_light_direction);
    vec3 albedo = f_base_colour.xyz;

    vec3 ambient = albedo * global_ambient;

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = NdotL * albedo;

    vec3 V = normalize(-f_position);	
    vec3 R = reflect(-L, N);
    vec3 specular = pow(max(dot(R, V), 0.0), shininess) * specular_albedo;

    vec3 emissive = vec3(0.0);
    if (emitmode == 1) emissive = vec3(1.0, 1.0, 0.8);

    vec3 final = ambient + diffuse + specular + emissive;

    float attenuation = 1.0;

    if (attenuationmode == 1)
    {
        float attenuation_k1 = 0.5;
        float attenuation_k2 = 0.5;
        float attenuation_k3 = 0.5;
        attenuation = 1.0 / (attenuation_k1 + attenuation_k2 * length(f_light_direction) + 
                             attenuation_k3 * pow(length(f_light_direction), 2));
    }

    outputColor = vec4(final * attenuation, f_base_colour.a);
}