#version 460
#include "extensions.glsl"
#include "common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform BlurPyramidPc {
    SamplerId in_sampler_id;
    SampledImageId in_texture_id;
    StorageImageId out_image_id;
    uint in_level;
    bool vertical_pass;
    uint width;
    uint height;
};

const float WEIGHTS[2] = {0.375, 0.3125};
const float OFFSETS[2] = {0.0, 0.2};

float sample_in(float x, float y, uint level) {
    return textureLod(vko_sampler2D(in_texture_id, in_sampler_id), vec2(x, y), level).r;
}

void main() {
    const uint in_width = width;
    const uint in_height = height;

    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    const uint out_width = vertical_pass ? (in_width / 2) : in_width;
    const uint out_height = vertical_pass ? (in_height / 2) : in_height;

    if (x >= out_width || y >= out_height) {
        return;
    }

    if (vertical_pass) {
        const uint in_texture_width = textureSize(vko_texture2D(in_texture_id), 0).x;
        const uint in_texture_height = textureSize(vko_texture2D(in_texture_id), 0).y;
        const float x_in = (2 * float(x) + 0.5) / float(in_texture_width);
        const float y_in = (2 * float(y) + 0.5) / float(in_texture_height);
        const float offset_y = (1.0 + OFFSETS[1]) / float(in_texture_height);
        const float sum = 
            sample_in(x_in, y_in, 0) * WEIGHTS[0] 
            + (sample_in(x_in, y_in - offset_y, 0) + sample_in(x_in, y_in + offset_y, 0)) * WEIGHTS[1];

        // FIXME: use f16 if f16?
        imageStore(vko_image2D_r32f(out_image_id), ivec2(x, y), vec4_FORMAT(sum, 0.0, 0.0, 1.0));
    } else {
        const uint in_texture_width = textureSize(vko_texture2D(in_texture_id), int(in_level)).x;
        const uint in_texture_height = textureSize(vko_texture2D(in_texture_id), int(in_level)).y;
        const float x_in = (float(x) + 0.5) / float(in_texture_width);
        const float y_in = (float(y) + 0.5) / float(in_texture_height);
        const float offset_x = (1.0 + OFFSETS[1]) / float(in_texture_width);
        const float sum = 
            sample_in(x_in, y_in, in_level) * WEIGHTS[0]
            + (sample_in(x_in - offset_x, y_in, in_level) + sample_in(x_in + offset_x, y_in, in_level)) * WEIGHTS[1];

        imageStore(vko_image2D_r32f(out_image_id), ivec2(x, y), vec4_FORMAT(sum, 0.0, 0.0, 1.0));
    }
} 
