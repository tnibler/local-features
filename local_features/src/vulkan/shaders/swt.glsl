#version 460
#include "extensions.glsl"
#include "common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform SwtPc {
    SamplerId coarse_sampler_id;
    SampledImageId coarse_texture_id;
    StorageImageId coarse_image_id;
    uint in_level;
    bool vertical_pass;
    uint width;
    uint height;
    uint n_coarse_levels;
};

const float KERNEL[] = { 6.0/16.0, 4.0/16.0, 1.0/16.0 };

float sampleIn(vec3 coords) {
    return texture(vko_sampler2DArray(coarse_texture_id, coarse_sampler_id), coords).r;
}

void main() {
    const uint texture_width = textureSize(vko_texture2DArray(coarse_texture_id), 0).x;
    const uint texture_height = textureSize(vko_texture2DArray(coarse_texture_id), 0).y;

    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    if (gl_GlobalInvocationID.x >= width || gl_GlobalInvocationID.y >= height) {
        return;
    }

    const float x_in = (float(x) + 0.5) / float(texture_width);
    const float y_in = (float(y) + 0.5) / float(texture_height);
    if (vertical_pass) {
        const vec3 coords_in = vec3(x_in, y_in, n_coarse_levels - 1);
        const float dilate = float(1 << in_level) / float(texture_height);
        float sum = sampleIn(coords_in).r * KERNEL[0];
        sum += sampleIn(coords_in - vec3(0, dilate * 1, 0)).r * KERNEL[1];
        sum += sampleIn(coords_in - vec3(0, dilate * 2, 0)).r * KERNEL[2];

        sum += sampleIn(coords_in + vec3(0, dilate * 2, 0)).r * KERNEL[2];
        sum += sampleIn(coords_in + vec3(0, dilate * 1, 0)).r * KERNEL[1];
        imageStore(vko_image2DArray_FORMAT(coarse_image_id), ivec3(x, y, in_level + 1), vec4_FORMAT(sum, 0.0, 0.0, 1.0));
    } else {
        const vec3 coords_in = vec3(x_in, y_in, in_level);
        const float dilate = float(1 << in_level) / float(texture_width);
        float sum = sampleIn(coords_in).r * KERNEL[0];
        sum += sampleIn(coords_in - vec3(dilate * 2, 0, 0)).r * KERNEL[2];
        sum += sampleIn(coords_in - vec3(dilate * 1, 0, 0)).r * KERNEL[1];

        sum += sampleIn(coords_in + vec3(dilate * 1, 0, 0)).r * KERNEL[1];
        sum += sampleIn(coords_in + vec3(dilate * 2, 0, 0)).r * KERNEL[2];
        imageStore(vko_image2DArray_FORMAT(coarse_image_id), ivec3(x, y, n_coarse_levels - 1), vec4_FORMAT(sum, 0.0, 0.0, 1.0));
    }
}

