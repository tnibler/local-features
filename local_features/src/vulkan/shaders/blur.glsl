#version 460
#include "extensions.glsl"
#include "pixel_type.glsl"
#include "common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform BlurPc {
    SamplerId coarse_sampler_id;
    SampledImageId coarse_texture_id;
    StorageImageId coarse_image_id;

    bool vertical_pass;
    uint width;
    uint height;
};

// sigma=0.6, kernel = { 0.00256627, 0.16552457, 0.66381833, 0.16552457, 0.00256627 }
const uint HALF_KERNEL_SIZE = 2;
const float KERNEL_WEIGHTS[HALF_KERNEL_SIZE] = {
    0.66381836,
    0.16809084,
};

const float KERNEL_OFFSETS[HALF_KERNEL_SIZE] = {
    0.0,
    0.015267163,
};

float sampleIn(vec3 coords) {
    return texture(vko_sampler2DArray(coarse_texture_id, coarse_sampler_id), coords).r;
}

void main() {
    const uint texture_width = textureSize(vko_texture2DArray(coarse_texture_id), 0).x;
    const uint texture_height = textureSize(vko_texture2DArray(coarse_texture_id), 0).y;

    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    if (x >= width || y >= height) {
        return;
    }

    const vec3 coords_in = vec3(
        (float(x) + 0.5) / float(texture_width), 
        (float(y) + 0.5) / float(texture_height), 
        vertical_pass ? 1 : 0);

    float sum = sampleIn(coords_in) * KERNEL_WEIGHTS[0];
    if (vertical_pass) {
        for(int i = 1, j = 1; i < HALF_KERNEL_SIZE; i++, j += 2) {
            const vec3 offset = vec3(0, (j + KERNEL_OFFSETS[i]) / float(texture_height), 0);
            sum += (sampleIn(coords_in - offset) + sampleIn(coords_in + offset)) * KERNEL_WEIGHTS[i];
        }
        imageStore(vko_image2DArray_FORMAT(coarse_image_id), ivec3(x, y, 0), vec4_FORMAT(sum, 0.0, 0.0, 1.0));
    } else {
        for(int i = 1, j = 1; i < HALF_KERNEL_SIZE; i++, j += 2) {
            const vec3 offset = vec3((j + KERNEL_OFFSETS[i]) / float(texture_width), 0, 0);
            sum += (sampleIn(coords_in - offset) + sampleIn(coords_in + offset)) * KERNEL_WEIGHTS[i];
        }
        // use layer 1 as temporary horizontal out
        imageStore(vko_image2DArray_FORMAT(coarse_image_id), ivec3(x, y, 1), vec4_FORMAT(sum, 0.0, 0.0, 1.0));
    }
} 
