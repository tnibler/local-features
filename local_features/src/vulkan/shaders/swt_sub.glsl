#version 460
#include "extensions.glsl"
#include "common.glsl"

layout(local_size_x = 32) in;

layout(push_constant) uniform SwtSubPc {
    SamplerId coarse_sampler_id;
    SampledImageId coarse_texture_id;
    PixelBuffer fine;

    uint width;
    uint height;
};

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint level = gl_GlobalInvocationID.z;

    if (x >= width || y >= height) {
        return;
    }

    const float xx = (float(x) + 0.5) / textureSize(vko_texture2DArray(coarse_texture_id), 0).x; 
    const float yy = (float(y) + 0.5) / textureSize(vko_texture2DArray(coarse_texture_id), 0).y; 
    const float prev = texture(vko_sampler2DArray(coarse_texture_id, coarse_sampler_id), vec3(xx, yy, level)).r;
    const float next = texture(vko_sampler2DArray(coarse_texture_id, coarse_sampler_id), vec3(xx, yy, level + 1)).r;

    fine.data[level * width * height + y * width + x] = PX_TYPE(prev - next);
}
