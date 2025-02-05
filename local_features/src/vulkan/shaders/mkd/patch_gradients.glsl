#version 460
#include "../extensions.glsl"
#include "../common.glsl"
#include "../atan2.glsl"

layout(local_size_x = 32, local_size_y = 32) in;

layout(push_constant) uniform PatchGradientsPc {
    ExtremumLocations extremum_locations;
    KeypointIndices keypoints;
    PatchBuffer patch_gradients;
    SampledImageId pyramid_texture_id;
    SamplerId pyramid_sampler_id;
    uint image_width;
    uint image_height;
    float patch_scale_factor;

};

const uint KERNEL_RAD = 2;
const uint KERNEL_SIZE = 2 * KERNEL_RAD + 1;
const float gauss_kernel[KERNEL_SIZE] = {
        0.0096,
        0.2054,
        0.5699,
        0.2054,
        0.0096
    };

shared float sh_patch[PATCH_SIZE * PATCH_SIZE];

shared float sh_blur_tmp[PATCH_SIZE * PATCH_SIZE];

void main() {
    const uint patch_index = gl_WorkGroupID.x;
    if (patch_index >= min(MAX_KEYPOINTS, keypoints.n_keypoints)) {
        return;
    }
    const uint local_x = gl_LocalInvocationID.x;
    const uint local_y = gl_LocalInvocationID.y;

    const uint kp_loc_idx = keypoints.data[patch_index];
    const float scale = get_extremum_scale_float(extremum_locations, kp_loc_idx) * patch_scale_factor / float(PATCH_SIZE);
    // const float scale = 1;

    const float log2_scale = log2(scale);
    // index of image in pyramid
    const float kp_scale_level = floor(log2_scale);
    // scale factor within correct pyramid image
    const float rem_scale = pow(2, log2_scale - kp_scale_level);

    const float patch_x = get_extremum_x_float(extremum_locations, kp_loc_idx);
    const float patch_y = get_extremum_y_float(extremum_locations, kp_loc_idx);
    const float patch_angle = radians(uintBitsToFloat(keypoints.data[MAX_KEYPOINTS + patch_index]));

    {
        const ivec2 img_size = textureSize(vko_texture2D(pyramid_texture_id), int(kp_scale_level));

        const float ps2 = float(PATCH_SIZE) / 2.0;
        const float xx = (float(local_x) - ps2) * cos(patch_angle) - (float(local_y) - ps2) * sin(patch_angle);
        const float yy = (float(local_x) - ps2) * sin(patch_angle) + (float(local_y) - ps2) * cos(patch_angle);
        const float sample_x = xx * rem_scale + patch_x / pow(2., kp_scale_level);
        const float sample_y = yy * rem_scale + patch_y / pow(2., kp_scale_level);
        sh_patch[local_y * PATCH_SIZE + local_x] = textureLod(
            vko_sampler2D(pyramid_texture_id, pyramid_sampler_id),
            vec2((sample_x + 0.5) / float(img_size.x), (sample_y + 0.5) / float(img_size.y)),
            kp_scale_level
        ).r;
        barrier();
    }

    {
        // vertical blur pass
        float sum = 0;
        for (int i = 0; i < KERNEL_SIZE; i++) {
            const int yy = clamp(int(local_y) + i - int(KERNEL_RAD), 0, int(PATCH_SIZE) - 1);
            sum += gauss_kernel[i] * sh_patch[yy * PATCH_SIZE + local_x];
        }
        sh_blur_tmp[local_y * PATCH_SIZE + local_x] = sum;
        barrier();
    }

    {
        // horizontal blur pass
        float sum = 0;
        for (int i = 0; i < KERNEL_SIZE; i++) {
            const int xx = clamp(int(local_x) + i - int(KERNEL_RAD), 0, int(PATCH_SIZE) - 1);
            sum += gauss_kernel[i] * sh_blur_tmp[local_y * PATCH_SIZE + xx];
        }
        sh_patch[local_y * PATCH_SIZE + local_x] = sum;
        barrier();
    }

    const float grad_x =
        sh_patch[local_y * PATCH_SIZE + clamp(local_x, 1, PATCH_SIZE - 1) - 1] - sh_patch[local_y * PATCH_SIZE + clamp(local_x, 0, PATCH_SIZE - 2) + 1];
    const float grad_y = sh_patch[(clamp(local_y, 0, PATCH_SIZE - 2) + 1) * PATCH_SIZE + local_x] - sh_patch[(clamp(local_y, 1, PATCH_SIZE - 1) - 1) * PATCH_SIZE + local_x];
    
    const float EPS = 1e-8;
    // sqrt pulled out from embed_gradients
    const float mag = sqrt(sqrt(grad_x * grad_x + grad_y * grad_y + EPS));
    const float angle = -atan2(grad_x, grad_y);

    patch_gradients.patches[patch_index].data[local_y * PATCH_SIZE + local_x] = mag;
    patch_gradients.patches[MAX_KEYPOINTS + patch_index].data[local_y * PATCH_SIZE + local_x] = angle;
}
