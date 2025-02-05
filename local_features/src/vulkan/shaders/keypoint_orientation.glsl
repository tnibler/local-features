#version 460
#include "extensions.glsl"
#include "common.glsl"
#include "atan2.glsl"

const uint WG_SIZE = 16;
layout(local_size_x = WG_SIZE, local_size_y = WG_SIZE) in;

const uint N_BINS = 36;
const float BIN_ANGLE_STEP = float(N_BINS) / (2 * PI);

layout(push_constant) uniform KeypointOrientationPc {
    ExtremumLocations extremum_locations;
    FilteredExtrema filtered_extrema;
    KeypointIndices keypoints;
    StorageImageId coarse_image_id;

    uint width;
    uint height;
    uint rt_max_keypoints;
};

const int MAX_ORI_PATCH_RADIUS = 7;
const uint ORI_PATCH_SIZE = 2 * MAX_ORI_PATCH_RADIUS + 1;
shared float sh_patch[ORI_PATCH_SIZE * ORI_PATCH_SIZE];
shared uint sh_bins[ORI_PATCH_SIZE * ORI_PATCH_SIZE];
shared float sh_hist[N_BINS];
shared float sh_rawhist[N_BINS + 4];
shared float sh_hist_localmax_thresh;

float imageLoadCoarse(ivec3 coords) {
    // FIXME: use f16 if f16
    return imageLoad(vko_image2DArray_r32f(coarse_image_id), coords).r;
}

void main() {
    // number of workgroups = number of filtered extrema, so WorkGroupID.x is a valid index
    const uint extremum_idx = filtered_extrema.indices[gl_WorkGroupID.x];

    const int local_x = int(gl_LocalInvocationID.x);
    const int local_y = int(gl_LocalInvocationID.y);

    const float kp_x = get_extremum_x_float(extremum_locations, extremum_idx);
    const float kp_y = get_extremum_y_float(extremum_locations, extremum_idx);
    const float kp_size = get_extremum_scale_float(extremum_locations, extremum_idx);
    const int kp_xi = int(kp_x);
    const int kp_yi = int(kp_y);


    // TODO: make sure the polynomial fix scale thing doesnt mess up scale -> scale level calculation
    const uint kp_scale_level = uint(round(log2(kp_size / (DOG_FIRST_SCALE_SIGMA * DOG_SIGMA_RADIUS_FACTOR))));

    const int step = 1 << kp_scale_level; // FIXME: 2^log2 kindof redundant

    const int radius = int(round(3 * 1.5 * kp_size / DOG_SIGMA_RADIUS_FACTOR));
    const float sigma = 1.5 * kp_size / DOG_SIGMA_RADIUS_FACTOR;

    const uint local_id = gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (local_id < N_BINS) {
        sh_hist[local_id] = 0;
    }
    if (local_id < N_BINS + 4) {
        sh_rawhist[local_id] = 0;
    }

    bool in_load_radius = false;
    bool in_grad_radius = false;
    const int x_patch = int(local_x) - MAX_ORI_PATCH_RADIUS;
    const int y_patch = int(local_y) - MAX_ORI_PATCH_RADIUS;
    const int x_patch_dilated = x_patch * step;
    const int y_patch_dilated = y_patch * step;
    const int x_img = kp_xi + x_patch_dilated;
    const int y_img = kp_yi + y_patch_dilated;
    if (local_x < ORI_PATCH_SIZE && local_y < ORI_PATCH_SIZE) {
        const bool valid_px = 0 <= x_img && x_img < width && 0 <= y_img && y_img <= height ;
        in_load_radius = valid_px && abs(x_patch_dilated) <= radius + step && abs(y_patch_dilated) <= radius + step;
        in_grad_radius = valid_px && abs(x_patch_dilated) <= radius && abs(y_patch_dilated) <= radius;

        if (valid_px /* && in_load_radius */) {
            sh_patch[local_y * ORI_PATCH_SIZE + local_x] = imageLoadCoarse(ivec3(x_img, y_img, kp_scale_level));
        } else {
            sh_patch[local_y * ORI_PATCH_SIZE + local_x] = 0;
        }
    }    
    barrier();
    memoryBarrierShared();

    const float grad_x = in_grad_radius ? (sh_patch[local_y * ORI_PATCH_SIZE + local_x + 1] - sh_patch[local_y * ORI_PATCH_SIZE + local_x - 1]) : 0.0;
    const float grad_y = in_grad_radius ? (sh_patch[(local_y - 1) * ORI_PATCH_SIZE + local_x] - sh_patch[(local_y + 1) * ORI_PATCH_SIZE + local_x]) : 0.0;
    // every thread must have read the gradients before we're allowed to write to sh_patch again
    memoryBarrierShared();
    barrier();

    if (in_grad_radius && (grad_x != 0.0 || grad_y != 0.0)) {
        const float grad_mag = sqrt(grad_x*grad_x + grad_y*grad_y);
        const float distance = float(x_patch) * float(step) * float(x_patch) * float(step) + float(y_patch) * float(step) * float(y_patch) * float(step);
        const float hist_weight = exp(-distance / (2.0 * sigma * sigma));

        // now patch contains weights
        sh_patch[local_y * ORI_PATCH_SIZE + local_x] = hist_weight * grad_mag;

        const float angle = atan2(grad_x, grad_y);

        const int raw_bin = int(round(angle * BIN_ANGLE_STEP));
        const uint bin = raw_bin < 0 ? raw_bin + N_BINS : (raw_bin >= N_BINS ? raw_bin - N_BINS : raw_bin);
        sh_bins[local_y * ORI_PATCH_SIZE + local_x] = bin;
    } else if (local_y < ORI_PATCH_SIZE && local_x < ORI_PATCH_SIZE) {
        // outside area considered for histogram, set to invalid bin number
        sh_bins[local_y * ORI_PATCH_SIZE + local_x] = N_BINS + 1;
    }
    barrier();
    memoryBarrierShared();

    if (local_id == 0) {
        for (uint y = 0; y < ORI_PATCH_SIZE; y++) {
            for (uint x = 0; x < ORI_PATCH_SIZE; x++) {
                const uint patchloc = y * ORI_PATCH_SIZE + x;
                const uint maybe_bin = sh_bins[patchloc];
                if (maybe_bin < N_BINS) {
                    sh_rawhist[2 + maybe_bin] += sh_patch[patchloc];
                }
            }
        }
        sh_rawhist[1] = sh_rawhist[N_BINS + 1];
        sh_rawhist[0] = sh_rawhist[N_BINS];
        sh_rawhist[N_BINS + 2] = sh_rawhist[2];
        sh_rawhist[N_BINS + 3] = sh_rawhist[3];
    }
    barrier();
    memoryBarrierShared();

    if (local_id < N_BINS) {
        const uint rawbin = local_id + 2;
        sh_hist[local_id] = (sh_rawhist[rawbin - 2] + sh_rawhist[rawbin + 2]) * 1.0/16.0 
            + (sh_rawhist[rawbin - 1] + sh_rawhist[rawbin + 1]) * 4.0/16.0 
            + sh_rawhist[rawbin] * 6.0/16.0;
    }
    barrier();
    memoryBarrierShared();
    if (local_id == 0) {
        float current_max = 0;
        for (uint i = 0; i < N_BINS; i++) {
            current_max = max(current_max, sh_hist[i]);
        }
        sh_hist_localmax_thresh = current_max * 0.8;
    }
    barrier();
    memoryBarrierShared();


    if (local_id < N_BINS) {
        const float histval = sh_hist[local_id];
        const uint left_idx = local_id > 0 ? (local_id - 1) : (N_BINS - 1);
        const uint right_idx = (local_id + 1) % N_BINS;
        const float left = sh_hist[left_idx];
        const float right = sh_hist[right_idx];
        const bool is_localmax = left < histval && right < histval && sh_hist_localmax_thresh <= histval;
        if (is_localmax) {
            const float interp = (left - right) / (left - 2 * histval + right);
            const float raw_bin = float(local_id) + interp / 2.0;
            const float bin = raw_bin < 0 ? (raw_bin + N_BINS) : (raw_bin > N_BINS ? (raw_bin - N_BINS) : raw_bin);
            const float angle = 360.0 - (360.0 / float(N_BINS)) * bin;
            const uint next_kpidx = atomicAdd(keypoints.n_keypoints, 1);
            if (next_kpidx < rt_max_keypoints) {
                keypoints.data[next_kpidx] = extremum_idx;
                // TODO: could use rt_max_keypoints as offset here to reduce copy/readback size
                keypoints.data[MAX_KEYPOINTS + next_kpidx] = floatBitsToUint(angle);
            }
        }
    }
}
