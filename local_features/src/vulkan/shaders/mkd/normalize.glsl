#version 460
#include "../extensions.glsl"
#include "../common.glsl"

const uint WG_SIZE = 64;
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform EmbeddingSumPc {
    KeypointIndices keypoints;
    RawDescriptorBuffer raw_descriptors;
    EmbeddingBuffer embeddings;
};

shared float sh_vector[DESCRIPTOR_SIZE];

const uint NORM_ACC_SIZE = WG_SIZE / MIN_SUBGROUP_SIZE;
shared float sh_norm_polar_sq[NORM_ACC_SIZE];
shared float sh_norm_cart_sq[NORM_ACC_SIZE];
shared float sh_norm_polar;
shared float sh_norm_cart;

void main() {
    const uint patch_index = gl_WorkGroupID.x;
    if (patch_index >= min(MAX_KEYPOINTS, keypoints.n_keypoints)) {
        return;
    }

    // polar embedding components
    for (uint in_dim = 0; in_dim < DIMS_INPUT; in_dim++) {
        if (gl_LocalInvocationID.x < DIMS_EMB_POLAR) {
            const uint offset = embedding_offset_polar(patch_index, in_dim, gl_LocalInvocationID.x);
            sh_vector[in_dim * DIMS_EMB_POLAR + gl_LocalInvocationID.x] = embeddings.data[offset];
        }
    }
    // cartesian embedding components
    for (uint in_dim = 0; in_dim < DIMS_INPUT; in_dim++) {
        if (gl_LocalInvocationID.x < DIMS_EMB_CARTESIAN) {
            const uint offset = embedding_offset_cartesian(patch_index, in_dim, gl_LocalInvocationID.x);
            sh_vector[DIMS_INPUT * DIMS_EMB_POLAR + in_dim * DIMS_EMB_CARTESIAN + gl_LocalInvocationID.x] = embeddings.data[offset];
        }
    }
    for (uint sg = 0; sg < gl_NumSubgroups; sg += WG_SIZE) {
        const uint idx = sg + gl_LocalInvocationID.x;
        if (idx < gl_NumSubgroups) {
            sh_norm_polar_sq[idx] = 0;
            sh_norm_cart_sq[idx] = 0;
        }
    }
    barrier();

    // L2-normalize polar and cartesian embeddings separately

    {
        float norm_polar_sq = 0;
        float norm_cart_sq = 0;
        for (uint base = 0; base < DESCRIPTOR_SIZE; base += WG_SIZE) {
            const uint idx = base + gl_LocalInvocationID.x;
            if (idx < DESCRIPTOR_SIZE) {
                const float value = sh_vector[idx];
                const float value_sq = value * value;
                // A bunch of warp level divergence could be avoided by zero-padding partial embedding sums
                // to multiples of workgroup sizes. 
                // Maybe faster? Very noisy to benchmark, this shader has no measurable impact
                if (idx < DIMS_EMB_POLAR * DIMS_INPUT) {
                    norm_polar_sq += value_sq;
                } else {
                    norm_cart_sq += value_sq;
                }
            }
        }
        const float sg_sum_polar = subgroupAdd(norm_polar_sq);
        const float sg_sum_cart = subgroupAdd(norm_cart_sq);
        if (subgroupElect()) {
            sh_norm_polar_sq[gl_SubgroupID] = sg_sum_polar;
            sh_norm_cart_sq[gl_SubgroupID] = sg_sum_cart;
        }
        barrier();

        if (gl_SubgroupID == 0) {
            float sum_sq_polar = 0;
            float sum_sq_cart = 0;
            for (uint sg = 0; sg < gl_NumSubgroups; sg += gl_SubgroupSize) {
                const uint idx = sg + gl_SubgroupInvocationID;
                if (idx < gl_NumSubgroups) {
                    sum_sq_polar += sh_norm_polar_sq[idx];
                    sum_sq_cart += sh_norm_cart_sq[idx];
                    sh_norm_polar_sq[idx] = 0;
                }
            }
            if (gl_SubgroupID == 0) {
                const float norm_polar = sqrt(subgroupAdd(sum_sq_polar));
                const float norm_cart = sqrt(subgroupAdd(sum_sq_cart));
                if (subgroupElect()) {
                    sh_norm_polar = norm_polar;
                    sh_norm_cart = norm_cart;
                }
            }
        }
        barrier();
    }

    float norm_sq = 0;
    for (uint i = 0; i < DESCRIPTOR_SIZE; i += WG_SIZE) {
        const uint ii = i + gl_LocalInvocationID.x;
        if (ii < DESCRIPTOR_SIZE) {
            const float norm = ii < DIMS_EMB_POLAR * DIMS_INPUT ? sh_norm_polar : sh_norm_cart;
            const float normalized = sh_vector[ii] / norm;
            sh_vector[ii] = normalized;
            norm_sq += normalized * normalized;
        }
    }

    // L2-Normalize again over entire sh_vector
    const float sum_norm_sq = subgroupAdd(norm_sq);
    if (subgroupElect()) {
        sh_norm_polar_sq[gl_SubgroupID] = sum_norm_sq;
    }
    barrier();

    if (gl_SubgroupID == 0) {
        float sum_sq = 0;
        for (uint sg = 0; sg < gl_NumSubgroups; sg += gl_SubgroupSize) {
            const uint idx = sg + gl_SubgroupInvocationID;
            if (idx < gl_NumSubgroups) {
                sum_sq += sh_norm_polar_sq[idx];
            }
        }
        const float norm = sqrt(subgroupAdd(sum_sq));
        if (subgroupElect()) {
            sh_norm_polar = norm;
        }
    }
    barrier();

    for (uint i = 0; i < DESCRIPTOR_SIZE; i += WG_SIZE) {
        const uint ii = i + gl_LocalInvocationID.x;
        if (ii < DESCRIPTOR_SIZE) {
            const float value = sh_vector[ii];
            raw_descriptors.patches[patch_index].data[ii] = value / sh_norm_polar;
        }
    }
}
