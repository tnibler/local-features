#version 460
#include "../extensions.glsl"
#include "../common.glsl"

const uint WG_SIZE = 64;
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform L2NormalizePc {
    KeypointIndices keypoints;
    DescriptorBuffer descriptors;
};

const uint ACC_SIZE = WG_SIZE / MIN_SUBGROUP_SIZE;
shared float sh_norm_sq[ACC_SIZE];
shared float sh_norm;

void main() {
    const uint idx = gl_WorkGroupID.x;
    if (idx >= min(MAX_KEYPOINTS, keypoints.n_keypoints)) {
        return;
    }
    if (gl_LocalInvocationID.x < gl_NumSubgroups) {
        sh_norm_sq[gl_LocalInvocationID.x] = 0;
    }
    barrier();

    const uint offset = idx * PCAD_DESCRIPTOR_SIZE;
    float val_sq = 0;
    for (uint col = 0; col < PCAD_DESCRIPTOR_SIZE; col += WG_SIZE) {
        const float val = descriptors.patches[idx].data[col + gl_LocalInvocationID.x];
        val_sq += val * val;
    }
    const float sg_sum = subgroupAdd(val_sq);
    if (subgroupElect()) {
        sh_norm_sq[gl_SubgroupID] = sg_sum;
    }
    barrier();

    if (gl_SubgroupID == 0) {
        float norm_sq = 0;
        for (uint sg = 0; sg < gl_NumSubgroups; sg += gl_SubgroupSize) {
            const uint idx = sg + gl_LocalInvocationID.x;
            if (idx < gl_NumSubgroups) {
                norm_sq += sh_norm_sq[idx];
            }
        }
        subgroupBarrier();
        const float norm = sqrt(subgroupAdd(norm_sq));
        if (subgroupElect()) {
            sh_norm = norm;
        }
    }

    barrier();
    // TODO: check if reading vector to shmem once only is faster than 2 buffer accesses like here 
    for (uint col = 0; col < PCAD_DESCRIPTOR_SIZE; col += WG_SIZE) {
        descriptors.patches[idx].data[col + gl_LocalInvocationID.x] /= sh_norm;
    }
}
