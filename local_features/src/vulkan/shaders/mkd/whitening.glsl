#version 460
#include "../extensions.glsl"
#include "../common.glsl"

const uint WG_SIZE = 64;
layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

const uint ROWS_PER_WG = 32;

layout(push_constant) uniform WhiteningPc {
    ConstantData consts;
    KeypointIndices keypoints;
    RawDescriptorBuffer raw_descriptors;
    DescriptorBuffer descriptors;
};

const uint ACC_SIZE = WG_SIZE / MIN_SUBGROUP_SIZE;
shared float acc[ACC_SIZE];

shared float vec[DESCRIPTOR_SIZE];

void main() {
    const uint kp_idx = gl_WorkGroupID.x;
    if (kp_idx >= min(MAX_KEYPOINTS, keypoints.n_keypoints)) {
        return;
    }
    const uint start_row = gl_WorkGroupID.y * ROWS_PER_WG;
    
    const uint in_offset = kp_idx * DESCRIPTOR_SIZE;
    for (uint row = 0; row < DESCRIPTOR_SIZE; row += WG_SIZE) {
        if (row + gl_LocalInvocationID.x < DESCRIPTOR_SIZE) {
            const uint idx = row + gl_LocalInvocationID.x;
            const float enc_val = raw_descriptors.patches[kp_idx].data[idx];
            const float mean_val = consts.mean_vec[idx];
            vec[idx] =  enc_val - mean_val;
        }
    }
    barrier();

    for (uint out_row = start_row; out_row < start_row + ROWS_PER_WG; out_row++) {
        if (gl_LocalInvocationID.x < gl_NumSubgroups) {
            acc[gl_LocalInvocationID.x] = 0;
        }
        barrier();

        for (uint col = 0; col < DESCRIPTOR_SIZE; col += WG_SIZE) {
            const uint idx = col + gl_LocalInvocationID.x;
            float val = 0;
            float matval = 0;
            if (idx < DESCRIPTOR_SIZE) {
                val = vec[idx];
                matval = consts.eigen_vecs[out_row * DESCRIPTOR_SIZE + idx];
            }
            const float p = val * matval;
            const float sg_sum = subgroupAdd(p);
            if (subgroupElect()) {
                acc[gl_SubgroupID] += sg_sum;
            }
            barrier();
        }
        if (gl_SubgroupID == 0) {
            float accval = 0;
            for (uint sg = 0; sg < gl_NumSubgroups; sg += gl_SubgroupSize) {
                const uint idx = sg + gl_SubgroupInvocationID;
                if (gl_SubgroupInvocationID < gl_NumSubgroups) {
                    accval += acc[idx];
                }
            }
            const float wg_sum = subgroupAdd(accval);
            // TODO: check if accumulating the repojected vector into shmem and writing all at once is faster
            if (subgroupElect()) {
                descriptors.patches[kp_idx].data[out_row] = wg_sum;
            }
        }
        barrier();
    }
}
