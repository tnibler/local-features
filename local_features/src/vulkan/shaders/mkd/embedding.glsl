#include "../extensions.glsl"
#include "../common.glsl"

#if defined(EMBEDDING_POLAR) && !defined(EMBEDDING_CARTESIAN)
const uint DIMS_EMB = DIMS_EMB_POLAR;
const uint DIMS_EMB_STRIDE = STRIDE_EMB_POLAR;
#elif !defined(EMBEDDING_POLAR) && defined(EMBEDDING_CARTESIAN)
const uint DIMS_EMB = DIMS_EMB_CARTESIAN;
const uint DIMS_EMB_STRIDE = STRIDE_EMB_CARTESIAN;
#else 
#error "Must define EMBEDDING_POLAR xor EMBEDDING_CARTESIAN"
#endif

// For every input dimension (0..7): 1 workgroup processes 1 whole patch, all embedding dimensions.
// Also tried, but slower: 
//  - split input patch over 2, 4 WGs. 
//  - one WG handles 2 input dims, whole patch, all embedding dims

const uint WG_SIZE_X = 32;
const uint WG_SIZE_Y = 2;
layout(local_size_x = WG_SIZE_X, local_size_y = WG_SIZE_Y, local_size_z = 1) in;

layout(push_constant) uniform EmbedPc {
    KeypointIndices keypoints;
    ConstantData consts;
    PatchBuffer patch_gradients;
    EmbeddingBuffer embeddings;
};

const uint ACC_SIZE = (WG_SIZE_X * WG_SIZE_Y) / MIN_SUBGROUP_SIZE;
shared float sh_acc[ACC_SIZE];
shared float sh_result[DIMS_EMB];

const float VM_FOURIER_N3_K8[] = {
        0.37872374,
        0.51796234,
        0.46882015,
        0.39798096
    };

// i_th component of von misen vector
float von_mises_n3k8(uint i, float ori, float mag) {
    const uint n = 3; // DIMS_IN = 2 * n + 1
    if (i == 0) {
        return VM_FOURIER_N3_K8[0] * mag;
    } else if (i < n + 1) {
        return cos(float(i) * ori) * VM_FOURIER_N3_K8[i] * mag;
    } else {
        return sin((float(i) - float(n)) * ori) * VM_FOURIER_N3_K8[i - n] * mag;
    }
}

void main() {
    const uint patch_index = gl_WorkGroupID.x;
    const uint in_dim = gl_WorkGroupID.y;

    if (patch_index >= min(MAX_KEYPOINTS, keypoints.n_keypoints)) {
        return;
    }

    float emb_dim_results[DIMS_EMB];
    for (uint i = 0; i < DIMS_EMB; i++) {
        emb_dim_results[i] = 0;
    }
    for (uint patch_row = 0; patch_row < PATCH_SIZE; patch_row += WG_SIZE_Y) {
        const uint px_y = patch_row + gl_LocalInvocationID.y;
        const uint px_x = gl_LocalInvocationID.x;

        // square root already included
        const float embedded_mag = patch_gradients.patches[patch_index].data[px_y * PATCH_SIZE + px_x];
        const float ori = patch_gradients.patches[MAX_KEYPOINTS + patch_index].data[px_y * PATCH_SIZE + px_x] ;
#if defined(EMBEDDING_POLAR)
        const float embedded_ori = ori + consts.gradient_angle[px_y * PATCH_SIZE + px_x];
#elif defined(EMBEDDING_CARTESIAN)
        const float embedded_ori = ori;
#endif
        const float embedded_grad = von_mises_n3k8(in_dim, embedded_ori, embedded_mag);

        for (uint emb_dim = 0; emb_dim < DIMS_EMB; emb_dim++) {
            const uint px_emb_idx = emb_dim * PATCH_SIZE * PATCH_SIZE + px_y * PATCH_SIZE + px_x;
#if defined(EMBEDDING_POLAR)
            const float emb = consts.embedding_polar[px_emb_idx];
#elif defined(EMBEDDING_CARTESIAN)
            const float emb = consts.embedding_cartesian[px_emb_idx];
#endif
            emb_dim_results[emb_dim] += embedded_grad * emb;
        }
    }

    for (uint emb_dim = 0; emb_dim < DIMS_EMB; emb_dim++) {
        float value = emb_dim_results[emb_dim];
        const float sg_sum = subgroupAdd(value);
        if (subgroupElect()) {
            sh_acc[gl_SubgroupID] = sg_sum;
        }
        barrier();
        if (gl_SubgroupID == 0) {
            float sg_sum = 0;
            for (uint sg = 0; sg < gl_NumSubgroups; sg += gl_SubgroupSize) {
                const uint idx = sg + gl_SubgroupInvocationID;
                if (idx < gl_NumSubgroups) { 
                    sg_sum += sh_acc[idx];
                }
            }
            float wg_sum = subgroupAdd(sg_sum);
            if (subgroupElect()) {
                sh_result[emb_dim] = wg_sum;
            }
        }
        barrier();
    }
    const uint emb_dim = gl_LocalInvocationID.x;
    if (emb_dim < DIMS_EMB) {
#if defined(EMBEDDING_POLAR)
        const uint offset = embedding_offset_polar(patch_index, in_dim, emb_dim);
#elif defined(EMBEDDING_CARTESIAN)
        const uint offset = embedding_offset_cartesian(patch_index, in_dim, emb_dim);
#endif
        embeddings.data[offset] = sh_result[emb_dim];
    }
}
