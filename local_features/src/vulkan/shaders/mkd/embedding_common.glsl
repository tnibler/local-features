#include "../common.gpp.glsl"

const float VM_FOURIER_N3_K8[] = {
        0.37872374,
        0.51796234,
        0.46882015,
        0.39798096
    };

// Maps [0..31] to [-1..1]
float normalized_pos(uint x) {
    return (float(x) - 15.5) / 15.5;
}

// x and y in [-1; 1]
float gaussian_weight(float x, float y) {
    const float MAX_SQ_NORM = 2.0;
    const float SIGMA = 1;

    const float sq_norm = (x * x + y * y) / MAX_SQ_NORM;
    // No factor 2 before sigma^2, intentional.
    return exp(-1.0 * sq_norm / (SIGMA * SIGMA));
}

float von_mises_n3k8_coeff(uint in_dim) {
    if (in_dim == 0) {
        return VM_FOURIER_N3_K8[0];
    } else if (in_dim < 3 + 1) {
        return VM_FOURIER_N3_K8[in_dim];
    } else {
        return VM_FOURIER_N3_K8[in_dim - 3];
    }
}

