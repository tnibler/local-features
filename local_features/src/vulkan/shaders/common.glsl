#ifndef _COMMON_GLSL
#define _COMMON_GLSL

#include "extensions.glsl"
#include <vulkano.glsl>
#include "pixel_type.glsl"

layout (constant_id = 0) const uint MIN_SUBGROUP_SIZE = 8;
layout (constant_id = 1) const uint MAX_EXTREMA = 1;
layout (constant_id = 2) const uint MAX_KEYPOINTS = 1;
layout (constant_id = 3) const uint EXTREMUM_BLOCK_LEN = 256;

// detected blob radius = sqrt(2) * sigma_LoG = sqrt(2) * ratio * sigma_DoG * sqrt(log(ratio) / (ratio^2 - 1)) $, where ratio = 2, sigma_DoG = 0.6 (first blur, before SWT)
// = 0.8157, but 0.82 works a bit better with all other constants in e.g., keypoint_orientation
const float DOG_FIRST_SCALE_SIGMA = 0.82;
// Blob radius = sqrt(2) * sigma_LoG
const float DOG_SIGMA_RADIUS_FACTOR = sqrt(2.0);

const uint PATCH_SIZE = 32;
const uint PCAD_DESCRIPTOR_SIZE = 128;

const uint DIMS_INPUT = 7;
const uint DIMS_EMB_CARTESIAN = 9;
const uint DIMS_EMB_POLAR = 25;
const uint MAX_DIMS_EMB = DIMS_EMB_POLAR;

// Each input dimension produces 9/25 output components.
// Maybe want to use 16/32 as stride if that's faster. 
// Not successfully measured any + or - impact yet.
const uint STRIDE_EMB_CARTESIAN = DIMS_EMB_CARTESIAN;
const uint STRIDE_EMB_POLAR = DIMS_EMB_POLAR;

const uint DESCRIPTOR_SIZE = DIMS_INPUT * (DIMS_EMB_CARTESIAN + DIMS_EMB_POLAR);
layout(buffer_reference, std430, buffer_reference_align=4) readonly buffer ConstantData {
    float gradient_angle[PATCH_SIZE * PATCH_SIZE];
    float embedding_polar[DIMS_EMB_POLAR * PATCH_SIZE * PATCH_SIZE];
    float embedding_cartesian[DIMS_EMB_CARTESIAN * PATCH_SIZE * PATCH_SIZE];
    float mean_vec[DESCRIPTOR_SIZE];
    float eigen_vecs[DESCRIPTOR_SIZE * PCAD_DESCRIPTOR_SIZE];
};

// == Scale space extremum detection ==

// NOTE: readback depends on field order here, careful when changing things
layout(buffer_reference, scalar, buffer_reference_align=4) buffer ExtremumLocations {
    // Counter incremented extremum scanning stage. Counts valid entries in data
    // detect: output, size of extremum_{scale,x,y,contrast}
    uint n_extrema;
    // simple attempt to keep atomic counters on their own cache line. Not verified
    uint _pad[15];

    // size: 4 * MAX_EXTREMA
    uint data[];
};

uint _coord_idx(uint extremum_idx, uint coord) {
    const uint N_COORDS = 4;
    return (extremum_idx / EXTREMUM_BLOCK_LEN) * N_COORDS * EXTREMUM_BLOCK_LEN + coord * EXTREMUM_BLOCK_LEN + extremum_idx % EXTREMUM_BLOCK_LEN;
}

uint get_extremum_x_uint(ExtremumLocations buf, uint i)  { return buf.data[_coord_idx(i, 0)]; }
void set_extremum_x_uint(ExtremumLocations buf, uint i, uint x) { buf.data[_coord_idx(i, 0)] = x; }
float get_extremum_x_float(ExtremumLocations buf, uint i) { return uintBitsToFloat(buf.data[_coord_idx(i, 0)]); }
void set_extremum_x_float(ExtremumLocations buf, uint i, float x) { buf.data[_coord_idx(i, 0)] = floatBitsToUint(x); }

uint get_extremum_y_uint(ExtremumLocations buf, uint i)  { return buf.data[_coord_idx(i, 1)]; }
void set_extremum_y_uint(ExtremumLocations buf, uint i, uint y) { buf.data[_coord_idx(i, 1)] = y; }
float get_extremum_y_float(ExtremumLocations buf, uint i) { return uintBitsToFloat(buf.data[_coord_idx(i, 1)]); }
void set_extremum_y_float(ExtremumLocations buf, uint i, float y) { buf.data[_coord_idx(i, 1)] = floatBitsToUint(y); }

// scale space layer written by scan_extrema
uint get_extremum_scale_uint(ExtremumLocations buf, uint i)  { return buf.data[_coord_idx(i, 2)]; }
void set_extremum_scale_uint(ExtremumLocations buf, uint i, uint s) { buf.data[_coord_idx(i, 2)] = s; }

// interpolated scale (float) written by refine_extrema
float get_extremum_scale_float(ExtremumLocations buf, uint i)  { return uintBitsToFloat(buf.data[_coord_idx(i, 2)]); }
void set_extremum_scale_float(ExtremumLocations buf, uint i, float s) { buf.data[_coord_idx(i, 2)] = floatBitsToUint(s); }

float get_extremum_contrast(ExtremumLocations buf, uint i)  { return uintBitsToFloat(buf.data[_coord_idx(i, 3)]); }
void set_extremum_contrast(ExtremumLocations buf, uint i, float c) { buf.data[_coord_idx(i, 3)] = floatBitsToUint(c); }

// Written by host
layout(buffer_reference, scalar, buffer_reference_align=4) readonly buffer FilteredExtrema {
    uint n_filtered_extrema;
    // Indices into extremum_locations
    // size: n_filtered_extrema
    uint indices[];
};

// == Descriptor extraction ==

// Written by keypoint_orientation stage
layout(buffer_reference, scalar, buffer_reference_align=4) buffer KeypointIndices {
    uint n_keypoints;
    uint _pad[15];

    // size: 2 * MAX_KEYPOINTS
    // 0..MAX_KEYPOINTS: Index into extremum_{scale,x,y}
    // MAX_KEYPOINTS..:  (float) keypoint_orientation
    uint data[];
};

struct Patch {
    float data[PATCH_SIZE * PATCH_SIZE];
};

// Written by patch_gradients stage
layout(buffer_reference, scalar, buffer_reference_align=4) buffer PatchBuffer {
    // size: 2 * MAX_KEYPOINTS
    // Order: Magnitude, Angle
    Patch patches[];
};

uint embedding_offset_polar(uint patch_idx, uint in_dim, uint emb_dim) {
    return patch_idx * DIMS_INPUT * STRIDE_EMB_POLAR +
            in_dim * STRIDE_EMB_POLAR + emb_dim;
}

uint embedding_offset_cartesian(uint patch_idx, uint in_dim, uint emb_dim) {
    const uint base = MAX_KEYPOINTS * DIMS_INPUT * STRIDE_EMB_POLAR;
    return base + patch_idx * DIMS_INPUT * STRIDE_EMB_CARTESIAN +
            in_dim * STRIDE_EMB_CARTESIAN + emb_dim;
}

// Written by embedding stage
layout(buffer_reference, scalar, buffer_reference_align=4) buffer EmbeddingBuffer {
    // Layout:
    // Polar: MAX_KEYPOINTS * DIMS_INPUT * STRIDE_EMB_POLAR
    // Cartesian: MAX_KEYPOINTS * DIMS_INPUT * STRIDE_EMB_CARTESIAN
    float data[];
};

struct RawDescriptor {
    float data[DESCRIPTOR_SIZE];
};

layout(buffer_reference, scalar, buffer_reference_align=4) buffer RawDescriptorBuffer {
    RawDescriptor[] patches;
};

struct Descriptor {
    float data[PCAD_DESCRIPTOR_SIZE];
};

layout(buffer_reference, scalar, buffer_reference_align=4) buffer DescriptorBuffer {
    Descriptor[] patches;
};
#endif // _COMMON_GLSL
