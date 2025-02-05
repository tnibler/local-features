#version 460
#include "extensions.glsl"
#include "common.glsl"

const uint CX = 4;
const uint CY = 4;
const uint CZ = 4;
const uint WG_LEN = CX * CY * CZ;

layout(local_size_x = CX, local_size_y = CY, local_size_z = CZ) in;

layout(push_constant) uniform ScanExtremaPc {
    PixelBuffer fine;
    ExtremumLocations extremum_locations;
    uint skip_layers;
    uint border;
    float contrast_threshold;

    uint width;
    uint height;
    uint max_extrema;

    uint n_fine_levels;
};

const uint CUBE_LEN = (CX + 2) * (CY + 2) * (CZ + 2);
shared float sh_cube[CUBE_LEN];

// How many local extrema i.e., samples that are larger/smaller than all their 26 neighbors.
// (Pretty sure) the exact value is ceil(CX/2)*ceil(CY/2)*ceil(CZ/2) + floor(CX/2)*floor(CY/2)*floor(CZ/2),
// generalized from the 1D case of a series like 1 0 1 0 1 0, where every value is a local min/max.
// That's a lot though so half that is probably ok (8 for a 4x4x4 cube).
const uint max_wg_extrema = ((CX / 2) * (CY/2) * (CZ/2) + ((CX+1) / 2) * (CY+1)/2 * (CZ+1)/2) / 2;

shared uint sh_n_extrema;

// TODO: take border into account
ivec3 wg_coords = (ivec3(gl_WorkGroupID) * ivec3(CX, CY, CZ) + ivec3(border, border, 1 + skip_layers));

float cube_at(ivec3 v) {
    return sh_cube[v.z * (CX + 2) * (CY + 2) + v.y * (CX + 2) + v.x];
}

float img_load(ivec3 v) {
    const ivec3 m = ivec3(width-1, height-1, n_fine_levels-1);
    if (v != clamp(v, ivec3(0, 0, 0), m)) {
        printf("%v3i, (%v3i)\n", v, m);
    }
    return fine.data[v.z * width * height + v.y * width + v.x];
}

float at(uint z, uint y, uint x) {
    const ivec3 coords = ivec3(x, y, z);
    // return img_load(coords - ivec3(0, 0, skip_layers));
    const ivec3 cube_coords = coords - wg_coords + ivec3(1, 1, 1);
    // if (cube_at(cube_coords) != y) {
    //     printf("y=%d, cube=%d, cube_coords=(%v3i), coords=(%v3i)\n", y, cube_at(cube_coords), cube_coords, coords);
    // }
    return cube_at(cube_coords);
}

const uint LOAD_LOOPS = (CUBE_LEN - 1) / WG_LEN + 1;
void load_shared_dog_volume() {
    const uint id = gl_LocalInvocationIndex;
    for (uint i = 0; i < LOAD_LOOPS; i++) {
        const int iid = int(id + i * WG_LEN);
        const int nz = int((CX + 2) * (CY + 2));
        const int ny = int((CX + 2));
        const int zz = iid / nz;
        const int yy = (iid - zz * nz) / ny;
        const int xx = iid - zz * nz - yy * ny;

        ivec3 load = ivec3(xx, yy, zz) + wg_coords - ivec3(1, 1, 1);
        if (iid < CUBE_LEN && load.x >= 0 && load.x <=  width - border && load.y >= 0 && load.y <= height - border && load.z >= 0 && load.z < n_fine_levels) {
            sh_cube[iid] = img_load(load);
        }
    }
}

bool is_extremum(const uint x, const uint y, const int z) {
    bool a = true;
    if (x < max(border, 1) || x >= width - max(border, 1) || y < max(border, 1) || y >= height - max(border, 1) || z <= 0 || z >= n_fine_levels - 1) {
        a = false;
    }
    const float val = at(z, y, x);
    if (abs(val) <= contrast_threshold) {
        a = false;
    }
    const float sgn = sign(val);
    if (a) {

    a =
        sgn * val >= sgn * at(z - 1, y - 1 , x - 1)
        && sgn * val >= sgn * at(z - 1, y - 1 , x - 0)
        && sgn * val >= sgn * at(z - 1, y - 1 , x + 1)
        && sgn * val >= sgn * at(z - 1, y + 0 , x - 1)
        && sgn * val >= sgn * at(z - 1, y + 0 , x + 0)
        && sgn * val >= sgn * at(z - 1, y + 0 , x + 1)
        && sgn * val >= sgn * at(z - 1, y + 1 , x - 1)
        && sgn * val >= sgn * at(z - 1, y + 1 , x + 0)
        && sgn * val >= sgn * at(z - 1, y + 1 , x + 1)
        && sgn * val >= sgn * at(z + 0, y - 1 , x - 1)
        && sgn * val >= sgn * at(z + 0, y - 1 , x - 0)
        && sgn * val >= sgn * at(z + 0, y - 1 , x + 1)
        && sgn * val >= sgn * at(z + 0, y + 0 , x - 1)
        && sgn * val >= sgn * at(z + 0, y + 0 , x + 1)
        && sgn * val >= sgn * at(z + 0, y + 1 , x - 1)
        && sgn * val >= sgn * at(z + 0, y + 1 , x + 0)
        && sgn * val >= sgn * at(z + 0, y + 1 , x + 1)
        && sgn * val >= sgn * at(z + 1, y - 1 , x - 1)
        && sgn * val >= sgn * at(z + 1, y - 1 , x - 0)
        && sgn * val >= sgn * at(z + 1, y - 1 , x + 1)
        && sgn * val >= sgn * at(z + 1, y + 0 , x - 1)
        && sgn * val >= sgn * at(z + 1, y + 0 , x + 0)
        && sgn * val >= sgn * at(z + 1, y + 0 , x + 1)
        && sgn * val >= sgn * at(z + 1, y + 1 , x - 1)
        && sgn * val >= sgn * at(z + 1, y + 1 , x + 0)
        && sgn * val >= sgn * at(z + 1, y + 1 , x + 1);
    }
    return a;
}

void main() {
    if (gl_LocalInvocationIndex == 0) {
        sh_n_extrema = 0;
    }
    load_shared_dog_volume();

    barrier();

    // if (gl_LocalInvocationIndex == 0 && gl_WorkGroupID== vec3(10, 2, 1)) {
    //     for (uint z = 0; z < CZ + 2; z++) {
    //         for (uint y = 0; y < CY + 2; y++) {
    //             for (uint x = 0; x < CX + 2; x++) {
    //                 printf("%f ", cube_at(ivec3(x, y, z)));
    //             }
    //             printf("\n");
    //         }
    //     }
    //     printf("\n\n");
    // }
    //
    const uint check_x = gl_GlobalInvocationID.x + border;
    const uint check_y = gl_GlobalInvocationID.y + border;
    const int check_z = int(gl_GlobalInvocationID.z + 1 + skip_layers);
    if (is_extremum(check_x, check_y, check_z)) {
        const uint our_extremum_index = atomicAdd(sh_n_extrema, 1);
        if (our_extremum_index < max_wg_extrema) {
            sh_cube[our_extremum_index] = uintBitsToFloat(check_x);
            sh_cube[max_wg_extrema + our_extremum_index] = uintBitsToFloat(check_y);
            sh_cube[2 * max_wg_extrema + our_extremum_index] = uintBitsToFloat(check_z);
        }
    }

    barrier();

    if (gl_LocalInvocationIndex >= min(sh_n_extrema, max_wg_extrema)) {
        return;
    }
    // printf("%v3i here %d\n", gl_WorkGroupID, gl_LocalInvocationIndex);

    const uint idx = gl_LocalInvocationIndex;
    uint x = floatBitsToUint(sh_cube[idx]);
    uint y = floatBitsToUint(sh_cube[max_wg_extrema + idx]);
    uint z = floatBitsToUint(sh_cube[2 * max_wg_extrema + idx]);

    const float dds = (at(z+1, y, x) - at(z-1, y, x)) / 2.0;
    const float ddy = (at(z, y+1, x) - at(z, y-1, x)) / 2.0;
    const float ddx = (at(z, y, x+1) - at(z, y, x-1)) / 2.0;

    const float value2x = at(z, y, x) * 2.0;
    // d2/ds2
    const float h11 = at(z+1, y, x) + at(z-1, y, x) - value2x;
    // d2/dy2
    const float h22 = at(z, y+1, x) + at(z, y-1, x) - value2x;
    // d2/dx2
    const float h33 = at(z, y, x+1) + at(z, y, x-1) - value2x;
    // d2/dyds
    const float h12 = (at(z+1, y+1, x) - at(z-1, y+1, x) - at(z+1, y-1, x) + at(z-1, y-1, x)) / 4.0;
    // d2/dxds
    const float h13 = (at(z+1, y, x+1) - at(z-1, y, x+1) - at(z+1, y, x-1) + at(z-1, y, x-1)) / 4.0;
    // d2/dxdy
    const float h23 = (at(z, y+1, x+1) - at(z, y+1, x-1) - at(z, y-1, x+1) + at(z, y-1, x-1)) / 4.0;

    const float det = h11 * h22 * h33 - h11 * h23 * h23 - h12 * h12 * h33 + 2. * h12 * h13 * h23
        - h13 * h13 * h22;
    const float hinv11 = (h22 * h33 - h23 * h23) / det;
    const float hinv12 = (h13 * h23 - h12 * h33) / det;
    const float hinv13 = (h12 * h23 - h13 * h22) / det;
    const float hinv22 = (h11 * h33 - h13 * h13) / det;
    const float hinv23 = (h12 * h13 - h11 * h23) / det;
    const float hinv33 = (h11 * h22 - h12 * h12) / det;

    const float offset_scale = -(hinv11 * dds + hinv12 * ddy + hinv13 * ddx);
    const float offset_y     = -(hinv12 * dds + hinv22 * ddy + hinv23 * ddx);
    const float offset_x     = -(hinv13 * dds + hinv23 * ddy + hinv33 * ddx);

    if (abs(offset_x) > 0.5 || abs(offset_y) > 0.5 || abs(offset_scale) > 0.5) {
        x = int(round(x + offset_x));
        y = int(round(y + offset_y));
        z = int(round(z + offset_scale));
    } else {

        if (x < border || width - border <= x || y < border || height - border <= y || z < 1 || n_fine_levels - 1 <= z) {
            return;
        }
        const float interp_value = offset_scale * dds + offset_y * ddy + offset_x * ddx;

        const float contrast = abs(at(z, y, x) + interp_value / 2.0);
        // This contrast can't be lower than the threshold here, since interpolating the extremum position
        // only increases the DoG response. 
        // You could have two thresholds though, one pre- and one post-interpolation if that's ever desired.
        // if (contrast < contrast_threshold) {
        //     return;
        // }

        // Check extremum edgeness  with hessian anisotropy criterion
        const float cm_tau_low = 0.7;
        const float cm_tau_high = 1.5;
        const float denom = (h22 + h33) * (h22 + h33);
        if (denom == 0) {
            return;
        }
        const float cm = 1 - 4 * (h22 * h33 - h23 * h23) / denom;
        if (cm_tau_low <= cm && cm <= cm_tau_high) {
            return;
        }

        float size = DOG_FIRST_SCALE_SIGMA * DOG_SIGMA_RADIUS_FACTOR * pow(2.0, float(z) + offset_scale);

        // printf("extr: %v3i\n", uvec3(x, y, z));
        const uint extremum_idx = atomicAdd(extremum_locations.n_extrema, 1);
        if (extremum_idx < max_extrema) {
            set_extremum_x_float(extremum_locations, extremum_idx, float(x) + offset_x);
            set_extremum_y_float(extremum_locations, extremum_idx, float(y) + offset_y);
            set_extremum_scale_float(extremum_locations, extremum_idx, size);
            set_extremum_contrast(extremum_locations, extremum_idx, contrast);
        }
        return;
    }
}
