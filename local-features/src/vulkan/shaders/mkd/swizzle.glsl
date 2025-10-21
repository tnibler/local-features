#ifndef _SWIZZLE_GLSL
#define _SWIZZLE_GLSL

// Transform coordinates in patch into a flat index grouping positions by X/Y reflection.
// Order in the swizzled array is top left, top right, bottom left, bottom right.
uint swizzled_index(uint x, uint y) {
    const bool right = x >= PATCH_SIZE / 2;
    const bool lower = y >= PATCH_SIZE / 2;
    const uint quadrant = 2 * int(lower) + int(right);
    // position of matched friend in the top left quadrant
    const uint group_x = right ? (PATCH_SIZE - 1 - x) : x;
    const uint group_y = lower ? (PATCH_SIZE - 1 - y) : y;
    const uint index = (PATCH_SIZE / 2 * group_y + group_x) * 4 + quadrant;
    return index;
}

uvec2 unswizzle_index(uint idx) {
    const uint quadrant = idx % 4;
    const bool right = (quadrant % 2) == 1;
    const bool lower = (quadrant / 2) == 1;
    const uint group_idx = idx / 4;
    const uint group_x = group_idx % (PATCH_SIZE / 2);
    const uint group_y = group_idx / (PATCH_SIZE / 2);
    const uint x = right ? (PATCH_SIZE - 1 - group_x) : group_x;
    const uint y = lower ? (PATCH_SIZE - 1 - group_y) : group_y;
    return uvec2(x, y);
}

#endif // _SWIZZLE_GLSL
