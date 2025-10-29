layout (constant_id = 0) const uint MIN_SUBGROUP_SIZE = 8;
layout (constant_id = 1) const uint MAX_EXTREMA = 1;
layout (constant_id = 2) const uint MAX_RETRY_EXTREMA = 1;
layout (constant_id = 3) const uint MAX_KEYPOINTS = 1;
layout (constant_id = 4) const uint EXTREMUM_BLOCK_LEN = 256;
layout (constant_id = 5) const uint PATCH_PYRAMID_LEVELS = 5;
#define NUM_GLOBAL_CONSTANTS 6

layout(binding = 0, r32f) uniform image2DArray image_coarse; 
layout(binding = 1) uniform texture2DArray texture_coarse; 
// One image view for each pyramid mip level
layout(binding = 2, r32f) uniform image2D images[2 + PATCH_PYRAMID_LEVELS]; 
layout(binding = 3) uniform texture2D textures[3]; 
layout(binding = 4) uniform sampler samplers[2]; 

#define TEXTURE_SCRATCH             textures[0]
#define TEXTURE_PATCH_PYR_SCRATCH   textures[1]
#define TEXTURE_PATCH_PYR           textures[2]

#define IMAGE_SCRATCH              images[0]
#define IMAGE_PATCH_PYR_SCRATCH    images[1]
#define IMAGE_PATCH_PYR_MIP(L)     images[2 + L]

#define SAMPLER_BIL                samplers[0]
#define SAMPLER_NN                 samplers[1]

