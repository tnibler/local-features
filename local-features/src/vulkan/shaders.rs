use std::sync::Arc;

use itertools::Itertools;
use vulkano::{
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
    },
    memory::allocator::DeviceLayout,
    pipeline::{
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateFlags, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
    },
    shader::{ShaderStages, SpecializationConstant},
};

use crate::vulkan::Vulkan;

use super::{BlobLocationsView, CandidateBlob, FixedParams, GpuExtremumIdx, Precision};

pub mod shaders_f32 {
    vulkano_shaders::shader! {
        vulkan_version: "1.2",
        define: [("PRECISION_FLOAT32", "1")],
        shaders: {
            blur: {
                path: "src/vulkan/shaders/blur.comp",
                ty: "compute",
            },
            swt_dense: {
                path: "src/vulkan/shaders/blur.comp",
                ty: "compute",
                define: [("BINOMIAL_FILTER", "1")],
            },
            swt_sparse1: {
                path: "src/vulkan/shaders/swt_sparse1.comp",
                ty: "compute",
            },
            swt_sparse2: {
                path: "src/vulkan/shaders/swt_sparse.comp",
                ty: "compute",
            },
            swt: {
                path: "src/vulkan/shaders/swt.comp",
                ty: "compute",
            },
            scan_extrema: {
                ty: "compute",
                path: "src/vulkan/shaders/scan_extrema.comp",
            },
            blur_pyramid: {
                ty: "compute",
                path: "src/vulkan/shaders/blur_pyramid.comp",
            },
            keypoint_orientation: {
                path: "src/vulkan/shaders/keypoint_orientation.comp",
                ty: "compute",
            },
            patch_gradients: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/patch_gradients.comp",
            },
            embedding_polar: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/embedding_polar.comp",
                // bytes: "embedding_polar.spv",
            },
            embedding_cartesian: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/embedding_cartesian.comp",
                // bytes: "embedding_cartesian.spv",
            },
            normalize: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/normalize.comp",
            },
            whitening: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/whitening.comp",
            },
            normalize_final: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/normalize_final.comp",
            },
        },
    }
}

pub struct ComputePipelines {
    pub blur: Arc<ComputePipeline>,
    pub swt: Arc<ComputePipeline>,
    pub swt_sparse1: Arc<ComputePipeline>,
    pub swt_sparse2: Arc<ComputePipeline>,
    pub swt_dense: Arc<ComputePipeline>,
    pub scan_extrema: Arc<ComputePipeline>,
    pub blur_pyramid: Arc<ComputePipeline>,
    pub keypoint_orientation: Arc<ComputePipeline>,
    pub patch_gradients: Arc<ComputePipeline>,
    pub embedding_polar: Arc<ComputePipeline>,
    pub embedding_cartesian: Arc<ComputePipeline>,
    pub normalize: Arc<ComputePipeline>,
    pub whitening: Arc<ComputePipeline>,
    pub normalize_final: Arc<ComputePipeline>,

    pub descriptor_set_layout: Arc<DescriptorSetLayout>,
}

pub fn create_pipelines(params: &FixedParams, vk: &Vulkan) -> Result<ComputePipelines, crate::vulkan::VulkanError> {
    let min_subgroup_size = vk.device.physical_device().properties().min_subgroup_size.expect("TODO what now");
    let specialization_constants = [
        (0u32, SpecializationConstant::U32(min_subgroup_size)),
        // MAX_EXTREMA
        (1, SpecializationConstant::U32(params.max_extrema)),
        // MAX_KEYPOINTS
        (2, SpecializationConstant::U32(params.max_keypoints)),
        // EXTREMUM_BLOCK_LEN
        (3, SpecializationConstant::U32(params.extremum_block_len)),
        // PATCH_PYRAMID_LEVELS
        (4, SpecializationConstant::U32(params.patch_pyr_levels)),
    ];
    const SPECIALIZATION_PANIC_MSG: &str = "Wrong specialization constants";

    macro_rules! shader_stage {
        ($func:ident) => {{
            let shader = match params.precision {
                Precision::Float32 => shaders_f32::$func(&vk.device),
                Precision::Float16 => {
                    todo!()
                    // shaders::scale_space::detect_f16::$func(vk.device.clone())
                }
            }?
            .specialize(&specialization_constants)
            .expect(SPECIALIZATION_PANIC_MSG);
            shader.entry_point("main").expect("main exists")
        }};
    }
    let blur_stage = shader_stage!(load_blur);
    let swt_dense_stage = shader_stage!(load_swt_dense);
    let swt_stage = shader_stage!(load_swt);
    let swt_sparse1_stage = shader_stage!(load_swt_sparse1);
    let swt_sparse2_stage = shader_stage!(load_swt_sparse2);
    let scan_extrema_stage = shader_stage!(load_scan_extrema);
    let blur_pyramid_stage = shader_stage!(load_blur_pyramid);

    let keypoint_orientation_stage = shader_stage!(load_keypoint_orientation);
    let patch_gradients_stage = shader_stage!(load_patch_gradients);
    let embedding_polar_stage = shader_stage!(load_embedding_polar);
    let embedding_cartesian_stage = shader_stage!(load_embedding_cartesian);
    let embedding_sum_stage = shader_stage!(load_normalize);
    let whitening_stage = shader_stage!(load_whitening);
    let normalize_stage = shader_stage!(load_normalize_final);

    let entry_points = [
        &blur_stage,
        &swt_dense_stage,
        &swt_stage,
        &swt_sparse1_stage,
        &swt_sparse2_stage,
        &scan_extrema_stage,
        &blur_pyramid_stage,
        &keypoint_orientation_stage,
        &patch_gradients_stage,
        &embedding_polar_stage,
        &embedding_cartesian_stage,
        &embedding_sum_stage,
        &whitening_stage,
        &normalize_stage,
    ];

    let largest_push_constant_range = entry_points
        .iter()
        .filter_map(|ep| ep.info().push_constant_requirements)
        .max_by_key(|pc| pc.size)
        .expect("there are >0 push constant ranges");

    let n_storage_images = 2 + params.patch_pyr_levels;
    let n_sampled_images = 2 + 1;

    let descriptor_set_layout = DescriptorSetLayout::new(
        &vk.device,
        &DescriptorSetLayoutCreateInfo {
            bindings: &[
                // SWT blurred images
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    binding: 0,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::StorageImage)
                },
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    binding: 1,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::SampledImage)
                },
                // Scratch image, patch pyramid
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    binding: 2,
                    descriptor_count: n_storage_images,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::StorageImage)
                },
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    binding: 3,
                    descriptor_count: n_sampled_images,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::SampledImage)
                },
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    binding: 4,
                    descriptor_count: 2,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::Sampler)
                },
            ],
            ..Default::default()
        },
    )?;
    let layout = PipelineLayout::new(
        &vk.device,
        &PipelineLayoutCreateInfo {
            set_layouts: &[&descriptor_set_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: largest_push_constant_range.size,
            }],
            ..Default::default()
        },
    )?;
    let make_pipeline = |entry_point| {
        ComputePipeline::new(
            &vk.device,
            None,
            &ComputePipelineCreateInfo::new(
                PipelineShaderStageCreateInfo {
                    flags: unsafe { std::mem::transmute::<u32, PipelineShaderStageCreateFlags>(1u32) },
                    ..PipelineShaderStageCreateInfo::new(&entry_point)
                },
                &layout,
            ),
        )
    };

    let blur = make_pipeline(blur_stage)?;
    let swt_dense = make_pipeline(swt_dense_stage)?;
    let swt = make_pipeline(swt_stage)?;
    let swt_sparse1 = make_pipeline(swt_sparse1_stage)?;
    let swt_sparse2 = make_pipeline(swt_sparse2_stage)?;
    let scan_extrema = make_pipeline(scan_extrema_stage)?;
    let blur_pyramid = make_pipeline(blur_pyramid_stage)?;
    let keypoint_orientation = make_pipeline(keypoint_orientation_stage)?;
    let patch_gradients = make_pipeline(patch_gradients_stage)?;
    let embedding_polar = make_pipeline(embedding_polar_stage)?;
    let embedding_cartesian = make_pipeline(embedding_cartesian_stage)?;
    let embedding_sum = make_pipeline(embedding_sum_stage)?;
    let whitening = make_pipeline(whitening_stage)?;
    let normalize = make_pipeline(normalize_stage)?;

    Ok(ComputePipelines {
        blur,
        swt_dense,
        swt_sparse1,
        swt_sparse2,
        swt,
        scan_extrema,
        blur_pyramid,
        keypoint_orientation,
        patch_gradients,
        embedding_polar,
        embedding_cartesian,
        normalize: embedding_sum,
        whitening,
        normalize_final: normalize,

        descriptor_set_layout,
    })
}

#[derive(Debug, Clone)]
pub struct ExtremumLocationsBufferLayout {
    pub offset_n_extrema: usize,
    pub offset_coords: usize,

    pub size_total: u64,
    pub layout: DeviceLayout,
    block_len: usize,
}

pub fn extremum_locations_buffer_layout(
    params: &FixedParams,
) -> Result<ExtremumLocationsBufferLayout, crate::vulkan::VulkanError> {
    let max_extrema = params.max_extrema;

    let elsize = size_of::<u32>();
    let offset_n_extrema = 0;
    let offset_coords: usize = 16 * elsize;

    let size_extremum_locations = u64::from(4 * max_extrema) * elsize as u64;
    let size_total = (1u64 + 15) * elsize as u64 + size_extremum_locations;
    assert!(size_extremum_locations <= size_total);
    let layout = DeviceLayout::from_size_alignment(size_total, 4u64).ok_or(crate::vulkan::VulkanError::TODO)?;
    assert!(params.extremum_block_len > 0);
    Ok(ExtremumLocationsBufferLayout {
        offset_n_extrema,
        offset_coords,
        layout,
        size_total,
        block_len: params.extremum_block_len as usize,
    })
}

const N_COORDS: usize = 4;

impl ExtremumLocationsBufferLayout {
    /// Size in bytes of buffer range containing extremum coordinates
    pub fn coords_size(&self, n_extrema: u32) -> u64 {
        self.coords_len(n_extrema) as u64 * size_of::<u32>() as u64
    }

    /// Length of [[f32]] buffer range containing extremum coordinates
    pub fn coords_len(&self, n_extrema: u32) -> usize {
        (n_extrema as usize).div_ceil(self.block_len) * self.block_len * N_COORDS
    }

    pub fn coords_ranges<'a>(
        &self,
        buffer: &'a [f32],
        n_extrema: u32,
    ) -> impl IntoIterator<Item = BlobLocationsView<'a>> {
        assert!(buffer.len() >= self.coords_len(n_extrema));
        let n_full_blocks = n_extrema as usize / self.block_len;
        let n_tail_elems = n_extrema as usize - n_full_blocks * self.block_len;
        let block_len = self.block_len;
        let full_blocks_end = n_full_blocks * N_COORDS * block_len;
        let tail = (n_tail_elems > 0).then(|| {
            let (xs, ys, scales, contrasts) = &buffer[full_blocks_end..full_blocks_end + N_COORDS * block_len]
                .chunks_exact(block_len)
                .map(|chunk| &chunk[..n_tail_elems])
                .collect_tuple()
                .unwrap();

            assert_eq!(xs.len(), n_tail_elems);
            assert_eq!(ys.len(), n_tail_elems);
            assert_eq!(scales.len(), n_tail_elems);
            assert_eq!(contrasts.len(), n_tail_elems);
            BlobLocationsView { xs, ys, scales, contrasts }
        });
        buffer[..full_blocks_end]
            .chunks_exact(N_COORDS * block_len)
            .map(move |block| {
                let (xs, ys, scales, contrasts) = block.chunks_exact(block_len).collect_tuple().unwrap();
                BlobLocationsView { xs, ys, scales, contrasts }
            })
            .chain(tail)
    }

    pub fn get(&self, buffer: &[f32], idx: GpuExtremumIdx) -> CandidateBlob {
        let base = (idx.index() / self.block_len) * N_COORDS * self.block_len;
        let off = idx.index() % self.block_len;
        CandidateBlob {
            x: buffer[base + off],
            y: buffer[base + self.block_len + off],
            size: buffer[base + 2 * self.block_len + off],
            contrast: buffer[base + 3 * self.block_len + off],
            extremum_index: idx,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeypointsBufferLayout {
    pub offset_n_keypoints: usize,
    pub offset_extremum_indices: usize,
    pub offset_keypoint_orientations: usize,
    pub size_total: u64,
    pub layout: DeviceLayout,
}

pub fn keypoints_buffer_layout(params: &FixedParams) -> Result<KeypointsBufferLayout, crate::vulkan::VulkanError> {
    let max_keypoints = params.max_keypoints;

    let elsize = size_of::<u32>();
    let offset_n_keypoints: usize = 0;
    let offset_extremum_indices = 16 * elsize;
    let offset_keypoint_orientations = offset_extremum_indices + max_keypoints as usize * elsize;

    // one u32 index into filtered_extremum_indices, one f32 orientation per keypoint
    let size_keypoints = u64::from(2 * max_keypoints) * elsize as u64;
    let size_total = (1u64 + 15) * elsize as u64 + size_keypoints;
    assert!(size_keypoints <= size_total);
    let layout = DeviceLayout::from_size_alignment(size_total, 4u64).ok_or(crate::vulkan::VulkanError::TODO)?;
    Ok(KeypointsBufferLayout {
        offset_n_keypoints,
        offset_extremum_indices,
        offset_keypoint_orientations,
        size_total,
        layout,
    })
}

#[derive(Debug, Clone)]
pub struct FilteredExtremaBufferLayout {
    pub layout: DeviceLayout,
    pub size_total: u64,
}

pub fn filtered_extrema_buffer_layout(
    params: &FixedParams,
) -> Result<FilteredExtremaBufferLayout, crate::vulkan::VulkanError> {
    let size_total = u64::from(1 + params.max_extrema) * size_of::<u32>() as u64;
    let layout = DeviceLayout::from_size_alignment(size_total, 4).ok_or(crate::vulkan::VulkanError::TODO)?;
    Ok(FilteredExtremaBufferLayout { layout, size_total })
}
