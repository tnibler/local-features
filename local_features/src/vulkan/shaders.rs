use std::sync::Arc;

use itertools::Itertools;
use log::debug;
use vulkano::{
    memory::allocator::DeviceLayout,
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderStages, SpecializationConstant},
};
use vulkano_taskgraph::descriptor_set::BindlessContext;

use crate::vulkan::Vulkan;

use super::{BlobLocationsView, CandidateBlob, FixedParams, GpuExtremumIdx, Precision};

pub mod shaders_f32 {
    vulkano_shaders::shader! {
        vulkan_version: "1.2",
        define: [("PRECISION_FLOAT32", "1")],
        shaders: {
            blur: {
                path: "src/vulkan/shaders/blur.glsl",
                ty: "compute",
            },
            swt: {
                path: "src/vulkan/shaders/swt.glsl",
                ty: "compute",
            },
            swt_sub: {
                path: "src/vulkan/shaders/swt_sub.glsl",
                ty: "compute",
            },
            scan_extrema: {
                ty: "compute",
                path: "src/vulkan/shaders/scan_extrema.glsl",
            },
            blur_pyramid: {
                ty: "compute",
                path: "src/vulkan/shaders/blur_pyramid.glsl",
            },
            keypoint_orientation: {
                path: "src/vulkan/shaders/keypoint_orientation.glsl",
                ty: "compute",
            },
            patch_gradients: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/patch_gradients.glsl",
            },
            embedding_polar: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/embedding_polar.glsl",
            },
            embedding_cartesian: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/embedding_cartesian.glsl",
            },
            normalize: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/normalize.glsl",
            },
            whitening: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/whitening.glsl",
            },
            normalize_final: {
                ty: "compute",
                path: "src/vulkan/shaders/mkd/normalize_final.glsl",
            },
        },
    }
}

pub struct ComputePipelines {
    pub blur: Arc<ComputePipeline>,
    pub swt: Arc<ComputePipeline>,
    pub swtsub: Arc<ComputePipeline>,
    pub scan_extrema: Arc<ComputePipeline>,
    pub blur_pyramid: Arc<ComputePipeline>,
    pub keypoint_orientation: Arc<ComputePipeline>,
    pub patch_gradients: Arc<ComputePipeline>,
    pub embedding_polar: Arc<ComputePipeline>,
    pub embedding_cartesian: Arc<ComputePipeline>,
    pub normalize: Arc<ComputePipeline>,
    pub whitening: Arc<ComputePipeline>,
    pub normalize_final: Arc<ComputePipeline>,
}

pub fn create_pipelines(
    params: &FixedParams,
    vk: &Vulkan,
    bcx: &BindlessContext,
) -> Result<ComputePipelines, crate::vulkan::Error> {
    let min_subgroup_size = vk
        .device
        .physical_device()
        .properties()
        .min_subgroup_size
        .expect("TODO what now");
    debug!("min_subgroup_size: {}", min_subgroup_size);
    assert!(
        min_subgroup_size >= 8,
        "must have subgroup_size >= num_subgroups in all shaders, otherwise broken reductions"
    );
    let specialization_constants = [
        (0u32, SpecializationConstant::U32(min_subgroup_size)),
        // MAX_EXTREMA
        (1, SpecializationConstant::U32(params.max_extrema)),
        // MAX_KEYPOINTS
        (2, SpecializationConstant::U32(params.max_keypoints)),
        // EXTREMUM_BLOCK_LEN
        (3, SpecializationConstant::U32(params.extremum_block_len)),
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
    let swt_stage = shader_stage!(load_swt);
    let swtsub_stage = shader_stage!(load_swt_sub);
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
        &swt_stage,
        &swtsub_stage,
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

    let layout = PipelineLayout::new(
        &vk.device,
        &PipelineLayoutCreateInfo {
            set_layouts: &[bcx.global_set_layout()],
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
                PipelineShaderStageCreateInfo::new(&entry_point),
                &layout,
            ),
        )
    };

    let blur = make_pipeline(blur_stage)?;
    let swt = make_pipeline(swt_stage)?;
    let swtsub = make_pipeline(swtsub_stage)?;
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
        swt,
        swtsub,
        scan_extrema,
        blur_pyramid,
        keypoint_orientation,
        patch_gradients,
        embedding_polar,
        embedding_cartesian,
        normalize: embedding_sum,
        whitening,
        normalize_final: normalize,
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
) -> Result<ExtremumLocationsBufferLayout, crate::vulkan::Error> {
    let max_extrema = params.max_extrema;

    let elsize = size_of::<u32>();
    let offset_n_extrema = 0;
    let offset_coords: usize = 16 * elsize;

    let size_extremum_locations = u64::from(4 * max_extrema) * elsize as u64;
    let size_total = (1u64 + 15) * elsize as u64 + size_extremum_locations;
    assert!(size_extremum_locations <= size_total);
    let layout =
        DeviceLayout::from_size_alignment(size_total, 4u64).ok_or(crate::vulkan::Error::TODO)?;
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
            let (xs, ys, scales, contrasts) = &buffer
                [full_blocks_end..full_blocks_end + N_COORDS * block_len]
                .chunks_exact(block_len)
                .map(|chunk| &chunk[..n_tail_elems])
                .collect_tuple()
                .unwrap();

            BlobLocationsView {
                xs,
                ys,
                scales,
                contrasts,
            }
        });
        buffer[..full_blocks_end]
            .chunks_exact(N_COORDS * block_len)
            .map(move |block| {
                let (xs, ys, scales, contrasts) =
                    block.chunks_exact(block_len).collect_tuple().unwrap();
                BlobLocationsView {
                    xs,
                    ys,
                    scales,
                    contrasts,
                }
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

pub fn keypoints_buffer_layout(
    params: &FixedParams,
) -> Result<KeypointsBufferLayout, crate::vulkan::Error> {
    let max_keypoints = params.max_keypoints;

    let elsize = size_of::<u32>();
    let offset_n_keypoints: usize = 0;
    let offset_extremum_indices = 16 * elsize;
    let offset_keypoint_orientations = offset_extremum_indices + max_keypoints as usize * elsize;

    // one u32 index into filtered_extremum_indices, one f32 orientation per keypoint
    let size_keypoints = u64::from(2 * max_keypoints) * elsize as u64;
    let size_total = (1u64 + 15) * elsize as u64 + size_keypoints;
    assert!(size_keypoints <= size_total);
    let layout =
        DeviceLayout::from_size_alignment(size_total, 4u64).ok_or(crate::vulkan::Error::TODO)?;
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
) -> Result<FilteredExtremaBufferLayout, crate::vulkan::Error> {
    let size_total = u64::from(1 + params.max_extrema) * size_of::<u32>() as u64;
    let layout =
        DeviceLayout::from_size_alignment(size_total, 4).ok_or(crate::vulkan::Error::TODO)?;
    Ok(FilteredExtremaBufferLayout { layout, size_total })
}
