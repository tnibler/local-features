use std::sync::Arc;

use float_ord::FloatOrd;
use fxhash::FxHashMap;
use index_vec::IndexVec;
use itertools::Itertools as _;
use log::{debug, trace};
use ndarray::{Array2, Array3, ArrayView2};
use shaders::ComputePipelines;
use thiserror::Error;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    descriptor_set::{DescriptorImageInfo, DescriptorSet, WriteDescriptorSet},
    device::Queue,
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageType, ImageUsage,
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    sync::{AccessFlags, PipelineStages},
};
use vulkano_taskgraph::{
    Id, QueueFamilyType, TaskContext,
    command_buffer::{BufferMemoryBarrier, CopyBufferInfo, DependencyInfo, RecordingCommandBuffer},
    graph::{CompileInfo, ExecutableTaskGraph, NodeId, ResourceMap, TaskGraph},
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources, ResourcesCreateInfo},
    resource_map,
};

use crate::{
    BuildTimeParams, DESCRIPTOR_LEN, DIMS_EMB_CARTESIAN, DIMS_EMB_POLAR, DIMS_INPUT, FeatureDetectParams,
    FeaturesResult, Keypoint, MKDPCA, PATCH_SIZE, RAW_DESCRIPTOR_LEN,
    vulkan::{patch_pyramid::BlitCopyImageTask, tasks_common::CopyBufferTask},
};

mod make_a_vulkan;
mod patch_pyramid;
mod shaders;
mod tasks_common;
mod tasks_detect;
mod tasks_extract;

pub use make_a_vulkan::{Vulkan, VulkanInitError};

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum VulkanError {
    #[error(transparent)]
    Vulkan(#[from] vulkano::VulkanError),
    #[error(transparent)]
    VulkanValidation(#[from] vulkano::Validated<vulkano::VulkanError>),
    #[error(transparent)]
    VulkanBufferAllocation(#[from] vulkano::Validated<vulkano::buffer::AllocateBufferError>),
    #[error(transparent)]
    VulkanImageAllocation(#[from] vulkano::Validated<vulkano::image::AllocateImageError>),
    #[error(transparent)]
    VulkanExecution(#[from] vulkano::command_buffer::CommandBufferExecError),

    #[error("TODO error types")]
    TODO,
}

const STRIDE_EMB_CARTESIAN: usize = DIMS_EMB_CARTESIAN;
const STRIDE_EMB_POLAR: usize = DIMS_EMB_POLAR;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CandidateBlob {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub contrast: f32,
    extremum_index: GpuExtremumIdx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)]
pub enum Precision {
    #[default]
    Float32,
    Float16,
}

pub struct LocalFeaturesVulkan {
    fixed_params: FixedParams,
    buffer_layouts: BufferLayouts,
    vk: Vulkan,
    resources: Arc<Resources>,
    physical_resources: PhysicalResources,
    detect_standalone: CompiledTaskGraph,
    extract_standalone: CompiledTaskGraph,
    detect_extract_all: CompiledTaskGraph,
    flight_id: Id<Flight>,

    // not fully sorted out, if ever multiple flights are allowed each flight needs these
    candidate_blobs_readback: IndexVec<FilteredExtremumIdx, CandidateBlob>,
    extremum_idxs: FxHashMap<GpuExtremumIdx, FilteredExtremumIdx>,
}

pub struct FilterBlobsOutput<'a> {
    pub indices: &'a mut Vec<u32>,
}

pub struct BlobLocationsView<'a> {
    pub xs: &'a [f32],
    pub ys: &'a [f32],
    pub scales: &'a [f32],
    pub contrasts: &'a [f32],
}

pub trait FilterBlobs {
    fn filter<'r, 'w>(
        &mut self,
        blobs: Box<dyn Iterator<Item = BlobLocationsView<'r>> + 'w>,
        out: FilterBlobsOutput<'w>,
    );
}

#[derive(Debug, Clone)]
struct FixedParams {
    pub precision: Precision,
    pub use_staging_buffers: StagingBuffers,
    pub max_image_width: u32,
    pub max_image_height: u32,
    pub n_scales: u32,
    pub max_extrema: u32,
    pub max_keypoints: u32,
    pub pca: MKDPCA,
    /// Extremum coordinates in the buffer are grouped into blocks, such that the layout within
    /// a block is e.g., [xs.., ys.., zs..] to limit how much empty buffer is copied around.
    pub extremum_block_len: u32,

    pub patch_pyr_levels: u32,
    pub debug_readout_patches: bool,
}

#[derive(Clone)]
struct BufferLayouts {
    extremum_locations: shaders::ExtremumLocationsBufferLayout,
    filtered_extrema: shaders::FilteredExtremaBufferLayout,
    keypoints: shaders::KeypointsBufferLayout,
}

struct CompiledTaskGraph {
    taskgraph: ExecutableTaskGraph<GlobalContext>,
    virtual_ids: BufferIds,
}

#[derive(Clone)]
struct BufferIds {
    /// Written only once upon initialization
    buf_constants: Id<Buffer>,
    /// Staging buffer: always used to store input image data before it's copied to a VkImage,
    /// optionally used for staging other inputs and results as well.
    buf_staging_write: Id<Buffer>,
    buf_staging_read: Id<Buffer>,
    buf_extremum_locations: Id<Buffer>,
    buf_filtered_extrema: Id<Buffer>,
    buf_keypoints: Id<Buffer>,
    buf_patch_gradients_or_raw_descriptors: Id<Buffer>,
    buf_embeddings_or_descriptors: Id<Buffer>,
    buf_staging_debug: Option<Id<Buffer>>,

    img_coarse: Id<Image>,
    img_scratch: Id<Image>,
    img_patch_pyr: Id<Image>,
    img_patch_pyr_tmp: Id<Image>,
}

#[derive(Clone)]
struct PhysicalResources {
    ids: BufferIds,
    descriptor_set: Arc<DescriptorSet>,
}

index_vec::define_index_type! {
    struct GpuExtremumIdx = u32;
    DISABLE_MAX_INDEX_CHECK = false;
}

index_vec::define_index_type! {
    struct FilteredExtremumIdx = u32;
    DISABLE_MAX_INDEX_CHECK = false;
}

unsafe impl bytemuck::Zeroable for GpuExtremumIdx {}
unsafe impl bytemuck::NoUninit for GpuExtremumIdx {}
unsafe impl bytemuck::AnyBitPattern for GpuExtremumIdx {}
unsafe impl bytemuck::Zeroable for FilteredExtremumIdx {}
unsafe impl bytemuck::AnyBitPattern for FilteredExtremumIdx {}

#[allow(dead_code)]
struct GlobalContext {
    physical_resources: PhysicalResources,
    buffer_layouts: BufferLayouts,
    fixed_params: FixedParams,

    input_image: *const [f32],
    border: u32,
    rt_max_extrema: u32,
    rt_max_keypoints: u32,
    image_width: u32,
    image_height: u32,
    padded_width: u32,
    extremum_min_scale: f32,
    extremum_skip_layers: u32,
    contrast_threshold: f32,
    edgeness_cm_low: f32,
    edgeness_cm_high: f32,
    fix_scale: bool,
    rt_patch_pyr_levels: u32,

    /// Number of scale space extrema (before orientation assignment)
    n_filtered_extrema: u32,
    patch_scale_factor: f32,
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum StagingBuffers {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlurDirection {
    Vertical,
    Horizontal,
}

impl LocalFeaturesVulkan {
    pub(crate) fn new(params: BuildTimeParams, vk: Vulkan) -> Result<Self, crate::vulkan::VulkanError> {
        let use_staging_buffers = StagingBuffers::Yes;
        let resources = Resources::new(
            &vk.device,
            &ResourcesCreateInfo {
                max_buffers: 100,
                max_images: 100,
                max_flights: 100,
                max_swapchains: 0,
                ..Default::default()
            },
        )?;

        let patch_pyr_levels =
            f32::min(params.max_image_width as f32, params.max_image_height as f32).log2().ceil().round() as u32;

        let block_len = 256;
        let fixed_params = FixedParams {
            precision: Precision::Float32,
            use_staging_buffers,
            max_image_width: params.max_image_width,
            max_image_height: params.max_image_height,
            n_scales: params.n_scales,
            max_extrema: block_len * params.max_blobs.div_ceil(block_len),
            max_keypoints: params.max_features,
            patch_pyr_levels,
            extremum_block_len: block_len,
            pca: params.pca,
            debug_readout_patches: params.debug_readout_patches,
        };

        let buffer_layouts = BufferLayouts {
            extremum_locations: shaders::extremum_locations_buffer_layout(&fixed_params)?,
            filtered_extrema: shaders::filtered_extrema_buffer_layout(&fixed_params)?,
            keypoints: shaders::keypoints_buffer_layout(&fixed_params)?,
        };

        let pipelines = shaders::create_pipelines(&fixed_params, &vk)?;
        let physical_resources = allocate_buffers(&resources, &vk, &pipelines, &fixed_params, &buffer_layouts)?;
        let flight_id = resources.create_flight(1)?;

        let detect_standalone = {
            let mut tg = TaskGraph::new(&resources);
            let virtual_ids = make_virtual_ids(&mut tg, &fixed_params);

            let last_detect_node =
                build_detect_taskgraph(&mut tg, &virtual_ids, &fixed_params, &buffer_layouts, &pipelines)?;

            if fixed_params.use_staging_buffers == StagingBuffers::Yes {
                let copy_size = buffer_layouts.extremum_locations.size_total;
                let copy_node_id = tg
                    .create_task_node(
                        "copy keypoints to staging",
                        QueueFamilyType::Transfer,
                        tasks_common::CopyBufferTask {
                            src: virtual_ids.buf_extremum_locations,
                            dst: virtual_ids.buf_staging_read,
                            size: copy_size,
                            src_offset: 0,
                        },
                    )
                    .buffer_access(virtual_ids.buf_staging_read, AccessTypes::COPY_TRANSFER_WRITE)
                    .buffer_access(virtual_ids.buf_extremum_locations, AccessTypes::COPY_TRANSFER_READ)
                    .build();
                tg.add_edge(last_detect_node, copy_node_id).unwrap();
            }

            let tg = unsafe {
                tg.compile(&CompileInfo { queues: &[&vk.queue], present_queue: None, flight_id, ..Default::default() })
                    .expect("TODO")
            };
            CompiledTaskGraph { taskgraph: tg, virtual_ids }
        };

        let extract_standalone = {
            let mut tg = TaskGraph::new(&resources);
            let virtual_ids = make_virtual_ids(&mut tg, &fixed_params);

            let chain_extract = if fixed_params.use_staging_buffers == StagingBuffers::Yes {
                // Copy in host filtered extremum indices from staging
                let copy_node_id = tg
                    .create_task_node(
                        "copy from staging",
                        QueueFamilyType::Transfer,
                        tasks_common::CopyBufferTask {
                            src: virtual_ids.buf_staging_write,
                            dst: virtual_ids.buf_filtered_extrema,
                            // copying more than necessary, could use n_filtered_extrema instead
                            size: buffer_layouts.filtered_extrema.size_total,
                            src_offset: 0,
                        },
                    )
                    .buffer_access(virtual_ids.buf_staging_write, AccessTypes::COPY_TRANSFER_READ)
                    .buffer_access(virtual_ids.buf_filtered_extrema, AccessTypes::COPY_TRANSFER_WRITE)
                    .build();
                ChainExtractTg::ConnectTo { connect_node: copy_node_id, use_host_filtered_extremum_indices: true }
            } else {
                ChainExtractTg::Standalone
            };

            let last_node = build_extract_taskgraph(&mut tg, chain_extract, &virtual_ids, &fixed_params, &pipelines)?;

            if fixed_params.use_staging_buffers == StagingBuffers::Yes {
                let copy_node_id = tg
                    .create_task_node(
                        "copy keypoints+descriptors to staging",
                        QueueFamilyType::Transfer,
                        tasks_extract::DescriptorAndKeypointCopyTask {
                            buffers: virtual_ids.clone(),
                            read_extremum_locations: false,
                        },
                    )
                    .buffer_access(virtual_ids.buf_staging_read, AccessTypes::COPY_TRANSFER_WRITE)
                    .buffer_access(virtual_ids.buf_embeddings_or_descriptors, AccessTypes::COPY_TRANSFER_READ)
                    .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COPY_TRANSFER_READ)
                    .build();
                tg.add_edge(last_node, copy_node_id).unwrap();
            }
            let tg = unsafe {
                tg.compile(&CompileInfo { queues: &[&vk.queue], present_queue: None, flight_id, ..Default::default() })
                    .expect("TODO")
            };
            CompiledTaskGraph { taskgraph: tg, virtual_ids }
        };

        let detect_extract_all = {
            let mut tg = TaskGraph::new(&resources);
            let virtual_ids = make_virtual_ids(&mut tg, &fixed_params);

            let last_detect_node =
                build_detect_taskgraph(&mut tg, &virtual_ids, &fixed_params, &buffer_layouts, &pipelines)?;

            let chain_extract =
                ChainExtractTg::ConnectTo { connect_node: last_detect_node, use_host_filtered_extremum_indices: false };
            let last_node = build_extract_taskgraph(&mut tg, chain_extract, &virtual_ids, &fixed_params, &pipelines)?;

            if fixed_params.use_staging_buffers == StagingBuffers::Yes {
                let copy_node_id = tg
                    .create_task_node(
                        "copy keypoints+descriptors to staging",
                        QueueFamilyType::Transfer,
                        tasks_extract::DescriptorAndKeypointCopyTask {
                            buffers: virtual_ids.clone(),
                            read_extremum_locations: true,
                        },
                    )
                    .buffer_access(virtual_ids.buf_staging_read, AccessTypes::COPY_TRANSFER_WRITE)
                    .buffer_access(virtual_ids.buf_embeddings_or_descriptors, AccessTypes::COPY_TRANSFER_READ)
                    .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COPY_TRANSFER_READ)
                    .buffer_access(virtual_ids.buf_extremum_locations, AccessTypes::COPY_TRANSFER_READ)
                    .build();
                tg.add_edge(last_node, copy_node_id).unwrap();
            }

            let tg = unsafe {
                tg.compile(&CompileInfo { queues: &[&vk.queue], present_queue: None, flight_id, ..Default::default() })
                    .expect("TODO")
            };
            CompiledTaskGraph { taskgraph: tg, virtual_ids }
        };

        upload_constant_data(&fixed_params, &physical_resources, &vk.queue, &resources, flight_id)?;

        Ok(Self {
            vk,
            resources,
            fixed_params,
            buffer_layouts,
            detect_standalone,
            extract_standalone,
            detect_extract_all,
            physical_resources,
            flight_id,
            candidate_blobs_readback: IndexVec::new(),
            extremum_idxs: Default::default(),
        })
    }

    pub fn detect_extract_all(
        &mut self,
        img: &ArrayView2<f32>,
        params: &FeatureDetectParams,
    ) -> Result<FeaturesResult, crate::vulkan::VulkanError> {
        let resource_map = resource_map(
            &self.detect_extract_all.taskgraph,
            &self.detect_extract_all.virtual_ids,
            &self.physical_resources.ids,
        );
        let mut global_context = self.make_global_context(img, params);
        global_context.n_filtered_extrema = global_context.rt_max_extrema;
        unsafe {
            self.detect_extract_all.taskgraph.execute(resource_map, &global_context, || {}).unwrap();
        }
        self.resources.flight(self.detect_extract_all.taskgraph.flight_id()).unwrap().wait(None)?;

        let mut keypoints: Vec<Keypoint> = Vec::new();
        let mut descriptors_flat: Vec<f32> = Vec::new();
        let mut dropped_features: u32 = 0;
        let mut dropped_extrema: u32 = 0;
        let readback_host_buffer_accesses: &[_] = match self.fixed_params.use_staging_buffers {
            StagingBuffers::Yes => &[(self.physical_resources.ids.buf_staging_read, HostAccessType::Read)],
            StagingBuffers::No => todo!(),
        };
        unsafe {
            vulkano_taskgraph::execute(
                &self.vk.queue,
                &self.resources,
                self.flight_id,
                |_cbf: &mut RecordingCommandBuffer<'_>, tcx: &mut TaskContext<'_>| {
                    let keypoints_layout = &self.buffer_layouts.keypoints;
                    let copy_keypoints_size = keypoints_layout.size_total;
                    let copy_descriptors_size =
                        u64::from(self.fixed_params.max_keypoints) * DESCRIPTOR_LEN as u64 * size_of::<f32>() as u64;
                    let copy_extremum_locations_size = self.buffer_layouts.extremum_locations.size_total;
                    let descriptors_offset = copy_keypoints_size;
                    let copy_size = copy_keypoints_size + copy_descriptors_size + copy_extremum_locations_size;

                    let buf: &[u32] = tcx.read_buffer(self.physical_resources.ids.buf_staging_read, ..copy_size)?;
                    let extremum_locations_offset = copy_descriptors_size + copy_keypoints_size;
                    let (buf_keypoints_descriptors, buf_extremum_locations) =
                        buf.split_at(extremum_locations_offset as usize / size_of::<u32>());
                    let (buf_keypoints, buf_descriptors) =
                        buf_keypoints_descriptors.split_at(descriptors_offset as usize / size_of::<u32>());

                    let n_scanned_extrema = buf_extremum_locations[0];
                    let n_extrema = n_scanned_extrema.min(global_context.rt_max_extrema);
                    dropped_extrema = n_scanned_extrema.saturating_sub(global_context.rt_max_extrema);
                    // be sure to chop off the n_extrema field before the coordinates
                    let buf_extremum_locations: &[f32] = bytemuck::cast_slice(
                        &buf_extremum_locations
                            [self.buffer_layouts.extremum_locations.offset_coords / size_of::<u32>()..],
                    );

                    if cfg!(debug_assertions) {
                        let mut istart = 0;
                        for view in self
                            .buffer_layouts
                            .extremum_locations
                            .coords_ranges(buf_extremum_locations, n_extrema)
                            .into_iter()
                        {
                            for ((i, x), y, scale, contrast) in
                                itertools::izip!(view.xs.iter().enumerate(), view.ys, view.scales, view.contrasts)
                            {
                                assert_eq!(
                                    CandidateBlob {
                                        x: *x,
                                        y: *y,
                                        size: *scale,
                                        contrast: *contrast,
                                        extremum_index: (istart + i).into()
                                    },
                                    self.buffer_layouts
                                        .extremum_locations
                                        .get(buf_extremum_locations, (i + istart).into())
                                );
                            }
                            istart += view.xs.len();
                        }
                    }

                    let n_total_keypoints = buf_keypoints[keypoints_layout.offset_n_keypoints / size_of::<u32>()];
                    let n_keypoints = n_total_keypoints.min(global_context.rt_max_keypoints) as usize;
                    dropped_features = n_total_keypoints.saturating_sub(global_context.rt_max_keypoints);
                    debug!(
                        "Keypoints readback: {}, ({} total, {} dropped)",
                        n_keypoints, n_total_keypoints, dropped_features
                    );

                    let start_orientations = keypoints_layout.offset_keypoint_orientations / size_of::<u32>();
                    let orientations = bytemuck::cast_slice::<_, f32>(&buf_keypoints[start_orientations..]);
                    let location_indices =
                        &buf_keypoints[keypoints_layout.offset_extremum_indices / size_of::<u32>()..start_orientations];
                    let read_keypoints = location_indices[..n_keypoints]
                        .iter()
                        .map(|idx| GpuExtremumIdx::from_raw(*idx))
                        .zip(&orientations[..n_keypoints])
                        .map(|(gpu_idx, orientation)| {
                            let blob = self.buffer_layouts.extremum_locations.get(buf_extremum_locations, gpu_idx);
                            Keypoint {
                                x: blob.x,
                                y: blob.y,
                                size: blob.size,
                                angle: *orientation,
                                response: blob.contrast,
                            }
                        });
                    keypoints.extend(read_keypoints);

                    let buf_descriptors = bytemuck::cast_slice(&buf_descriptors[..(n_keypoints * DESCRIPTOR_LEN)]);
                    descriptors_flat.extend(buf_descriptors);

                    Ok(())
                },
                readback_host_buffer_accesses.iter().copied(),
                [],
                [],
            )
            .expect("TODO");
        }
        self.resources.flight(self.flight_id).unwrap().wait(None)?;
        assert_eq!(descriptors_flat.len() % 128, 0);
        let descriptors =
            Array2::from_shape_vec((keypoints.len(), DESCRIPTOR_LEN), descriptors_flat).expect("shape is correct");
        Ok(FeaturesResult { keypoints, descriptors, dropped_blobs: dropped_extrema, dropped_features })
    }

    pub fn detect_top_n(
        &mut self,
        img: &ArrayView2<f32>,
        n: u32,
        params: &FeatureDetectParams,
    ) -> Result<FeaturesResult, crate::vulkan::VulkanError> {
        let mut filter = TopKContrastFilter { n };
        self.detect(img, params, Some(&mut filter))
    }

    pub fn detect(
        &mut self,
        img: &ArrayView2<f32>,
        params: &FeatureDetectParams,
        filter_keypoints: Option<&mut dyn FilterBlobs>,
    ) -> Result<FeaturesResult, crate::vulkan::VulkanError> {
        assert!(img.as_slice().is_some());

        self.candidate_blobs_readback.clear();
        self.extremum_idxs.clear();
        let detect_resource_map = resource_map(
            &self.detect_standalone.taskgraph,
            &self.detect_standalone.virtual_ids,
            &self.physical_resources.ids,
        );

        let mut global_context = self.make_global_context(img, params);

        unsafe {
            self.detect_standalone.taskgraph.execute(detect_resource_map, &global_context, || {}).unwrap();
        }
        self.resources.flight(self.detect_standalone.taskgraph.flight_id()).unwrap().wait(None)?;

        let start = std::time::Instant::now();
        let dropped_extrema: u32 = self.do_extremum_readback(&mut global_context, filter_keypoints)?;
        debug!("Extrema readback took {:?}", start.elapsed());

        let extract_resource_map = resource_map(
            &self.extract_standalone.taskgraph,
            &self.extract_standalone.virtual_ids,
            &self.physical_resources.ids,
        );

        trace!("Execute extract taskgraph");

        unsafe {
            self.extract_standalone.taskgraph.execute(extract_resource_map, &global_context, || {}).unwrap();
        }
        trace!("Wait on extract flight");
        self.resources.flight(self.extract_standalone.taskgraph.flight_id()).unwrap().wait(None)?;
        trace!("Waited on extract flight");

        let mut keypoints: Vec<Keypoint> = Vec::new();
        let mut descriptors_flat: Vec<f32> = Vec::new();
        let mut dropped_features: u32 = 0;
        if self.fixed_params.use_staging_buffers == StagingBuffers::Yes {
            // Keypoints and descriptors have already been copied to staging
            let readback_host_buffer_accesses = [(self.physical_resources.ids.buf_staging_read, HostAccessType::Read)];
            let readback_buffer_accesses = [];
            let readback_image_accesses = [];
            let start = std::time::Instant::now();
            trace!("Executing descriptor readback taskgraph");
            unsafe {
                vulkano_taskgraph::execute(
                    &self.vk.queue,
                    &self.resources,
                    self.flight_id,
                    |_cbf: &mut RecordingCommandBuffer<'_>, tcx: &mut TaskContext<'_>| {
                        let keypoints_layout = &self.buffer_layouts.keypoints;
                        let copy_keypoints_size = keypoints_layout.size_total;
                        let copy_descriptors_size = u64::from(self.fixed_params.max_keypoints)
                            * DESCRIPTOR_LEN as u64
                            * size_of::<f32>() as u64;
                        let descriptors_offset = copy_keypoints_size;
                        let copy_size = copy_keypoints_size + copy_descriptors_size;
                        let buf: &[u32] = tcx.read_buffer(self.physical_resources.ids.buf_staging_read, ..copy_size)?;

                        let n_total_keypoints = buf[keypoints_layout.offset_n_keypoints / size_of::<u32>()];
                        let n_keypoints = n_total_keypoints.min(global_context.rt_max_keypoints) as usize;
                        dropped_features = n_total_keypoints.saturating_sub(global_context.rt_max_keypoints);
                        debug!(
                            "Keypoints readback: {}, ({} total, {} dropped)",
                            n_keypoints, n_total_keypoints, dropped_features
                        );

                        let start_orientations = keypoints_layout.offset_keypoint_orientations / size_of::<u32>();
                        let orientations = bytemuck::cast_slice::<_, f32>(&buf[start_orientations..]);
                        let location_indices =
                            &buf[keypoints_layout.offset_extremum_indices / size_of::<u32>()..start_orientations];
                        let read_keypoints = location_indices[..n_keypoints]
                            .iter()
                            .map(|idx| GpuExtremumIdx::from_raw(*idx))
                            .zip(&orientations[..n_keypoints])
                            .map(|(gpu_idx, orientation)| {
                                let cpu_idx: FilteredExtremumIdx = self.extremum_idxs[&gpu_idx];
                                let blob = &self.candidate_blobs_readback[cpu_idx];
                                Keypoint {
                                    x: blob.x,
                                    y: blob.y,
                                    size: blob.size,
                                    angle: *orientation,
                                    response: blob.contrast,
                                }
                            });
                        keypoints.extend(read_keypoints);

                        let descriptors_len = n_keypoints * DESCRIPTOR_LEN;
                        let descriptors_start = descriptors_offset as usize / size_of::<u32>();
                        let read_descriptors = bytemuck::cast_slice::<u32, f32>(
                            &buf[descriptors_start..descriptors_start + descriptors_len],
                        );
                        descriptors_flat.extend(read_descriptors);
                        Ok(())
                    },
                    readback_host_buffer_accesses,
                    readback_buffer_accesses,
                    readback_image_accesses,
                )
                .expect("TODO");
                trace!("Wait on descriptor readback flight");
                self.resources.flight(self.flight_id).unwrap().wait(None)?;
                trace!("Waited on descriptor readback flight");
            }
            debug!("Descriptor readback took {:?}", start.elapsed());
        } else {
            let readback_host_buffer_accesses = [
                (self.physical_resources.ids.buf_keypoints, HostAccessType::Read),
                (self.physical_resources.ids.buf_embeddings_or_descriptors, HostAccessType::Read),
            ];
            let readback_buffer_accesses = [
                (self.physical_resources.ids.buf_keypoints, AccessTypes::COPY_TRANSFER_READ),
                (self.physical_resources.ids.buf_embeddings_or_descriptors, AccessTypes::COPY_TRANSFER_READ),
            ];
            let readback_image_accesses = [];
            unsafe {
                vulkano_taskgraph::execute(
                    &self.vk.queue,
                    &self.resources,
                    self.flight_id,
                    |_cbf: &mut RecordingCommandBuffer<'_>, _tcx: &mut TaskContext<'_>| todo!(),
                    readback_host_buffer_accesses,
                    readback_buffer_accesses,
                    readback_image_accesses,
                )
                .expect("TODO");
            }
            todo!()
        }

        assert_eq!(descriptors_flat.len() % 128, 0);
        let descriptors =
            Array2::from_shape_vec((keypoints.len(), DESCRIPTOR_LEN), descriptors_flat).expect("shape is correct");
        Ok(FeaturesResult { keypoints, descriptors, dropped_blobs: dropped_extrema, dropped_features })
    }

    pub fn debug_read_patches(&mut self) -> Result<Array3<f32>, VulkanError> {
        assert!(
            self.fixed_params.debug_readout_patches,
            "Instance must have been created with debug_readout_patches set"
        );
        let mut data: Vec<f32> = Vec::with_capacity(self.fixed_params.max_keypoints as usize * 32 * 32);
        let buf_staging_debug = self.physical_resources.ids.buf_staging_debug.expect("must be Some");
        unsafe {
            vulkano_taskgraph::execute(
                &self.vk.queue,
                &self.resources,
                self.flight_id,
                |_cbf, tcx| {
                    let copy_size = u64::from(self.fixed_params.max_keypoints) * 32 * 32 * size_of::<f32>() as u64;
                    let buf: &[f32] = tcx.read_buffer(buf_staging_debug, ..copy_size)?;
                    data.extend(buf);
                    Ok(())
                },
                [(buf_staging_debug, HostAccessType::Read)],
                [],
                [],
            )
            .expect("TODO");
        }
        self.resources.flight(self.flight_id).unwrap().wait(None)?;
        Ok(Array3::from_shape_vec((self.fixed_params.max_keypoints as usize, 32, 32), data).expect("shape is correct"))
    }

    fn do_extremum_readback(
        &mut self,
        global_context: &mut GlobalContext,
        filter_keypoints: Option<&mut dyn FilterBlobs>,
    ) -> Result<u32, crate::vulkan::VulkanError> {
        let readback_from_buffer = match self.fixed_params.use_staging_buffers {
            StagingBuffers::Yes => self.physical_resources.ids.buf_staging_read,
            StagingBuffers::No => self.physical_resources.ids.buf_extremum_locations,
        };
        let result_buffer_id = match self.fixed_params.use_staging_buffers {
            StagingBuffers::Yes => self.physical_resources.ids.buf_staging_write,
            StagingBuffers::No => self.physical_resources.ids.buf_filtered_extrema,
        };
        let readback_host_buffer_accesses =
            [(readback_from_buffer, HostAccessType::Read), (result_buffer_id, HostAccessType::Write)];
        let readback_buffer_accesses = [(readback_from_buffer, AccessTypes::COPY_TRANSFER_READ)];
        let readback_image_accesses = [];
        let mut dropped_extrema: u32 = 0;
        unsafe {
            vulkano_taskgraph::execute(
                &self.vk.queue,
                &self.resources,
                self.flight_id,
                |_cbf: &mut RecordingCommandBuffer<'_>, tcx: &mut TaskContext<'_>| {
                    let layout = &self.buffer_layouts.extremum_locations;

                    let head_size = size_of::<u32>() as u64;
                    let n_scanned_extrema: u32 = *tcx.read_buffer::<u32>(readback_from_buffer, 0..head_size)?;
                    let n_extrema = n_scanned_extrema.min(global_context.rt_max_extrema);
                    assert!(n_extrema <= self.fixed_params.max_extrema);

                    // number of extrema before refinement/discaring bad extrema, so upper bound
                    // on how many there actually are
                    dropped_extrema = n_scanned_extrema.saturating_sub(global_context.rt_max_extrema);
                    debug!(
                        "Extrema readback: {}, ({} total, {} dropped)",
                        n_extrema, n_scanned_extrema, dropped_extrema
                    );

                    if n_extrema == 0 {
                        return Ok(());
                    }
                    let read_start = layout.offset_coords as u64;
                    let read_end = read_start + layout.coords_size(n_extrema);
                    let blob_coords = tcx.read_buffer::<[f32]>(readback_from_buffer, read_start..read_end)?.to_vec();

                    let mut indices: Vec<u32> = Vec::new();
                    if let Some(filter) = filter_keypoints {
                        let extrema_iter = layout.coords_ranges(&blob_coords, n_extrema);
                        filter.filter(Box::new(extrema_iter.into_iter()), FilterBlobsOutput { indices: &mut indices });
                    } else {
                        indices.extend(0..n_extrema);
                    }
                    debug!("Extrema after filtering: {}", indices.len());
                    assert!(indices.len() <= self.fixed_params.max_extrema as usize);

                    let filtered_buffer_size = u64::from(n_extrema + 1) * size_of::<u32>() as u64;
                    let result_buffer: &mut [u32] = tcx.write_buffer(result_buffer_id, ..filtered_buffer_size)?;
                    let (out_count, out_indices) = result_buffer.split_first_mut().expect("slice has length >= 1");
                    *out_count = indices.len() as u32;
                    out_indices[..indices.len()].copy_from_slice(&indices);
                    global_context.n_filtered_extrema = indices.len() as u32;

                    for gpu_idx in indices.iter().copied().map(GpuExtremumIdx::from_raw) {
                        let blob = layout.get(&blob_coords, gpu_idx);
                        let idx = self.candidate_blobs_readback.push(blob);
                        self.extremum_idxs.insert(gpu_idx, idx);
                    }
                    trace!("Wrote {} retained indices", indices.len());

                    Ok(())
                },
                readback_host_buffer_accesses,
                readback_buffer_accesses,
                readback_image_accesses,
            )
            .expect("TODO");
        }
        self.resources.flight(self.flight_id).unwrap().wait(None)?;
        trace!("Extremum readback: Waited on flight");
        Ok(dropped_extrema)
    }

    fn make_global_context(&self, img: &ArrayView2<f32>, params: &FeatureDetectParams) -> GlobalContext {
        let width: u32 = img.ncols().try_into().expect("TODO");
        let height: u32 = img.nrows().try_into().expect("TODO");
        let rt_patch_pyr_levels = f32::min(width as f32, height as f32).log2().ceil().round() as u32;
        let rt_max_extrema = self.fixed_params.max_extrema;
        let rt_max_keypoints = self.fixed_params.max_keypoints;
        let img_bytes = match self.fixed_params.precision {
            Precision::Float32 => unsafe { std::slice::from_raw_parts(img.as_ptr() as *const _, img.len()) },
            Precision::Float16 => {
                todo!()
            }
        };
        GlobalContext {
            buffer_layouts: self.buffer_layouts.clone(),
            fixed_params: self.fixed_params.clone(),
            image_height: height,
            image_width: width,
            padded_width: width,
            extremum_min_scale: params.min_keypoint_scale,
            // TODO: start layer based on minimum feature size
            extremum_skip_layers: 0,
            contrast_threshold: params.extremum_min_response,
            // reference to img is valid during this entire function, and the pointer created here
            // is only ever read while a flight is executing, which we all wait on in this function
            input_image: img_bytes,
            physical_resources: self.physical_resources.clone(),
            // TODO: check if extremum scan works with border
            border: 0,
            rt_max_extrema,
            rt_max_keypoints,
            rt_patch_pyr_levels,
            patch_scale_factor: params.patch_scale_factor,

            // set after blob filtering
            n_filtered_extrema: 0,

            // unused right now
            edgeness_cm_low: params.edgeness_cm_low,
            edgeness_cm_high: params.edgeness_cm_high,
            fix_scale: false,
        }
    }
}

fn allocate_buffers(
    resources: &Resources,
    vk: &Vulkan,
    pipelines: &ComputePipelines,
    params: &FixedParams,
    buffers: &BufferLayouts,
) -> Result<PhysicalResources, crate::vulkan::VulkanError> {
    let img_size: u64 = u64::from(params.max_image_width) * u64::from(params.max_image_height);
    let img_size = img_size + 4 - (img_size % 4);
    let max_keypoints: u64 = params.max_keypoints.into();
    let patch_size_sq: u64 = (PATCH_SIZE * PATCH_SIZE).into();

    let img_buffer_layout = DeviceLayout::new_unsized::<[f32]>(img_size).unwrap();

    let constant_data_layout = DeviceLayout::new_sized::<shaders::shaders_f32::ConstantData>();

    let patch_gradients_len: u64 = if params.debug_readout_patches { 3 } else { 2 } * max_keypoints * patch_size_sq;
    let raw_descriptors_len: u64 = max_keypoints * RAW_DESCRIPTOR_LEN as u64;
    let patch_gradients_or_raw_descriptors_layout =
        DeviceLayout::new_unsized::<[f32]>(patch_gradients_len.max(raw_descriptors_len)).expect("TODO");

    let embeddings_len: u64 =
        (DIMS_INPUT * (STRIDE_EMB_POLAR + STRIDE_EMB_CARTESIAN)) as u64 * u64::from(params.max_keypoints);
    let descriptors_len: u64 = DESCRIPTOR_LEN as u64 * max_keypoints;
    let embeddings_or_descriptors_layout =
        DeviceLayout::new_unsized::<[f32]>(descriptors_len.max(embeddings_len)).unwrap();

    let buf_staging_write: Id<Buffer> = match params.use_staging_buffers {
        StagingBuffers::Yes => {
            let staging_layout = {
                let align: u64 = 4; // only u32 and f32 in all buffers
                let size =
                    [constant_data_layout.size(), img_buffer_layout.size(), buffers.filtered_extrema.layout.size()]
                        .into_iter()
                        .max()
                        .expect("array not empty");
                DeviceLayout::from_size_alignment(size, align)
            }
            .ok_or(crate::vulkan::VulkanError::TODO)?;
            resources.create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_HOST,
                    ..Default::default()
                },
                staging_layout,
            )?
        }
        StagingBuffers::No => resources.create_buffer(
            &BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST, ..Default::default() },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            img_buffer_layout,
        )?,
    };
    let buf_staging_read = {
        let layout = {
            let align: u64 = 4; // only u32 and f32 in all buffers
            let size = buffers.extremum_locations.layout.size()
                + descriptors_len * size_of::<f32>() as u64
                + buffers.keypoints.layout.size();
            DeviceLayout::from_size_alignment(size, align)
        }
        .ok_or(crate::vulkan::VulkanError::TODO)?;
        resources.create_buffer(
            &BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST, ..Default::default() },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS | MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            layout,
        )?
    };
    let buf_staging_debug = if params.debug_readout_patches {
        let patches_len = u64::from(params.max_keypoints) * 32 * 32;
        let buf = resources.create_buffer(
            &BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST, ..Default::default() },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS | MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[f32]>(patches_len).expect("is valid buffer size"),
        )?;
        Some(buf)
    } else {
        None
    };

    let make_device_buffer = |layout| {
        resources.create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            &AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
            layout,
        )
    };

    // TODO: if not staging buffers, host_random_access and stuff
    let buf_constants = make_device_buffer(constant_data_layout)?;
    let buf_extremum_locations = make_device_buffer(buffers.extremum_locations.layout)?;
    let buf_filtered_extrema = make_device_buffer(buffers.filtered_extrema.layout)?;
    let buf_keypoint_indices = make_device_buffer(buffers.keypoints.layout)?;
    let buf_patch_gradients_or_raw_descriptors = make_device_buffer(patch_gradients_or_raw_descriptors_layout)?;
    let buf_embeddings_or_descriptors = make_device_buffer(embeddings_or_descriptors_layout)?;

    let img_coarse_id = resources.create_image(
        &ImageCreateInfo {
            image_type: ImageType::Dim2d,
            array_layers: params.n_scales + 3,
            format: match params.precision {
                Precision::Float32 => Format::R32_SFLOAT,
                Precision::Float16 => Format::R16_SFLOAT,
            },
            extent: [params.max_image_width, params.max_image_height, 1],
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        &AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
    )?;

    let img_scratch_id = resources.create_image(
        &ImageCreateInfo {
            format: match params.precision {
                Precision::Float32 => Format::R32_SFLOAT,
                Precision::Float16 => Format::R16_SFLOAT,
            },
            extent: [params.max_image_width, params.max_image_height, 1],
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
        &AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
    )?;

    let img_patch_pyr_scratch_id = resources.create_image(
        &ImageCreateInfo {
            format: Format::R32_SFLOAT,
            extent: [params.max_image_width / 2, params.max_image_height / 2, 1],
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
    )?;

    let img_patch_pyr_id = resources.create_image(
        &ImageCreateInfo {
            mip_levels: params.patch_pyr_levels,
            extent: [params.max_image_width, params.max_image_height, 1],
            format: Format::R32_SFLOAT,
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
    )?;

    let img_coarse = resources.image(img_coarse_id).unwrap();
    let imgview_coarse = ImageView::new(
        img_coarse.image(),
        &ImageViewCreateInfo {
            view_type: ImageViewType::Dim2dArray,
            format: match params.precision {
                Precision::Float32 => Format::R32_SFLOAT,
                Precision::Float16 => Format::R16_SFLOAT,
            },
            subresource_range: img_coarse.image().subresource_range(),
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
    )?;

    let img_scratch = resources.image(img_scratch_id).unwrap();
    let imgview_scratch = ImageView::new(
        img_scratch.image(),
        &ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            format: match params.precision {
                Precision::Float32 => Format::R32_SFLOAT,
                Precision::Float16 => Format::R16_SFLOAT,
            },
            subresource_range: img_scratch.image().subresource_range(),
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
    )?;

    let img_patch_pyr = resources.image(img_patch_pyr_id).unwrap();
    let imgview_patch_pyr = ImageView::new(
        img_patch_pyr.image(),
        &ImageViewCreateInfo {
            format: Format::R32_SFLOAT,
            subresource_range: img_patch_pyr.image().subresource_range(),
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
    )?;

    let imgviews_patch_pyr_mips: Vec<Arc<ImageView>> = (0..params.patch_pyr_levels)
        .map(|mip_level| {
            ImageView::new(
                img_patch_pyr.image(),
                &ImageViewCreateInfo {
                    view_type: ImageViewType::Dim2d,
                    format: Format::R32_SFLOAT,
                    usage: ImageUsage::STORAGE,
                    subresource_range: vulkano::image::ImageSubresourceRange {
                        aspects: ImageAspects::COLOR,
                        base_mip_level: mip_level,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
        })
        .try_collect()?;

    let img_patch_pyr_scratch = resources.image(img_patch_pyr_scratch_id).unwrap();
    let imgview_patch_pyr_scratch = ImageView::new(
        img_patch_pyr_scratch.image(),
        &ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            format: Format::R32_SFLOAT,
            subresource_range: img_patch_pyr_scratch.image().subresource_range(),
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
    )?;

    let sampler_bil = Sampler::new(
        &vk.device,
        &SamplerCreateInfo {
            address_mode: [SamplerAddressMode::MirroredRepeat; 3],
            ..SamplerCreateInfo::simple_repeat_linear()
        },
    )?;

    let sampler_nn = Sampler::new(
        &vk.device,
        &SamplerCreateInfo { address_mode: [SamplerAddressMode::MirroredRepeat; 3], ..SamplerCreateInfo::new() },
    )?;

    let imgviews_sampled = [&imgview_scratch, &imgview_patch_pyr_scratch, &imgview_patch_pyr]
        .into_iter()
        .cloned()
        .map(|imgview| DescriptorImageInfo {
            sampler: None,
            image_view: Some(imgview),
            image_layout: ImageLayout::General,
        })
        .collect_vec();
    let imgviews_storage = [Some(imgview_scratch), Some(imgview_patch_pyr_scratch)]
        .into_iter()
        .chain(imgviews_patch_pyr_mips.into_iter().map(Some));

    WriteDescriptorSet::image(
        0,
        DescriptorImageInfo {
            sampler: None,
            image_view: Some(imgview_coarse.clone()),
            image_layout: ImageLayout::General,
        },
    );
    let descriptor_set = DescriptorSet::new(
        vk.descriptor_set_allocator.clone(),
        pipelines.descriptor_set_layout.clone(),
        [
            WriteDescriptorSet::image_view(0, imgview_coarse.clone()),
            WriteDescriptorSet::image(
                1,
                DescriptorImageInfo {
                    sampler: None,
                    image_view: Some(imgview_coarse.clone()),
                    image_layout: ImageLayout::General,
                },
            ),
            WriteDescriptorSet::image_view_array(2, 0, imgviews_storage.clone()),
            WriteDescriptorSet::image_array(3, 0, imgviews_sampled.clone()),
            WriteDescriptorSet::sampler_array(4, 0, [Some(sampler_bil.clone()), Some(sampler_nn.clone())]),
        ],
        [],
    )?;

    Ok(PhysicalResources {
        ids: BufferIds {
            buf_staging_write,
            buf_staging_read,
            buf_staging_debug,
            img_patch_pyr: img_patch_pyr_id,
            img_patch_pyr_tmp: img_patch_pyr_scratch_id,
            buf_constants,
            buf_extremum_locations,
            buf_keypoints: buf_keypoint_indices,
            buf_patch_gradients_or_raw_descriptors,
            buf_embeddings_or_descriptors,
            img_coarse: img_coarse_id,
            buf_filtered_extrema,
            img_scratch: img_scratch_id,
        },
        descriptor_set,
    })
}

fn build_detect_taskgraph(
    taskgraph: &mut TaskGraph<GlobalContext>,
    virtual_ids: &BufferIds,
    params: &FixedParams,
    buffer_layouts: &BufferLayouts,
    pipelines: &ComputePipelines,
) -> Result<NodeId, crate::vulkan::VulkanError> {
    use tasks_detect::*;

    let upload_node_id = taskgraph
        .create_task_node(
            "upload",
            QueueFamilyType::Transfer,
            UploadImageTask { dst_buffer: virtual_ids.buf_staging_write },
        )
        .build();
    taskgraph.add_host_buffer_access(virtual_ids.buf_staging_write, HostAccessType::Write);

    let copy_buffer_image_node_id = taskgraph
        .create_task_node(
            "copy image",
            QueueFamilyType::Compute,
            CopyInputImageTask { buffer: virtual_ids.buf_staging_write, image: virtual_ids.img_coarse },
        )
        .buffer_access(virtual_ids.buf_staging_write, AccessTypes::COPY_TRANSFER_READ)
        .image_access(virtual_ids.img_coarse, AccessTypes::COPY_TRANSFER_WRITE, ImageLayoutType::General)
        .build();
    taskgraph.add_edge(upload_node_id, copy_buffer_image_node_id).unwrap();

    let buffers_to_zero = Vec::from([
        (
            virtual_ids.buf_extremum_locations,
            (buffer_layouts.extremum_locations.offset_n_extrema + size_of::<u32>()) as u64,
        ),
        (virtual_ids.buf_keypoints, (buffer_layouts.keypoints.offset_extremum_indices + size_of::<u32>()) as u64),
    ]);
    let zero_buffers_node_id = {
        let mut t = taskgraph.create_task_node(
            "setup",
            QueueFamilyType::Compute,
            ZeroBuffersTask { zero_buffers: buffers_to_zero.clone() },
        );
        buffers_to_zero.into_iter().for_each(|(buf, _size)| {
            t.buffer_access(buf, AccessTypes::COPY_TRANSFER_WRITE);
        });
        t.build()
    };

    let horz_blur_node_id = taskgraph
        .create_task_node(
            "blur horizontal",
            QueueFamilyType::Compute,
            BlurTask {
                pipeline: pipelines.blur.clone(),
                direction: BlurDirection::Horizontal,
                do_bind_pipeline: true,
                in_out_level: 0,
            },
        )
        .image_access(virtual_ids.img_coarse, AccessTypes::COMPUTE_SHADER_SAMPLED_READ, ImageLayoutType::General)
        .image_access(virtual_ids.img_scratch, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE, ImageLayoutType::General)
        .build();

    let vert_blur_node_id = taskgraph
        .create_task_node(
            "blur vertical",
            QueueFamilyType::Compute,
            BlurTask {
                pipeline: pipelines.blur.clone(),
                direction: BlurDirection::Vertical,
                do_bind_pipeline: false,
                in_out_level: 0,
            },
        )
        .image_access(virtual_ids.img_scratch, AccessTypes::COMPUTE_SHADER_SAMPLED_READ, ImageLayoutType::General)
        .image_access(virtual_ids.img_coarse, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE, ImageLayoutType::General)
        .build();

    taskgraph.add_edge(copy_buffer_image_node_id, horz_blur_node_id).unwrap();
    taskgraph.add_edge(horz_blur_node_id, vert_blur_node_id).unwrap();
    taskgraph.add_edge(zero_buffers_node_id, vert_blur_node_id).unwrap();

    let node_before_swt = vert_blur_node_id;

    let n_coarse_levels = params.n_scales + 3;

    // There are some weird things here which are different from the Ghahremani et al. paper.
    // Normally, the first blurred image (with sigma=0.6) would just be the input to the first SWT
    // convolution with the 14641 filter. But (by accident, initially) we're keeping it as the 0th
    // scale in the gaussian scale space.
    // It all ends up working pretty well, you get a ton more small blobs, which might not be
    // useful for every task, but MKD does a very good job matching them up, so why not keep them?
    let swt_blur_passes = (0..n_coarse_levels - 1)
        .map(|in_scale| {
            let mut horz_node = if in_scale == 0 {
                taskgraph.create_task_node(
                    format!("swtdense-{in_scale}-horz"),
                    QueueFamilyType::Compute,
                    BlurTask {
                        pipeline: pipelines.swt_dense.clone(),
                        direction: BlurDirection::Horizontal,
                        do_bind_pipeline: true,
                        in_out_level: 0,
                    },
                )
            } else {
                taskgraph.create_task_node(
                    format!("swt-{in_scale}-horz"),
                    QueueFamilyType::Compute,
                    SWTTask {
                        pipeline: pipelines.swt.clone(),
                        input_level: in_scale,
                        direction: BlurDirection::Horizontal,
                        do_bind_pipeline: in_scale == 1,
                        pipeline_sparse1: pipelines.swt_sparse1.clone(),
                        pipeline_sparse2: pipelines.swt_sparse2.clone(),
                    },
                )
            };
            let horizontal = horz_node
                .image_access(
                    virtual_ids.img_coarse,
                    AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                    ImageLayoutType::General,
                )
                .image_access(
                    virtual_ids.img_scratch,
                    AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                    ImageLayoutType::General,
                )
                .build();

            let mut vert_node = if in_scale == 0 {
                taskgraph.create_task_node(
                    format!("swtdense-{in_scale}-vert"),
                    QueueFamilyType::Compute,
                    BlurTask {
                        pipeline: pipelines.swt_dense.clone(),
                        direction: BlurDirection::Vertical,
                        do_bind_pipeline: false,
                        // cv above: here we're not overwriting level 0
                        in_out_level: 1,
                    },
                )
            } else {
                taskgraph.create_task_node(
                    format!("swt-{in_scale}-vert"),
                    QueueFamilyType::Compute,
                    SWTTask {
                        pipeline: pipelines.swt.clone(),
                        input_level: in_scale,
                        direction: BlurDirection::Vertical,
                        do_bind_pipeline: false,
                        pipeline_sparse1: pipelines.swt_sparse1.clone(),
                        pipeline_sparse2: pipelines.swt_sparse2.clone(),
                    },
                )
            };
            let vertical = vert_node
                .image_access(
                    virtual_ids.img_scratch,
                    AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                    ImageLayoutType::General,
                )
                .image_access(
                    virtual_ids.img_coarse,
                    AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                    ImageLayoutType::General,
                )
                .build();
            taskgraph.add_edge(horizontal, vertical).unwrap();
            (horizontal, vertical)
        })
        .collect_vec();

    let scales_available = swt_blur_passes.iter().map(|(_horz, vert)| vert).collect_vec();

    taskgraph.add_edge(node_before_swt, swt_blur_passes.first().unwrap().0).unwrap();
    // Connect previous vertical pass to this horizontal
    for ((_, vert1), (horz2, _)) in swt_blur_passes.iter().tuple_windows() {
        taskgraph.add_edge(*vert1, *horz2).unwrap();
    }
    let last_swt_node = swt_blur_passes.last().unwrap().1;

    // Since the initial blur level is kept and not overwritten, this first pyramid layer here
    // is just an unncecessary copy.
    let copy_scale_0_to_patch_pyr = taskgraph
        .create_task_node(
            "copy coarse image 0",
            QueueFamilyType::Graphics,
            BlitCopyImageTask {
                vimg_src: virtual_ids.img_coarse,
                vimg_dst: virtual_ids.img_patch_pyr,
                src_array_layer: 0,
                dst_mip_level: 0,
                half_size: false,
            },
        )
        .image_access(virtual_ids.img_coarse, AccessTypes::BLIT_TRANSFER_READ, ImageLayoutType::General)
        .image_access(virtual_ids.img_patch_pyr, AccessTypes::BLIT_TRANSFER_WRITE, ImageLayoutType::General)
        .build();
    taskgraph.add_edge(*scales_available[0], copy_scale_0_to_patch_pyr).unwrap();

    let copy_scale_1_to_patch_pyr = taskgraph
        .create_task_node(
            "decimate coarse image layer 0",
            QueueFamilyType::Graphics,
            BlitCopyImageTask {
                vimg_src: virtual_ids.img_coarse,
                vimg_dst: virtual_ids.img_patch_pyr,
                // Just decimating level 0 (blurred with sigma=0.6) to pyramid level 1 at half
                // scale also works pretty well.
                src_array_layer: 1,
                dst_mip_level: 1,
                half_size: true,
            },
        )
        .image_access(virtual_ids.img_coarse, AccessTypes::BLIT_TRANSFER_READ, ImageLayoutType::General)
        .image_access(virtual_ids.img_patch_pyr, AccessTypes::BLIT_TRANSFER_WRITE, ImageLayoutType::General)
        .build();
    taskgraph.add_edge(*scales_available[1], copy_scale_1_to_patch_pyr).unwrap();

    let (pyr_start_node_id, pyr_end_node_id) = patch_pyramid::patch_pyramid_nodes(
        patch_pyramid::PatchPyramidArgs {
            n_levels: params.patch_pyr_levels,
            pyr_image_id: virtual_ids.img_patch_pyr,
            tmp_image_id: virtual_ids.img_patch_pyr_tmp,
        },
        pipelines.blur_pyramid.clone(),
        taskgraph,
    );
    // Wait until all blurring is done so we don't switch/rebind pipelines every time,
    // since the pyramid involves a blur compute stage
    taskgraph.add_edge(last_swt_node, pyr_start_node_id).unwrap();

    let scan_extrema_node_id = taskgraph
        .create_task_node(
            "scan extrema",
            QueueFamilyType::Compute,
            ScanExtremaTask {
                pipeline: pipelines.scan_extrema.clone(),
                n_fine_scales: n_coarse_levels - 1,
                vbuf_extremum_locations: virtual_ids.buf_extremum_locations,
            },
        )
        .image_access(virtual_ids.img_coarse, AccessTypes::COMPUTE_SHADER_SAMPLED_READ, ImageLayoutType::General)
        .image_access(virtual_ids.img_coarse, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE, ImageLayoutType::General)
        .buffer_access(virtual_ids.buf_extremum_locations, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_extremum_locations, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
        .build();

    // Wait for pyramid blur shaders
    taskgraph.add_edge(pyr_end_node_id, scan_extrema_node_id).unwrap();

    Ok(scan_extrema_node_id)
}

enum ChainExtractTg {
    Standalone,
    ConnectTo { connect_node: NodeId, use_host_filtered_extremum_indices: bool },
}

fn build_extract_taskgraph(
    taskgraph: &mut TaskGraph<GlobalContext>,
    chain: ChainExtractTg,
    virtual_ids: &BufferIds,
    params: &FixedParams,
    pipelines: &ComputePipelines,
) -> Result<NodeId, crate::vulkan::VulkanError> {
    use tasks_extract::*;

    let use_host_filtered_extremum_indices = match chain {
        ChainExtractTg::Standalone => true,
        ChainExtractTg::ConnectTo { use_host_filtered_extremum_indices, .. } => use_host_filtered_extremum_indices,
    };
    let kp_ori_node_id = taskgraph
        .create_task_node(
            "keypoint orientation",
            QueueFamilyType::Compute,
            KeypointOrientationTask {
                pipeline: pipelines.keypoint_orientation.clone(),
                vbuf_filtered_extrema: virtual_ids.buf_filtered_extrema,
                vbuf_extremum_locations: virtual_ids.buf_extremum_locations,
                vbuf_keypoint_indices: virtual_ids.buf_keypoints,
                use_host_filtered_extremum_idx: use_host_filtered_extremum_indices,
            },
        )
        .image_access(virtual_ids.img_patch_pyr, AccessTypes::COMPUTE_SHADER_SAMPLED_READ, ImageLayoutType::General)
        .buffer_access(virtual_ids.buf_filtered_extrema, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_extremum_locations, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_extremum_locations, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
        .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
        .build();
    if let ChainExtractTg::ConnectTo { connect_node, .. } = chain {
        taskgraph.add_edge(connect_node, kp_ori_node_id).unwrap();
    }

    let patch_gradients_node_id = {
        let gradients_node = taskgraph
            .create_task_node(
                "patches+gradients",
                QueueFamilyType::Compute,
                PatchBlurGradientTask { pipeline: pipelines.patch_gradients.clone(), buffers: virtual_ids.clone() },
            )
            .buffer_access(
                virtual_ids.buf_patch_gradients_or_raw_descriptors,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
            )
            .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
            .buffer_access(virtual_ids.buf_extremum_locations, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
            .buffer_access(
                virtual_ids.buf_patch_gradients_or_raw_descriptors,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
            )
            .image_access(virtual_ids.img_patch_pyr, AccessTypes::COMPUTE_SHADER_SAMPLED_READ, ImageLayoutType::General)
            .build();
        taskgraph.add_edge(kp_ori_node_id, gradients_node).unwrap();
        if params.debug_readout_patches {
            let copy_patches_node = taskgraph
                .create_task_node(
                    "Copy patches to staging",
                    QueueFamilyType::Transfer,
                    CopyBufferTask {
                        src: virtual_ids.buf_patch_gradients_or_raw_descriptors,
                        dst: virtual_ids.buf_staging_debug.expect("must be Some if debug patches enabled"),
                        size: u64::from(params.max_keypoints) * 32 * 32 * size_of::<f32>() as u64,
                        src_offset: 2 * u64::from(params.max_keypoints) * 32 * 32 * size_of::<f32>() as u64,
                    },
                )
                .buffer_access(virtual_ids.buf_patch_gradients_or_raw_descriptors, AccessTypes::COPY_TRANSFER_READ)
                .buffer_access(
                    virtual_ids.buf_staging_debug.expect("must be Some if debug patches enabled"),
                    AccessTypes::COPY_TRANSFER_WRITE,
                )
                .build();
            taskgraph.add_edge(gradients_node, copy_patches_node).unwrap();
            copy_patches_node
        } else {
            gradients_node
        }
    };

    let embedding_polar_node_id = taskgraph
        .create_task_node(
            "Spatial Encoding Polar",
            QueueFamilyType::Compute,
            EmbeddingTask { pipeline: pipelines.embedding_polar.clone(), buffers: virtual_ids.clone() },
        )
        .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_patch_gradients_or_raw_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_constants, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_embeddings_or_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
        .build();

    let embedding_cartesian_node_id = taskgraph
        .create_task_node(
            "Spatial Encoding Cartesian",
            QueueFamilyType::Compute,
            EmbeddingCartesianTask { pipeline: pipelines.embedding_cartesian.clone(), buffers: virtual_ids.clone() },
        )
        .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_patch_gradients_or_raw_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_constants, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_embeddings_or_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
        .build();

    taskgraph.add_edge(patch_gradients_node_id, embedding_polar_node_id).unwrap();

    taskgraph.add_edge(patch_gradients_node_id, embedding_cartesian_node_id).unwrap();

    let embedding_sum_node_id = taskgraph
        .create_task_node(
            "embedding sum",
            QueueFamilyType::Compute,
            FirstNormalizeTask { pipeline: pipelines.normalize.clone(), buffers: virtual_ids.clone() },
        )
        .buffer_access(virtual_ids.buf_embeddings_or_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_patch_gradients_or_raw_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
        .build();

    taskgraph.add_edge(embedding_polar_node_id, embedding_sum_node_id).unwrap();

    taskgraph.add_edge(embedding_cartesian_node_id, embedding_sum_node_id).unwrap();

    let whiten_node_id = taskgraph
        .create_task_node(
            "whitening",
            QueueFamilyType::Compute,
            WhiteningTask { pipeline: pipelines.whitening.clone(), buffers: virtual_ids.clone() },
        )
        .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_patch_gradients_or_raw_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_constants, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_embeddings_or_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
        .build();

    taskgraph.add_edge(embedding_sum_node_id, whiten_node_id).unwrap();

    let normalize_id = taskgraph
        .create_task_node(
            "normalize",
            QueueFamilyType::Compute,
            FinalNormalizeTask { pipeline: pipelines.normalize_final.clone(), buffers: virtual_ids.clone() },
        )
        .buffer_access(virtual_ids.buf_keypoints, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_constants, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_embeddings_or_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
        .buffer_access(virtual_ids.buf_embeddings_or_descriptors, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
        .build();

    taskgraph.add_edge(whiten_node_id, normalize_id).unwrap();

    Ok(normalize_id)
}

const fn size_of_return_value<F, T, U>(_f: &F) -> usize
where
    F: FnOnce(T) -> U,
{
    std::mem::size_of::<U>()
}

macro_rules! size_of_field {
    ($type:ty, $field:ident) => {
        size_of_return_value(&|s: $type| s.$field)
    };
}

fn upload_constant_data(
    params: &FixedParams,
    physical_resources: &PhysicalResources,
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
) -> Result<(), crate::vulkan::VulkanError> {
    let crate::mkd_ref::PCAModel { mean, eigvals, eigvecs } =
        crate::mkd_ref::PCAModel::from_safetensors(match params.pca {
            MKDPCA::LIBERTY => crate::mkd_ref::PCA_SAFETENSORS_LIBERTY,
            MKDPCA::NOTREDAME => crate::mkd_ref::PCA_SAFETENSORS_NOTREDAME,
            MKDPCA::YOSEMITE => crate::mkd_ref::PCA_SAFETENSORS_YOSEMITE,
        })
        .expect("TODO unhandled");
    let t = 0.7;
    let m = -0.5 * t;
    let out_dims = 128;
    let in_dims = 238;
    let eigvecs = eigvecs.slice(ndarray::s![.., ..out_dims]).to_owned();
    let eigvals = eigvals.slice(ndarray::s![..out_dims]);
    let eigvecs = eigvecs * eigvals.powf(m).broadcast((in_dims, out_dims)).unwrap();
    let eigvecs_t = eigvecs.slice(ndarray::s![.., ..128]).t().to_owned();
    let eigvecs_t = eigvecs_t.as_standard_layout().to_owned();

    let embedding_polar = crate::mkd_ref::spatial_embedding_polar_nokronecker();
    let grad_angles = crate::mkd_ref::cart2pol(&crate::mkd_ref::mesh_grid().view()).slice_move(ndarray::s![1, .., ..]);

    let upload_to_buffer = match params.use_staging_buffers {
        StagingBuffers::Yes => physical_resources.ids.buf_staging_write,
        StagingBuffers::No => physical_resources.ids.buf_extremum_locations,
    };
    let host_buffer_accesses = [(upload_to_buffer, HostAccessType::Write)];
    let buffer_accesses = [
        (upload_to_buffer, AccessTypes::COPY_TRANSFER_WRITE),
        (upload_to_buffer, AccessTypes::COPY_TRANSFER_READ),
        (physical_resources.ids.buf_constants, AccessTypes::COPY_TRANSFER_WRITE),
    ];
    let image_accesses = [];

    use shaders::shaders_f32::ConstantData;
    let offset_gradient_angle = std::mem::offset_of!(ConstantData, gradient_angle);
    let size_gradient_angle = size_of_field!(ConstantData, gradient_angle);
    let offset_embedding_polar = std::mem::offset_of!(ConstantData, embedding_polar);
    let size_embedding_polar = size_of_field!(ConstantData, embedding_polar);
    let offset_mean_vec = std::mem::offset_of!(ConstantData, mean_vec);
    let size_mean_vec = size_of_field!(ConstantData, mean_vec);
    let offset_eigen_vecs = std::mem::offset_of!(ConstantData, eigen_vecs);
    let size_eigen_vecs = size_of_field!(ConstantData, eigen_vecs);

    unsafe {
        vulkano_taskgraph::execute(
            queue,
            resources,
            flight_id,
            |cbf: &mut RecordingCommandBuffer<'_>, tcx: &mut TaskContext<'_>| {
                let buf: &mut [u8] = tcx.write_buffer::<[u8]>(upload_to_buffer, 0..size_of::<ConstantData>() as u64)?;

                for (offset, size, slice) in [
                    (offset_gradient_angle, size_gradient_angle, grad_angles.as_slice().unwrap()),
                    (offset_embedding_polar, size_embedding_polar, embedding_polar.as_slice().unwrap()),
                    (offset_mean_vec, size_mean_vec, mean.as_slice().unwrap()),
                    (offset_eigen_vecs, size_eigen_vecs, eigvecs_t.as_slice().unwrap()),
                ] {
                    buf[offset..offset + size].copy_from_slice(bytemuck::cast_slice(slice));
                }

                if params.use_staging_buffers == StagingBuffers::Yes {
                    cbf.pipeline_barrier(&DependencyInfo {
                        buffer_memory_barriers: &[BufferMemoryBarrier {
                            buffer: upload_to_buffer,
                            src_stages: PipelineStages::HOST,
                            src_access: AccessFlags::HOST_WRITE,
                            dst_stages: PipelineStages::ALL_TRANSFER,
                            dst_access: AccessFlags::TRANSFER_WRITE,
                            size: buf.len() as u64,
                            ..BufferMemoryBarrier::new()
                        }],
                        ..Default::default()
                    })?;
                    cbf.copy_buffer(&CopyBufferInfo {
                        src_buffer: upload_to_buffer,
                        dst_buffer: physical_resources.ids.buf_constants,
                        ..Default::default()
                    })?;
                }

                Ok(())
            },
            host_buffer_accesses,
            buffer_accesses,
            image_accesses,
        )
        .expect("TODO");
        resources.flight(flight_id).unwrap().wait(None)?;
    }
    Ok(())
}

fn resource_map<'a, W>(
    taskgraph: &'a ExecutableTaskGraph<W>,
    virt: &BufferIds,
    phys_buffers: &BufferIds,
) -> ResourceMap<'a> {
    let mut rm = resource_map!(taskgraph,
        virt.buf_constants => phys_buffers.buf_constants,
        virt.buf_staging_write => phys_buffers.buf_staging_write,
        virt.buf_staging_read => phys_buffers.buf_staging_read,
        virt.buf_extremum_locations => phys_buffers.buf_extremum_locations,
        virt.buf_filtered_extrema => phys_buffers.buf_filtered_extrema,
        virt.buf_keypoints => phys_buffers.buf_keypoints,
        virt.buf_patch_gradients_or_raw_descriptors => phys_buffers.buf_patch_gradients_or_raw_descriptors,
        virt.buf_embeddings_or_descriptors => phys_buffers.buf_embeddings_or_descriptors,

        virt.img_coarse => phys_buffers.img_coarse,
        virt.img_scratch => phys_buffers.img_scratch,
        virt.img_patch_pyr => phys_buffers.img_patch_pyr,
        virt.img_patch_pyr_tmp => phys_buffers.img_patch_pyr_tmp,
    )
    .expect("TODO");
    if let Some(phys_debug) = phys_buffers.buf_staging_debug {
        rm.insert_buffer(virt.buf_staging_debug.expect("must be Some"), phys_debug).expect("TODO");
    }
    rm
}

fn make_virtual_ids<W>(taskgraph: &mut TaskGraph<W>, params: &FixedParams) -> BufferIds {
    BufferIds {
        buf_staging_write: taskgraph.add_buffer(&BufferCreateInfo::default()),
        buf_staging_read: taskgraph.add_buffer(&BufferCreateInfo::default()),
        buf_constants: taskgraph.add_buffer(&BufferCreateInfo::default()),
        buf_patch_gradients_or_raw_descriptors: taskgraph.add_buffer(&BufferCreateInfo::default()),
        buf_embeddings_or_descriptors: taskgraph.add_buffer(&BufferCreateInfo::default()),
        buf_extremum_locations: taskgraph.add_buffer(&BufferCreateInfo::default()),
        buf_keypoints: taskgraph.add_buffer(&BufferCreateInfo::default()),
        buf_filtered_extrema: taskgraph.add_buffer(&BufferCreateInfo::default()),
        buf_staging_debug: params.debug_readout_patches.then(|| taskgraph.add_buffer(&BufferCreateInfo::default())),
        img_coarse: taskgraph.add_image(&ImageCreateInfo::default()),
        img_scratch: taskgraph.add_image(&ImageCreateInfo::default()),
        img_patch_pyr: taskgraph.add_image(&ImageCreateInfo::default()),
        img_patch_pyr_tmp: taskgraph.add_image(&ImageCreateInfo::default()),
    }
}

struct TopKContrastFilter {
    pub n: u32,
}

impl FilterBlobs for TopKContrastFilter {
    fn filter<'r, 'w>(
        &mut self,
        blobs: Box<dyn Iterator<Item = BlobLocationsView<'r>> + 'w>,
        out: FilterBlobsOutput<'w>,
    ) {
        let contrasts: Vec<FloatOrd<f32>> =
            blobs.flat_map(|chunk| chunk.contrasts).map(|contrast| FloatOrd(-(contrast.abs()))).collect_vec();

        let n_blobs = contrasts.len();
        if n_blobs <= self.n as usize {
            out.indices.extend(0..n_blobs as u32);
        } else {
            let mut contrasts2 = contrasts.clone();
            let cutoff = order_stat::kth(&mut contrasts2, self.n as usize).0.abs();
            for (i, contrast) in contrasts.into_iter().enumerate() {
                if contrast.0.abs() >= cutoff {
                    out.indices.push(i as u32);
                }
                if out.indices.len() == self.n as usize {
                    break;
                }
            }
        }
    }
}

#[test]
fn detect_nofiltering_same_as_all_noop_filter() {
    use image::buffer::ConvertBuffer;
    use nshare::IntoNdarray2;

    struct NoOpFilter;
    impl FilterBlobs for NoOpFilter {
        fn filter<'r, 'w>(
            &mut self,
            blobs: Box<dyn Iterator<Item = BlobLocationsView<'r>> + 'w>,
            out: FilterBlobsOutput<'w>,
        ) {
            let n: usize = blobs.map(|v| v.xs.len()).sum();
            out.indices.extend(0..n as u32);
        }
    }

    let vk = make_a_vulkan::Vulkan::new().unwrap();
    let img = match image::open("../sample_data/bird.jpg").unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => {
            panic!("wrong image type");
        }
    };
    let img_f32: image::ImageBuffer<image::Luma<f32>, Vec<f32>> = img.convert();
    let img_f32 = img_f32.into_ndarray2();

    let mut lf = LocalFeaturesVulkan::new(
        BuildTimeParams {
            n_scales: 5,
            max_image_width: img.width(),
            max_image_height: img.height(),
            max_features: 5000,
            max_blobs: 10000,
            ..Default::default()
        },
        vk,
    )
    .unwrap();

    let result_nofilter = lf.detect_extract_all(&img_f32.view(), &Default::default()).unwrap();
    let result_filter = lf.detect(&img_f32.view(), &Default::default(), Some(&mut NoOpFilter)).unwrap();
    assert!(result_nofilter.keypoints.len() > 0);
    assert_eq!(result_filter.keypoints.len(), result_nofilter.keypoints.len());

    let sort_result = |mut res: FeaturesResult| {
        let mut indices = (0..res.keypoints.len()).collect_vec();
        indices.sort_by(|a, b| {
            let kpa = &res.keypoints[*a];
            let kpb = &res.keypoints[*b];
            kpa.x
                .total_cmp(&kpb.x)
                .then(kpa.y.total_cmp(&kpb.y))
                .then(kpa.angle.total_cmp(&kpb.angle))
                .then(kpa.size.total_cmp(&kpb.size))
                .then(kpa.response.total_cmp(&kpb.response))
        });
        res.keypoints = indices.iter().map(|i| res.keypoints[*i].clone()).collect_vec();
        res.descriptors = res.descriptors.select(ndarray::Axis(0), &indices);
        res
    };
    let result_nofilter_sorted = sort_result(result_nofilter.clone());
    assert_eq!(result_nofilter_sorted.keypoints.len(), result_nofilter.keypoints.len());
    let result_filter_sorted = sort_result(result_filter.clone());
    assert_eq!(result_filter_sorted.keypoints.len(), result_filter.keypoints.len());
    assert_eq!(result_filter_sorted, result_nofilter_sorted);
}

#[test]
// this isn't a great test, more of a sanity check if things are really wrong.
fn mkd_matches_ref() {
    use crate::mkd_ref;
    use image::buffer::ConvertBuffer;
    use ndarray::s;
    use nshare::IntoNdarray2;
    let vk = make_a_vulkan::Vulkan::new().unwrap();
    let img = match image::open("../sample_data/bird.jpg").unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => {
            panic!("wrong image type");
        }
    };
    let img_f32: image::ImageBuffer<image::Luma<f32>, Vec<f32>> = img.convert();
    let img_f32 = img_f32.into_ndarray2();

    let mut lf = LocalFeaturesVulkan::new(
        BuildTimeParams {
            n_scales: 6,
            max_image_width: img.width(),
            max_image_height: img.height(),
            max_features: 5000,
            max_blobs: 10000,
            debug_readout_patches: true,
            ..Default::default()
        },
        vk,
    )
    .unwrap();
    let FeaturesResult { descriptors, .. } = lf.detect_extract_all(&img_f32.view(), &Default::default()).unwrap();
    let patches = lf.debug_read_patches().unwrap().slice_move(s![..descriptors.shape()[0], .., ..]);

    let mkd = crate::mkd_ref::Mkd::new(&mkd_ref::MkdOptions { whitening: mkd_ref::Whitening::PcaAttenutated });
    #[derive(Debug)]
    #[allow(unused)]
    struct FailedCase {
        index: usize,
        l2: f32,
        max_diff: f32,
    }
    let mut failures = Vec::new();
    for (i, (patch, actual)) in patches.outer_iter().zip_eq(descriptors.outer_iter()).enumerate() {
        let expected = mkd.patch(&patch);
        let diff = expected.clone() - actual;
        let max_diff = diff.abs().iter().copied().max_by(f32::total_cmp).unwrap();
        let l2 = diff.pow2().sum().sqrt();
        if l2 >= 1e-3 || max_diff >= 1e-3 {
            failures.push(FailedCase { index: i, l2, max_diff });
        }
    }
    assert_eq!(failures.len(), 0, "Failures:\n{:?}", failures);
}
