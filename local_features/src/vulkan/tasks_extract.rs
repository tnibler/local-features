use std::sync::Arc;

use log::trace;
use vulkano::{
    buffer::Buffer,
    pipeline::{ComputePipeline, Pipeline},
};
use vulkano_taskgraph::{
    command_buffer::{BufferCopy, CopyBufferInfo, RecordingCommandBuffer},
    Id, Task, TaskContext, TaskResult,
};

use crate::{DESCRIPTOR_LEN, DIMS_INPUT};

use super::{shaders, BufferIds, GlobalContext};

pub(super) struct KeypointOrientationTask {
    pub vbuf_extremum_locations: Id<Buffer>,
    pub vbuf_filtered_extrema: Id<Buffer>,
    pub vbuf_keypoint_indices: Id<Buffer>,
    pub pipeline: Arc<ComputePipeline>,
}

impl Task for KeypointOrientationTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let extremum_locations = tcx
            .buffer(self.vbuf_extremum_locations)?
            .buffer()
            .device_address()?
            .get();
        let filtered_extrema = tcx
            .buffer(self.vbuf_filtered_extrema)?
            .buffer()
            .device_address()?
            .get();
        let keypoints = tcx
            .buffer(self.vbuf_keypoint_indices)?
            .buffer()
            .device_address()?
            .get();

        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::KeypointOrientationPc {
                    extremum_locations,
                    keypoints,
                    filtered_extrema,
                    coarse_image_id: world.physical_resources.stor_coarse,
                    height: world.image_height,
                    width: world.image_width,
                    rt_max_keypoints: world.rt_max_keypoints,
                },
            )?;
            let wg_count = [world.n_filtered_extrema, 1, 1];
            trace!("Dispatch keypoint_orientation: {wg_count:?}");
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct PatchBlurGradientTask {
    pub buffers: BufferIds,
    pub pipeline: Arc<ComputePipeline>,
}

impl Task for PatchBlurGradientTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer<'_>,
        tcx: &mut vulkano_taskgraph::TaskContext<'_>,
        world: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        let patch_gradients_addr = tcx
            .buffer(self.buffers.buf_patch_gradients_or_raw_descriptors)?
            .buffer()
            .device_address()?
            .get();
        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::PatchGradientsPc {
                    image_width: world.image_width,
                    image_height: world.image_height,
                    patch_scale_factor: world.patch_scale_factor,
                    pyramid_texture_id: world.physical_resources.samp_patch_pyr,
                    pyramid_sampler_id: world.physical_resources.sampler,
                    extremum_locations: tcx
                        .buffer(self.buffers.buf_extremum_locations)?
                        .buffer()
                        .device_address()?
                        .into(),
                    keypoints: tcx
                        .buffer(self.buffers.buf_keypoints)?
                        .buffer()
                        .device_address()?
                        .into(),
                    patch_gradients: patch_gradients_addr,
                },
            )?;
            let wg_count = [world.rt_max_keypoints, 1, 1];
            trace!("Dispatch patch_gradient: {wg_count:?}");
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}
pub(super) struct EmbeddingTask {
    pub buffers: BufferIds,
    pub pipeline: Arc<ComputePipeline>,
}

impl Task for EmbeddingTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let keypoints_addr = tcx
            .buffer(self.buffers.buf_keypoints)?
            .buffer()
            .device_address()?
            .get();
        let patch_gradients_addr = tcx
            .buffer(self.buffers.buf_patch_gradients_or_raw_descriptors)?
            .buffer()
            .device_address()?
            .get();
        let embeddings_addr = tcx
            .buffer(self.buffers.buf_embeddings_or_descriptors)?
            .buffer()
            .device_address()?
            .get();
        let consts_addr = tcx
            .buffer(self.buffers.buf_constants)?
            .buffer()
            .device_address()?
            .get();

        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::EmbedPc {
                    keypoints: keypoints_addr,
                    patch_gradients: patch_gradients_addr,
                    consts: consts_addr,
                    embeddings: embeddings_addr,
                },
            )?;
        }
        let wg_count = [world.rt_max_keypoints, DIMS_INPUT as u32, 1];
        trace!("Dispatch embedding: {wg_count:?}");
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct FirstNormalizeTask {
    pub buffers: BufferIds,
    pub pipeline: Arc<ComputePipeline>,
}

impl Task for FirstNormalizeTask {
    type World = GlobalContext;
    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let keypoints_addr = tcx
            .buffer(self.buffers.buf_keypoints)?
            .buffer()
            .device_address()?
            .get();
        let embeddings_addr = tcx
            .buffer(self.buffers.buf_embeddings_or_descriptors)?
            .buffer()
            .device_address()?
            .get();
        let raw_descriptors_addr = tcx
            .buffer(self.buffers.buf_patch_gradients_or_raw_descriptors)?
            .buffer()
            .device_address()?
            .get();

        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::EmbeddingSumPc {
                    keypoints: keypoints_addr,
                    raw_descriptors: raw_descriptors_addr,
                    embeddings: embeddings_addr,
                },
            )?;
        }
        let wg_count = [world.rt_max_keypoints, 1, 1];
        unsafe {
            trace!("Dispatch normalize: {wg_count:?}");
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct WhiteningTask {
    pub buffers: BufferIds,
    pub pipeline: Arc<ComputePipeline>,
}

impl Task for WhiteningTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let rows_per_wg = 32;
        let rows_out = 128;

        let keypoints_addr = tcx
            .buffer(self.buffers.buf_keypoints)?
            .buffer()
            .device_address()?
            .get();
        let raw_descriptors_addr = tcx
            .buffer(self.buffers.buf_patch_gradients_or_raw_descriptors)?
            .buffer()
            .device_address()?
            .get();
        let descriptors_addr = tcx
            .buffer(self.buffers.buf_embeddings_or_descriptors)?
            .buffer()
            .device_address()?
            .get();
        let consts_addr = tcx
            .buffer(self.buffers.buf_constants)?
            .buffer()
            .device_address()?
            .get();

        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::WhiteningPc {
                    consts: consts_addr,
                    keypoints: keypoints_addr,
                    raw_descriptors: raw_descriptors_addr,
                    descriptors: descriptors_addr,
                },
            )?;
        }
        let wg_count = [world.rt_max_keypoints, rows_out / rows_per_wg, 1];
        unsafe {
            trace!("Dispatch whitening: {wg_count:?}");
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct FinalNormalizeTask {
    pub buffers: BufferIds,
    pub pipeline: Arc<ComputePipeline>,
}

impl Task for FinalNormalizeTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let keypoints_addr = tcx
            .buffer(self.buffers.buf_keypoints)?
            .buffer()
            .device_address()?
            .get();

        let descriptors_addr = tcx
            .buffer(self.buffers.buf_embeddings_or_descriptors)?
            .buffer()
            .device_address()?
            .get();

        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::L2NormalizePc {
                    keypoints: keypoints_addr,
                    descriptors: descriptors_addr,
                },
            )?;
        }
        let wg_count = [world.rt_max_keypoints, 1, 1];
        trace!("Dispatch normalize_final: {wg_count:?}");
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct DescriptorAndKeypointCopyTask {
    pub buffers: BufferIds,
}

impl Task for DescriptorAndKeypointCopyTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let keypoints_copy_size = world.buffer_layouts.keypoints.size_total;
        let descriptor_copy_size =
            u64::from(world.rt_max_keypoints) * DESCRIPTOR_LEN as u64 * size_of::<f32>() as u64;
        trace!("Copy keypoints and descriptors to staging");
        unsafe {
            cbf.copy_buffer(&CopyBufferInfo {
                src_buffer: self.buffers.buf_keypoints,
                dst_buffer: self.buffers.buf_staging,
                regions: &[BufferCopy {
                    size: keypoints_copy_size,
                    ..Default::default()
                }],
                ..Default::default()
            })?;
            cbf.copy_buffer(&CopyBufferInfo {
                src_buffer: self.buffers.buf_embeddings_or_descriptors,
                dst_buffer: self.buffers.buf_staging,
                regions: &[BufferCopy {
                    size: descriptor_copy_size,
                    dst_offset: keypoints_copy_size,
                    ..Default::default()
                }],
                ..Default::default()
            })?;
        }
        Ok(())
    }
}
