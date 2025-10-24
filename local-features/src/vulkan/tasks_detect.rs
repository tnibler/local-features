use std::sync::Arc;

use log::trace;
use vulkano::{
    buffer::Buffer,
    image::{Image, ImageAspects},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{BufferImageCopy, RecordingCommandBuffer},
    resource::ImageLayoutType,
};

use super::{BlurDirection, GlobalContext, shaders};

pub(super) struct UploadImageTask {
    pub dst_buffer: Id<Buffer>,
}

impl Task for UploadImageTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        _cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        // SAFETY: pointer input_image is live for as long as the flight executes.
        unsafe {
            // NOTE: this must be taken care of if ever multiple flights are allowed: all fields in
            // Worlds become arrays indexed by tcx.current_frame_index, and each pointer must be
            // valid for the duration of its corresponding flight

            tcx.write_buffer::<[f32]>(self.dst_buffer, ..(world.input_image.len() * size_of::<f32>()) as u64)?
                .copy_from_slice(world.input_image.as_ref().expect("input image must be valid pointer"));
        }
        Ok(())
    }
}

pub(super) struct CopyInputImageTask {
    pub buffer: Id<Buffer>,
    pub image: Id<Image>,
}

impl Task for CopyInputImageTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let image_width = world.image_width;
        let image_height = world.image_height;
        unsafe {
            cbf.copy_buffer_to_image(&vulkano_taskgraph::command_buffer::CopyBufferToImageInfo {
                src_buffer: self.buffer,
                dst_image: self.image,
                dst_image_layout: ImageLayoutType::General,
                regions: &[BufferImageCopy {
                    image_subresource: vulkano::image::ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        ..Default::default()
                    },
                    image_extent: [image_width, image_height, 1],
                    ..Default::default()
                }],
                ..Default::default()
            })?;
        }
        Ok(())
    }
}

pub(super) struct ZeroBuffersTask {
    pub zero_buffers: Vec<(Id<Buffer>, u64)>,
}

impl Task for ZeroBuffersTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        unsafe {
            for (buf, size) in &self.zero_buffers {
                cbf.fill_buffer(&vulkano_taskgraph::command_buffer::FillBufferInfo {
                    dst_buffer: *buf,
                    data: 0,
                    size: *size,
                    ..Default::default()
                })?;
            }
        }
        Ok(())
    }
}

pub(super) struct BlurTask {
    pub direction: BlurDirection,
    pub do_bind_pipeline: bool,
    pub pipeline: Arc<ComputePipeline>,
    pub in_out_level: u32,
}

impl Task for BlurTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        unsafe {
            cbf.as_raw().bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout(),
                0,
                &[world.physical_resources.descriptor_set.as_raw()],
                &[],
            )?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::ConvPc {
                    in_level: self.in_out_level,
                    width: world.image_width,
                    height: world.image_height,
                    vertical_pass: match self.direction {
                        BlurDirection::Vertical => 1,
                        _ => 0,
                    },
                },
            )?;
        }
        if self.do_bind_pipeline {
            unsafe {
                cbf.bind_pipeline_compute(&self.pipeline)?;
            }
        }
        let width = world.image_width;
        let height = world.image_height;
        let wg_size_x = 64;
        let wg_size_y = 2;
        let wg_count = match self.direction {
            BlurDirection::Horizontal => [width.div_ceil(wg_size_x), height.div_ceil(wg_size_y), 1],
            BlurDirection::Vertical => [height.div_ceil(wg_size_x), width.div_ceil(wg_size_y), 1],
        };
        trace!("Dense blur: {:?}, in/out_level={}, dispatch={:?}", self.direction, self.in_out_level, wg_count);
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct SWTTask {
    pub input_level: u32,
    pub direction: BlurDirection,
    pub do_bind_pipeline: bool,
    pub pipeline: Arc<ComputePipeline>,
    pub pipeline_sparse1: Arc<ComputePipeline>,
    pub pipeline_sparse2: Arc<ComputePipeline>,
}

impl Task for SWTTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer<'_>,
        _tcx: &mut vulkano_taskgraph::TaskContext<'_>,
        world: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        unsafe {
            match self.input_level {
                0 => {
                    panic!("wrong pipeline, Blur is the one");
                }
                1 => {
                    cbf.bind_pipeline_compute(&self.pipeline_sparse1)?;
                }
                2.. => {
                    cbf.bind_pipeline_compute(&self.pipeline_sparse2)?;
                }
            }

            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::ConvPc {
                    vertical_pass: match self.direction {
                        BlurDirection::Horizontal => 0,
                        BlurDirection::Vertical => 1,
                    },
                    in_level: self.input_level,
                    width: world.image_width,
                    height: world.image_height,
                },
            )?;
        }
        let width = world.image_width;
        let height = world.image_height;
        let wg_count = match self.input_level {
            0 => panic!("wrong"),
            1 => {
                let wg_cover_x = 64;
                let wg_cover_y = 2;
                match self.direction {
                    BlurDirection::Horizontal => [width.div_ceil(wg_cover_x), height.div_ceil(wg_cover_y), 1],
                    BlurDirection::Vertical => [height.div_ceil(wg_cover_x), width.div_ceil(wg_cover_y), 1],
                }
            }
            2.. => {
                let (wg_cover_x, wg_cover_y) = (32, 8);
                let wg_per_row = width.div_ceil(wg_cover_x);
                let wg_per_col = height.div_ceil(wg_cover_y);
                [wg_per_row, wg_per_col, 1]
            }
        };
        trace!("SWT blur: {:?}, in_level={}, dispatch {:?}", self.direction, self.input_level, wg_count);
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct ScanExtremaTask {
    pub n_fine_scales: u32,
    pub vbuf_extremum_locations: Id<Buffer>,
    pub pipeline: Arc<ComputePipeline>,
}

impl Task for ScanExtremaTask {
    type World = GlobalContext;
    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let addr = |id: Id<Buffer>| Ok::<_, vulkano_taskgraph::TaskError>(tcx.buffer(id)?.buffer().device_address()?);
        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::ScanExtremaPc {
                    extremum_locations: addr(self.vbuf_extremum_locations)?.into(),
                    width: world.image_width,
                    height: world.image_height,
                    n_fine_levels: self.n_fine_scales,
                    border: world.border,
                    contrast_threshold: world.contrast_threshold,
                    min_scale: world.extremum_min_scale,
                    edgeness_cm_low: world.edgeness_cm_low,
                    edgeness_cm_high: world.edgeness_cm_high,
                    // TODO: unused. If min_scale implies that all extrema below a certain layer
                    // get filtered, no need to even process it.
                    skip_layers: world.extremum_skip_layers,
                    max_extrema: world.rt_max_extrema,
                },
            )?;
        }
        let wg_count = {
            let wg_cover_x = (6 + 1) * 2;
            let wg_cover_y = (6 + 1) * 2;
            // scan from start_layer to n_fine_scale - 1 (inclusive),
            [
                (world.image_width - 2 * world.border).div_ceil(wg_cover_x),
                (world.image_height - 2 * world.border).div_ceil(wg_cover_y),
                1,
            ]
        };
        trace!("DoG Extrema: dispatch {:?}", wg_count);
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}
