use std::sync::Arc;

use vulkano::{
    buffer::Buffer,
    image::{Image, ImageAspects},
    pipeline::{ComputePipeline, Pipeline},
};
use vulkano_taskgraph::{
    command_buffer::{BufferImageCopy, RecordingCommandBuffer},
    resource::ImageLayoutType,
    Id, Task, TaskContext, TaskResult,
};

use super::{shaders, BlurDirection, GlobalContext};

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

            tcx.write_buffer::<[f32]>(
                self.dst_buffer,
                ..(world.input_image.len() * size_of::<f32>()) as u64,
            )?
            .copy_from_slice(
                world
                    .input_image
                    .as_ref()
                    .expect("input image must be valid pointer"),
            );
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
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::BlurPc {
                    coarse_texture_id: world.physical_resources.samp_coarse,
                    coarse_image_id: world.physical_resources.stor_coarse,
                    coarse_sampler_id: world.physical_resources.sampler,
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
        let wg_count = {
            let wg_size_x = 8;
            let wg_size_y = 8;
            let wg_per_col = height.div_ceil(wg_size_y);
            let wg_per_row = width.div_ceil(wg_size_x);
            [wg_per_row, wg_per_col, 1]
        };
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct SWTTask {
    /// Number of scale space layers plus the extra scratch image
    pub n_coarse_levels_plus_one: u32,
    pub input_level: u32,
    pub direction: BlurDirection,
    pub do_bind_pipeline: bool,
    pub pipeline: Arc<ComputePipeline>,
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
            if self.do_bind_pipeline {
                cbf.bind_pipeline_compute(&self.pipeline)?;
            }

            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::SwtPc {
                    vertical_pass: match self.direction {
                        BlurDirection::Horizontal => 0,
                        BlurDirection::Vertical => 1,
                    },
                    in_level: self.input_level,
                    n_coarse_levels: self.n_coarse_levels_plus_one,
                    coarse_texture_id: world.physical_resources.samp_coarse,
                    coarse_image_id: world.physical_resources.stor_coarse,
                    coarse_sampler_id: world.physical_resources.sampler,
                    width: world.image_width,
                    height: world.image_height,
                },
            )?;
        }
        let width = world.image_width;
        let height = world.image_height;
        let wg_count = {
            let wg_size_x = 8;
            let wg_size_y = 8;
            let wg_per_row = width.div_ceil(wg_size_x);
            let wg_per_col = height.div_ceil(wg_size_y);
            [wg_per_row, wg_per_col, 1]
        };
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct SWTSubTask {
    pub vbuf_fine: Id<Buffer>,
    pub n_fine_scales: u32,
    pub pipeline: Arc<ComputePipeline>,
}

impl Task for SWTSubTask {
    type World = GlobalContext;
    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let buf_fine = tcx.buffer(self.vbuf_fine)?.buffer();
        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::SwtSubPc {
                    fine: buf_fine.device_address()?.into(),
                    coarse_texture_id: world.physical_resources.samp_coarse,
                    coarse_sampler_id: world.physical_resources.sampler,
                    width: world.image_width,
                    height: world.image_height,
                },
            )?;
        }
        let wg_count = {
            let wg_size_x = 32;
            let wg_per_row = world.image_width.div_ceil(wg_size_x);
            [wg_per_row, world.image_height, self.n_fine_scales]
        };
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct ScanExtremaTask {
    pub n_fine_scales: u32,
    pub vbuf_fine: Id<Buffer>,
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
        let addr = |id: Id<Buffer>| {
            Ok::<_, vulkano_taskgraph::TaskError>(tcx.buffer(id)?.buffer().device_address()?)
        };
        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::ScanExtremaPc {
                    fine: addr(self.vbuf_fine)?.into(),
                    extremum_locations: addr(self.vbuf_extremum_locations)?.into(),
                    width: world.image_width,
                    height: world.image_height,
                    n_fine_levels: self.n_fine_scales,
                    border: world.border,
                    contrast_threshold: world.contrast_threshold,
                    skip_layers: world.extremum_skip_layers,
                    max_extrema: world.rt_max_extrema,
                },
            )?;
        }
        let wg_count = {
            let wg_size_x = 4;
            let wg_size_y = 4;
            let wg_size_z = 4;
            // scan from start_layer to n_fine_scale - 1 (inclusive),
            [
                (world.image_width - 2 * world.border).div_ceil(wg_size_x),
                (world.image_height - 2 * world.border).div_ceil(wg_size_y),
                (self.n_fine_scales - 2 - world.extremum_skip_layers).div_ceil(wg_size_z),
            ]
        };
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}
