use std::sync::Arc;

use log::trace;
use vulkano::{
    image::{Image, ImageAspects, ImageSubresourceLayers},
    pipeline::{ComputePipeline, Pipeline},
};
use vulkano_taskgraph::{
    Id, Task,
    command_buffer::{BlitImageInfo, ImageBlit},
    graph::{NodeId, TaskGraph},
    resource::{AccessTypes, ImageLayoutType},
};

use super::{BlurDirection, GlobalContext, shaders};

struct PatchPyramidTask {
    in_level: u32,
    do_bind_pipeline: bool,
    direction: BlurDirection,
    pipeline: Arc<ComputePipeline>,
}

pub struct PatchPyramidArgs {
    pub pyr_image_id: Id<Image>,
    pub tmp_image_id: Id<Image>,
    pub n_levels: u32,
}

pub fn patch_pyramid_nodes(
    PatchPyramidArgs { n_levels, pyr_image_id, tmp_image_id }: PatchPyramidArgs,
    pipeline: Arc<ComputePipeline>,
    taskgraph: &mut TaskGraph<GlobalContext>,
) -> (NodeId, NodeId) {
    let mut start_node = None;
    let mut connect_prev_node = None;
    for in_level in 1..n_levels - 1 {
        let horz_node_id = taskgraph
            .create_task_node(
                "patch pyramid horz",
                vulkano_taskgraph::QueueFamilyType::Compute,
                PatchPyramidTask {
                    in_level,
                    do_bind_pipeline: in_level == 1,
                    direction: BlurDirection::Horizontal,
                    pipeline: pipeline.clone(),
                },
            )
            .image_access(pyr_image_id, AccessTypes::COMPUTE_SHADER_SAMPLED_READ, ImageLayoutType::General)
            .image_access(tmp_image_id, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE, ImageLayoutType::General)
            .build();
        if start_node.is_none() {
            start_node = Some(horz_node_id);
        }
        let vert_node_id = taskgraph
            .create_task_node(
                "patch pyramid horz",
                vulkano_taskgraph::QueueFamilyType::Compute,
                PatchPyramidTask {
                    in_level,
                    direction: BlurDirection::Vertical,
                    do_bind_pipeline: false,
                    pipeline: pipeline.clone(),
                },
            )
            .image_access(tmp_image_id, AccessTypes::COMPUTE_SHADER_SAMPLED_READ, ImageLayoutType::General)
            .image_access(pyr_image_id, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE, ImageLayoutType::General)
            .build();
        if let Some(node) = connect_prev_node {
            taskgraph.add_edge(node, horz_node_id).unwrap();
        }
        taskgraph.add_edge(horz_node_id, vert_node_id).unwrap();
        connect_prev_node = Some(vert_node_id);
    }
    (start_node.unwrap(), connect_prev_node.unwrap())
}

impl Task for PatchPyramidTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer<'_>,
        _tcx: &mut vulkano_taskgraph::TaskContext<'_>,
        world: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        if self.in_level >= world.rt_patch_pyr_levels {
            return Ok(());
        }
        let in_width = world.image_width / 2u32.pow(self.in_level);
        let in_height = world.image_height / 2u32.pow(self.in_level);
        trace!("Patch Pyramid Blur, in_level: {} ({:?}), {}x{}", self.in_level, self.direction, in_width, in_height);
        if self.do_bind_pipeline {
            unsafe {
                cbf.bind_pipeline_compute(&self.pipeline)?;
            }
        }
        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::ConvPc {
                    vertical_pass: match self.direction {
                        BlurDirection::Vertical => 1,
                        BlurDirection::Horizontal => 0,
                    },
                    in_level: self.in_level,
                    width: in_width,
                    height: in_height,
                },
            )?;
        }
        let wg_count = {
            let wg_size_x = 8;
            let wg_size_y = 8;
            let (out_width, out_height) = match self.direction {
                BlurDirection::Vertical => (in_width / 2, in_height / 2),
                BlurDirection::Horizontal => (in_width, in_height),
            };
            let wg_per_col = out_height.div_ceil(wg_size_y);
            let wg_per_row = out_width.div_ceil(wg_size_x);
            [wg_per_row, wg_per_col, 1]
        };
        unsafe {
            cbf.dispatch(wg_count)?;
        }
        Ok(())
    }
}

pub(super) struct BlitCopyImageTask {
    pub vimg_src: Id<Image>,
    pub vimg_dst: Id<Image>,
    pub src_array_layer: u32,
    pub dst_mip_level: u32,
    pub half_size: bool,
}

impl Task for BlitCopyImageTask {
    type World = GlobalContext;
    unsafe fn execute(
        &self,
        cbf: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer<'_>,
        _tcx: &mut vulkano_taskgraph::TaskContext<'_>,
        world: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        if self.src_array_layer >= world.rt_patch_pyr_levels {
            return Ok(());
        }
        let dst_offset = if self.half_size {
            [world.image_width / 2, world.image_height / 2, 1]
        } else {
            [world.image_width, world.image_height, 1]
        };
        trace!("BlitCopy in_level: {} to {}x{}", self.src_array_layer, dst_offset[0], dst_offset[1]);
        unsafe {
            cbf.blit_image(&BlitImageInfo {
                src_image: self.vimg_src,
                dst_image: self.vimg_dst,
                src_image_layout: ImageLayoutType::General,
                dst_image_layout: ImageLayoutType::General,
                filter: vulkano::image::sampler::Filter::Nearest,
                regions: &[ImageBlit {
                    src_subresource: ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        base_array_layer: self.src_array_layer,
                        ..Default::default()
                    },
                    src_offsets: [[0, 0, 0], [world.image_width, world.image_height, 1]],
                    dst_subresource: ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        mip_level: self.dst_mip_level,
                        ..Default::default()
                    },
                    dst_offsets: [[0, 0, 0], dst_offset],
                    ..Default::default()
                }],
                ..Default::default()
            })?;
        }
        Ok(())
    }
}
