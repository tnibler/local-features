use std::sync::Arc;

use log::trace;
use vulkano::{
    image::{Image, ImageAspects, ImageSubresourceLayers},
    pipeline::{ComputePipeline, Pipeline},
};
use vulkano_taskgraph::{
    command_buffer::{BlitImageInfo, ImageBlit},
    graph::{NodeId, TaskGraph},
    resource::{AccessTypes, ImageLayoutType},
    Id, QueueFamilyType, Task,
};

use super::{shaders, BlurDirection, GlobalContext};

struct PatchPyramidTask {
    in_level: u32,
    do_bind_pipeline: bool,
    direction: BlurDirection,
    // FIXME: this should be runtime set in world, n_levels is max_n_levels there might be fewer
    n_levels: u32,
    pipeline: Arc<ComputePipeline>,
}

pub struct PatchPyramidArgs {
    pub coarse_image_id: Id<Image>,
    pub pyr_image_id: Id<Image>,
    pub tmp_image_id: Id<Image>,
    // FIXME: this should be runtime set in world, n_levels is max_n_levels there might be fewer
    pub n_levels: u32,
}

// copy 0th coarse image to pyramid level 0
// decimate 1st corase image to pyramid level 1
// then recursively filter horizontal, filter and subsample vertical

pub fn patch_pyramid_nodes(
    PatchPyramidArgs {
        coarse_image_id,
        n_levels,
        pyr_image_id,
        tmp_image_id,
    }: PatchPyramidArgs,
    pipeline: Arc<ComputePipeline>,
    taskgraph: &mut TaskGraph<GlobalContext>,
) -> Result<(NodeId, NodeId), crate::vulkan::Error> {
    assert!(n_levels > 1); // TODO: > or >= 1, and this shoudln't assert just do nothing if we don't have to

    let copy0_node_id = taskgraph
        .create_task_node(
            "copy coarse image 0",
            QueueFamilyType::Graphics,
            BlitCopyImageTask {
                vimg_src: coarse_image_id,
                vimg_dst: pyr_image_id,
                src_array_layer: 0,
                dst_mip_level: 0,
                half_size: false,
            },
        )
        .image_access(
            coarse_image_id,
            AccessTypes::BLIT_TRANSFER_READ,
            ImageLayoutType::General,
        )
        .image_access(
            pyr_image_id,
            AccessTypes::BLIT_TRANSFER_WRITE,
            ImageLayoutType::General,
        )
        .build();

    let decimate1_node_id = taskgraph
        .create_task_node(
            "decimate coarse image 1",
            QueueFamilyType::Graphics,
            BlitCopyImageTask {
                vimg_src: coarse_image_id,
                vimg_dst: pyr_image_id,
                src_array_layer: 1,
                dst_mip_level: 1,
                half_size: true,
            },
        )
        .image_access(
            coarse_image_id,
            AccessTypes::BLIT_TRANSFER_READ,
            ImageLayoutType::General,
        )
        .image_access(
            pyr_image_id,
            AccessTypes::BLIT_TRANSFER_WRITE,
            ImageLayoutType::General,
        )
        .build();

    taskgraph
        .add_edge(copy0_node_id, decimate1_node_id)
        .unwrap();

    let start_node = copy0_node_id;
    let mut end_node = decimate1_node_id;
    for in_level in 1..n_levels - 1 {
        let horz_node_id = taskgraph
            .create_task_node(
                "patch pyramid horz",
                vulkano_taskgraph::QueueFamilyType::Compute,
                PatchPyramidTask {
                    in_level,
                    do_bind_pipeline: in_level == 1,
                    direction: BlurDirection::Horizontal,
                    n_levels,
                    pipeline: pipeline.clone(),
                },
            )
            .image_access(
                pyr_image_id,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .image_access(
                tmp_image_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .build();
        let vert_node_id = taskgraph
            .create_task_node(
                "patch pyramid horz",
                vulkano_taskgraph::QueueFamilyType::Compute,
                PatchPyramidTask {
                    in_level,
                    direction: BlurDirection::Vertical,
                    do_bind_pipeline: false,
                    n_levels,
                    pipeline: pipeline.clone(),
                },
            )
            .image_access(
                tmp_image_id,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .image_access(
                pyr_image_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .build();
        taskgraph.add_edge(end_node, horz_node_id).unwrap();
        taskgraph.add_edge(horz_node_id, vert_node_id).unwrap();
        end_node = vert_node_id;
    }
    Ok((start_node, end_node))
}

impl Task for PatchPyramidTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer<'_>,
        _tcx: &mut vulkano_taskgraph::TaskContext<'_>,
        world: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        assert_eq!(
            self.n_levels as usize,
            world.physical_resources.stor_patch_pyr.len()
        ); // TODO: n_levels is more like max_n_levels and must be <= world.pyr.len()
        if self.in_level >= world.rt_patch_pyr_levels {
            return Ok(());
        }
        let in_width = world.image_width / 2u32.pow(self.in_level);
        let in_height = world.image_height / 2u32.pow(self.in_level);
        trace!(
            "Blur in_level: {} ({:?}), {}x{}",
            self.in_level,
            self.direction,
            in_width,
            in_height
        );
        if self.do_bind_pipeline {
            unsafe {
                cbf.bind_pipeline_compute(&self.pipeline)?;
            }
        }
        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &shaders::shaders_f32::BlurPyramidPc {
                    vertical_pass: match self.direction {
                        BlurDirection::Vertical => 1,
                        _ => 0,
                    },
                    in_level: self.in_level,
                    in_sampler_id: world.physical_resources.sampler,
                    in_texture_id: match self.direction {
                        BlurDirection::Horizontal => world.physical_resources.samp_patch_pyr,
                        BlurDirection::Vertical => world.physical_resources.samp_patch_pyr_tmp,
                    },
                    out_image_id: match self.direction {
                        BlurDirection::Horizontal => world.physical_resources.stor_patch_pyr_tmp,
                        BlurDirection::Vertical => {
                            world.physical_resources.stor_patch_pyr[self.in_level as usize + 1]
                        }
                    },
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

struct BlitCopyImageTask {
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
        trace!(
            "Downsample in_level: {} to {}x{}",
            self.src_array_layer,
            dst_offset[0],
            dst_offset[1]
        );
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
