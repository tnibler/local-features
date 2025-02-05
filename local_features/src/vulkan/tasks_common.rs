use vulkano::buffer::Buffer;
use vulkano_taskgraph::{
    command_buffer::{BufferCopy, CopyBufferInfo, RecordingCommandBuffer},
    Id, Task, TaskContext, TaskResult,
};

use super::GlobalContext;

pub(super) struct CopyBufferTask {
    pub src: Id<Buffer>,
    pub dst: Id<Buffer>,
    pub size: u64,
    pub src_offset: u64,
}

impl Task for CopyBufferTask {
    type World = GlobalContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        unsafe {
            cbf.copy_buffer(&CopyBufferInfo {
                src_buffer: self.src,
                dst_buffer: self.dst,
                regions: &[BufferCopy {
                    size: self.size,
                    src_offset: self.src_offset,
                    ..Default::default()
                }],
                ..Default::default()
            })?;
        }
        Ok(())
    }
}
