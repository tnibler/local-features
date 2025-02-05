use std::sync::Arc;

use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};
use vulkano_taskgraph::descriptor_set::BindlessContext;

use super::Error;

#[derive(Clone)]
pub struct Vulkan {
    pub instance: Arc<Instance>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue_family_index: u32,
    pub queue: Arc<Queue>,
}

impl Vulkan {
    pub fn new() -> Result<Self, Error> {
        let vklib = VulkanLibrary::new().map_err(Error::Loading)?;
        let version = vklib.api_version();
        if version.minor < 2 {
            return Err(Error::IncompatibleVulkanVersion {
                have: format!("{}.{}.{}", version.major, version.minor, version.patch),
                want: "1.2".to_string(),
            });
        }
        let instance = Instance::new(&vklib, &InstanceCreateInfo::default())?;

        let required_features = BindlessContext::required_features(&instance);
        let required_extensions = BindlessContext::required_extensions(&instance);

        let physical_device: Arc<PhysicalDevice> = instance
            .enumerate_physical_devices()
            .map_err(Error::Vulkan)?
            .find(|p| {
                p.properties().device_type
                    == vulkano::device::physical::PhysicalDeviceType::IntegratedGpu
                    && p.supported_extensions().contains(&required_extensions)
                    && p.supported_features().contains(&required_features)
                    && p.supported_extensions().contains(&DeviceExtensions {
                        ext_subgroup_size_control: true,
                        ..Default::default()
                    })
            })
            .ok_or(Error::NoDeviceFound)?;

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|props| props.queue_flags.contains(QueueFlags::COMPUTE))
            .ok_or(Error::NoQueueFound)? as u32;

        let (device, mut queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                enabled_features: &required_features.union(&DeviceFeatures {
                    uniform_buffer_standard_layout: true,
                    runtime_descriptor_array: true,
                    buffer_device_address: true,

                    subgroup_size_control: true,
                    storage_buffer16_bit_access: true,
                    shader_float16: true,
                    ..Default::default()
                }),
                enabled_extensions: &required_extensions.union(&DeviceExtensions {
                    ext_subgroup_size_control: true,
                    khr_buffer_device_address: true,
                    ext_descriptor_indexing: true,
                    ..Default::default()
                }),
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;
        let queue = queues.next().ok_or(Error::NoQueueFound)?;

        Ok(Self {
            instance,
            physical_device,
            queue_family_index,
            device,
            queue,
        })
    }
}
