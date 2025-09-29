use std::sync::Arc;

use itertools::Itertools;
use log::{debug, trace};
use thiserror::Error;
use vulkano::{
    Version, VulkanLibrary,
    descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDevice,
    },
    instance::{Instance, InstanceCreateInfo},
};

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum VulkanInitError {
    #[error(transparent)]
    Loading(#[from] vulkano::LoadingError),

    #[error("Unsupported vulkan version: found {have}, need at least {want}")]
    IncompatibleVulkanVersion { have: Version, want: Version },

    #[error("No suitable Vulkan device found")]
    NoDeviceFound,
    #[error("No suitable Vulkan compute queue  found")]
    NoQueueFound,

    #[error(transparent)]
    Other(#[from] vulkano::Validated<vulkano::VulkanError>),
}

#[derive(Clone)]
pub struct Vulkan {
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl Vulkan {
    pub fn new() -> Result<Self, VulkanInitError> {
        let vklib = VulkanLibrary::new().map_err(VulkanInitError::Loading)?;
        let version = vklib.api_version();
        debug!("Vulkan version: {version}");
        if version.minor < 2 {
            return Err(VulkanInitError::IncompatibleVulkanVersion { have: version, want: Version::major_minor(1, 2) });
        }
        let instance = Instance::new(&vklib, &InstanceCreateInfo::default())?;
        trace!("Created vulkan instance");

        let physical_device: Arc<PhysicalDevice> = instance
            .enumerate_physical_devices()
            .map_err(|err| VulkanInitError::Other(vulkano::Validated::Error(err)))?
            .inspect(|dev| {
                trace!(
                    "Available device: {} ({:?}), {}",
                    dev.properties().device_name,
                    dev.properties().device_type,
                    dev.properties().driver_name.as_deref().unwrap_or("no driver name"),
                )
            })
            .map(|dev| match dev.properties().device_type {
                vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => (0, dev),
                vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => (1, dev),
                vulkano::device::physical::PhysicalDeviceType::VirtualGpu => (2, dev),
                vulkano::device::physical::PhysicalDeviceType::Cpu => (3, dev),
                _ => (i32::MAX, dev),
            })
            .sorted_by_key(|(i, _dev)| *i)
            .map(|(_, dev)| dev)
            .find(|p| {
                p.supported_extensions()
                    .contains(&DeviceExtensions { ext_subgroup_size_control: true, ..Default::default() })
            })
            .ok_or(VulkanInitError::NoDeviceFound)?;

        debug!("Using device: {}", physical_device.properties().device_name);

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|props| props.queue_flags.contains(QueueFlags::COMPUTE))
            .ok_or(VulkanInitError::NoQueueFound)? as u32;

        trace!("Found suitable queue family {queue_family_index}, creating Vulkan device");

        let (device, mut queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                enabled_features: &DeviceFeatures {
                    uniform_buffer_standard_layout: true,
                    buffer_device_address: true,

                    subgroup_broadcast_dynamic_id: true,
                    subgroup_size_control: true,
                    ..Default::default()
                },
                enabled_extensions: &DeviceExtensions {
                    ext_subgroup_size_control: true,
                    khr_buffer_device_address: true,
                    ..Default::default()
                },
                queue_create_infos: &[QueueCreateInfo { queue_family_index, ..Default::default() }],
                ..Default::default()
            },
        )?;
        trace!("Created device");
        let queue = queues
            .next()
            .expect("there should be exactly 1 queue, since 1 queue family index was supplied to Device::new");
        trace!(
            "Vulkan initialized with queue family index {}, queue index {}",
            queue.queue_family_index(),
            queue.queue_index()
        );

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            &device,
            &StandardDescriptorSetAllocatorCreateInfo { set_count: 1, ..Default::default() },
        );
        Ok(Self { device, queue, descriptor_set_allocator: descriptor_set_allocator.into() })
    }
}
