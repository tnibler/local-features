use std::mem::MaybeUninit;

use anyhow::{Result, ensure};

use crate::vulkan_sift::bindings::{_BITS_STDINT_INTN_H, vksift_Feature};

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(unused)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Debug, Clone)]
pub struct VulkanSiftConfig {
    pub max_image_width: u32,
    pub max_image_height: u32,
    pub do_upscale: bool,
    pub max_features: u32,
}

pub struct VulkanSift {
    config: VulkanSiftConfig,
    instance_ptr: *mut bindings::vksift_Instance_T,
    _loaded_vulkan: LoadedVulkan,
}

pub struct VulkanSiftFeature {
    pub x: f32,
    pub y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_idx: u32,
    pub octave_idx: i32,
    pub sigma: f32,
    pub orientation: f32,
    pub contrast: f32,
}

struct LoadedVulkan;

impl VulkanSift {
    pub unsafe fn new(config: VulkanSiftConfig) -> Result<Self> {
        unsafe {
            ensure!(
                bindings::vksift_loadVulkan() == bindings::vksift_Result_VKSIFT_SUCCESS,
                "Error loading vulkan"
            );
            let loaded_vulkan = LoadedVulkan;

            unsafe {
                bindings::vksift_setLogLevel(bindings::vksift_LogLevel_VKSIFT_NO_LOG);
            }
            let mut vk_config = bindings::vksift_getDefaultConfig();
            vk_config.sift_buffer_count = 1;
            vk_config.use_hardware_interpolated_blur = true;
            vk_config.use_input_upsampling = true;
            vk_config.pyramid_precision_mode =
                bindings::vksift_PyramidPrecisionMode_VKSIFT_PYRAMID_PRECISION_FLOAT32;
            vk_config.max_nb_sift_per_buffer = config.max_features;
            vk_config.input_image_max_size = config.max_image_width * config.max_image_height;
            let mut instance_ptr: *mut bindings::vksift_Instance_T = std::ptr::null_mut();
            ensure!(
                bindings::vksift_createInstance((&mut instance_ptr) as *mut _, &vk_config)
                    == bindings::vksift_Result_VKSIFT_SUCCESS,
                "Error creating VulkanSift Instance"
            );
            Ok(Self {
                config,
                instance_ptr,
                _loaded_vulkan: loaded_vulkan,
            })
        }
    }

    pub fn detect(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<VulkanSiftFeature>, Vec<[u8; 128]>)> {
        assert_eq!(image.len(), width as usize * height as usize);
        let mut features: Vec<MaybeUninit<bindings::vksift_Feature>> = Vec::default();
        unsafe {
            bindings::vksift_detectFeatures(self.instance_ptr, image.as_ptr(), width, height, 0);
            let n_features = bindings::vksift_getFeaturesNumber(self.instance_ptr, 0);
            features.resize(n_features as usize, MaybeUninit::uninit());
            bindings::vksift_downloadFeatures(
                self.instance_ptr,
                features.as_mut_ptr() as *mut _,
                0,
            );
        }
        let (features, descriptors): (Vec<VulkanSiftFeature>, Vec<[u8; 128]>) = unsafe {
            features
                .into_iter()
                .map(|feat| {
                    let feat: vksift_Feature = feat.assume_init();
                    (
                        VulkanSiftFeature {
                            x: feat.x,
                            y: feat.y,
                            scale_x: feat.scale_x,
                            scale_y: feat.scale_y,
                            scale_idx: feat.scale_idx,
                            octave_idx: feat.octave_idx,
                            sigma: feat.sigma,
                            orientation: feat.orientation,
                            contrast: feat.intensity,
                        },
                        feat.descriptor,
                    )
                })
                .unzip()
        };
        Ok((features, descriptors))
    }
}

impl Drop for VulkanSift {
    fn drop(&mut self) {
        unsafe {
            bindings::vksift_destroyInstance((&mut self.instance_ptr) as *mut _);
        }
    }
}

impl Drop for LoadedVulkan {
    fn drop(&mut self) {
        unsafe {
            bindings::vksift_unloadVulkan();
        }
    }
}
