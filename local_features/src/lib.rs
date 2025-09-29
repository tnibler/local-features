#![forbid(unsafe_op_in_unsafe_fn)]

use ndarray::Array2;

use crate::vulkan::LocalFeaturesVulkan;

mod mkd_ref;
pub mod vulkan;

const DIMS_INPUT: usize = 7;
const DIMS_EMB_CARTESIAN: usize = 9;
const DIMS_EMB_POLAR: usize = 25;
const RAW_DESCRIPTOR_LEN: usize = DIMS_INPUT * (DIMS_EMB_CARTESIAN + DIMS_EMB_POLAR);
const DESCRIPTOR_LEN: usize = 128;
const PATCH_SIZE: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum MKDPCA {
    LIBERTY,
    NOTREDAME,
    YOSEMITE,
}

#[derive(Debug, Clone)]
/// Algorithm-related parameters that can be adjusted at runtime
pub struct FeatureDetectParams {
    pub patch_scale_factor: f32,
    // pub fix_scale: bool,
    // pub cm_low: f32,
    // pub cm_high: f32,
}

impl Default for FeatureDetectParams {
    fn default() -> Self {
        Self {
            patch_scale_factor: 24.,
            // cm_low: 0.7,
            // cm_high: 1.5,
            // fix_scale: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BuildTimeParams {
    pub n_scales: u32,
    pub max_image_width: u32,
    pub max_image_height: u32,
    pub max_features: u32,
    pub max_blobs: u32,
    pub pca: MKDPCA,
    /// Exponent, so 1 means downsampling by a factor of 2
    pub keypoint_orientation_patch_subsample: u32,
}

impl Default for BuildTimeParams {
    fn default() -> Self {
        Self {
            n_scales: 4,
            max_features: 2000,
            max_blobs: 8000,
            max_image_height: 0,
            max_image_width: 0,
            pca: MKDPCA::LIBERTY,
            keypoint_orientation_patch_subsample: 0,
        }
    }
}

#[derive(Clone)]
pub struct FeaturesResult {
    pub keypoints: Vec<Keypoint>,
    pub descriptors: Array2<f32>,
    pub dropped_blobs: u32,
    pub dropped_features: u32,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LocalFeaturesError {
    #[error("Illegal parameter: {0}")]
    InvalidParameters(String),
    #[error(transparent)]
    VulkanError(#[from] vulkan::VulkanError),
}

pub fn new_vulkan(
    vk: &vulkan::Vulkan,
    fixed_params: BuildTimeParams,
    params: FeatureDetectParams,
) -> Result<LocalFeaturesVulkan, LocalFeaturesError> {
    Ok(LocalFeaturesVulkan::new(fixed_params, params, vk.clone())?)
}
