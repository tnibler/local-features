use std::sync::Mutex;

use local_features::FeaturesResult;
use numpy::{IntoPyArray as _, PyReadonlyArrayDyn};
use pyo3::{
    prelude::*,
    types::{PyList, PyTuple},
};

#[pyclass]
struct Keypoint {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
    #[pyo3(get, set)]
    pub size: f32,
    #[pyo3(get, set)]
    pub angle: f32,
    #[pyo3(get, set)]
    pub response: f32,
}

#[pyclass]
struct Extremum {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
    #[pyo3(get, set)]
    pub size: f32,
    #[pyo3(get, set)]
    pub response: f32,
}

#[pyclass]
struct LocalFeatures {
    inner: Mutex<local_features::LocalFeaturesVulkan>,
}

#[pymethods]
impl LocalFeatures {
    #[new]
    #[pyo3(
        text_signature = "(max_image_width, max_image_height, max_features, max_blobs, n_scales, pca)"
    )]
    fn new(
        max_image_width: u32,
        max_image_height: u32,
        max_features: u32,
        max_blobs: u32,
        n_scales: u32,
        pca: &str,
    ) -> PyResult<LocalFeatures> {
        let pca = match pca {
            "yosemite" => local_features::MKDPCA::YOSEMITE,
            "liberty" => local_features::MKDPCA::LIBERTY,
            "notredame" => local_features::MKDPCA::NOTREDAME,
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Invalid PCA argument",
                ));
            }
        };
        let lf = local_features::new_vulkan(
            local_features::BuildTimeParams {
                max_image_width,
                max_image_height,
                n_scales,
                max_features,
                max_blobs,
                pca,
            },
            local_features::FeatureDetectParams::default(),
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err((
                "Failed to initialize local features",
                e.to_string(),
            ))
        })?;
        Ok(LocalFeatures { inner: lf.into() })
    }

    #[pyo3(text_signature = "(img, /)")]
    fn detect(&self, py: Python<'_>, img: PyReadonlyArrayDyn<'_, f32>) -> PyResult<Py<PyTuple>> {
        let arr2 = img.as_array().into_dimensionality().unwrap();
        let FeaturesResult {
            keypoints,
            descriptors,
            ..
        } = self
            .inner
            .lock()
            .unwrap()
            .detect_extract_all(&arr2)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err((
                    "Failed to extract features",
                    e.to_string(),
                ))
            })?;
        let kps = keypoints.into_iter().map(|kp| Keypoint {
            x: kp.x,
            y: kp.y,
            size: kp.size,
            angle: kp.angle,
            response: kp.response,
        });
        let kps = PyList::new(py, kps)?;
        let desc = descriptors.into_pyarray(py);
        Ok((kps, desc).into_pyobject(py)?.into())
    }

    #[pyo3(text_signature = "(img, n, min_size/)")]
    fn detect_top_n(
        &self,
        py: Python<'_>,
        img: PyReadonlyArrayDyn<'_, f32>,
        n: u32,
        min_size: f32,
    ) -> PyResult<Py<PyTuple>> {
        let arr2 = img.as_array().into_dimensionality().unwrap();
        let FeaturesResult {
            keypoints,
            descriptors,
            ..
        } = self
            .inner
            .lock()
            .unwrap()
            .detect_top_n(&arr2, n, min_size)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err((
                    "Failed to extract features",
                    e.to_string(),
                ))
            })?;
        let kps = keypoints.into_iter().map(|kp| Keypoint {
            x: kp.x,
            y: kp.y,
            size: kp.size,
            angle: kp.angle,
            response: kp.response,
        });
        let kps = PyList::new(py, kps)?;
        let desc = descriptors.into_pyarray(py);
        Ok((kps, desc).into_pyobject(py)?.into())
    }
}

/// The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn local_features_python(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Keypoint>()?;
    m.add_class::<LocalFeatures>()?;

    Ok(())
}
