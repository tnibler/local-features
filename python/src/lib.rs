use std::sync::Mutex;

use local_features::FeatureDetectParams;
use numpy::{IntoPyArray as _, PyReadonlyArrayDyn};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PyDictMethods, PyList, PyString, PyStringMethods, PyTuple},
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
struct LocalFeatures {
    inner: Mutex<local_features::vulkan::LocalFeaturesVulkan>,
}

#[pymethods]
impl LocalFeatures {
    #[new]
    #[pyo3(
        text_signature = "(max_image_width, max_image_height, n_scales, max_features, max_blobs, /)"
    )]
    fn new(
        max_image_width: u32,
        max_image_height: u32,
        n_scales: u32,
        max_features: u32,
        max_blobs: u32,
    ) -> PyResult<LocalFeatures> {
        let vk = local_features::vulkan::Vulkan::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err((
                "Failed to initialize Vulkan",
                e.to_string(),
            ))
        })?;
        let lf = local_features::new_vulkan(
            &vk,
            local_features::BuildTimeParams {
                n_scales,
                max_image_width,
                max_image_height,
                max_features,
                max_blobs,
                ..Default::default()
            },
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err((
                "Failed to initialize local features",
                e.to_string(),
            ))
        })?;
        Ok(LocalFeatures { inner: lf.into() })
    }

    #[pyo3(signature = (img, **kwargs))]
    fn detect_extract_all(
        &self,
        py: Python<'_>,
        img: PyReadonlyArrayDyn<'_, f32>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyTuple>> {
        let mut params = FeatureDetectParams::default();
        if let Some(kwargs) = kwargs {
            let any_float = |obj: Bound<'_, PyAny>| {
                obj.extract::<f32>()
                    .or_else(|_| obj.extract::<i32>().map(|i| i as f32))
            };
            for (k, v) in kwargs {
                let k = k.cast_into::<PyString>()?;
                match k.to_str()? {
                    "edgeness_cm_low" => {
                        params.edgeness_cm_low = any_float(v)?;
                    }
                    "edgeness_cm_high" => {
                        params.edgeness_cm_high = any_float(v)?;
                    }
                    "patch_scale_factor" => {
                        params.patch_scale_factor = any_float(v)?;
                    }
                    "min_response" => {
                        params.extremum_min_response = any_float(v)?;
                    }
                    _ => {
                        return Err(PyRuntimeError::new_err(format!(
                            "Unexpected keyword argument {k}"
                        )));
                    }
                }
            }
        }
        let arr2 = img.as_array().into_dimensionality().unwrap();
        let result = self
            .inner
            .lock()
            .unwrap()
            .detect_extract_all(&arr2, &params)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err((
                    "Failed to extract features",
                    e.to_string(),
                ))
            })?;

        let kps = result.keypoints.into_iter().map(|kp| Keypoint {
            x: kp.x,
            y: kp.y,
            size: kp.size,
            angle: kp.angle,
            response: kp.response,
        });
        let kps = PyList::new(py, kps)?;
        let desc = result.descriptors.into_pyarray(py);
        Ok((kps, desc).into_pyobject(py)?.into())
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn local_features_python(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Keypoint>()?;
    m.add_class::<LocalFeatures>()?;

    Ok(())
}
