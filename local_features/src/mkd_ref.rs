#![allow(dead_code)]
use itertools::Itertools as _;
use ndarray::{
    concatenate, s, stack, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis,
};

const VM_FOURIER_N3_K8: &[f32] = &[0.378_723_74, 0.517_962_34, 0.468_820_15, 0.397_980_96];
const VM_FOURIER_N1_K1: &[f32] = &[0.618_176, 0.693_472_5];
const VM_FOURIER_N2_K8: &[f32] = &[0.378_723_74, 0.517_962_34, 0.468_820_15];

#[derive(Debug, Clone, Copy)]
pub enum Whitening {
    None,
    PcaAttenutated,
}

#[derive(Debug, Clone)]
pub struct MkdOptions {
    pub whitening: Whitening,
}

pub struct Mkd {
    whitening: Option<PCAModel>,
}

pub(crate) const PCA_SAFETENSORS_LIBERTY: &[u8] =
    include_bytes!("../models/mkd/concat-pca-liberty.safetensors");
pub(crate) const PCA_SAFETENSORS_NOTREDAME: &[u8] =
    include_bytes!("../models/mkd/concat-pca-notredame.safetensors");
pub(crate) const PCA_SAFETENSORS_YOSEMITE: &[u8] =
    include_bytes!("../models/mkd/concat-pca-yosemite.safetensors");

impl Mkd {
    pub fn new(opts: &MkdOptions) -> Self {
        let pca_model = match opts.whitening {
            Whitening::None => None,
            Whitening::PcaAttenutated => {
                // let bytes = std::fs::read("models/mkd/concat-pca-liberty.safetensors")
                //     .expect("TODO unhandled");
                let model =
                    PCAModel::from_safetensors(PCA_SAFETENSORS_LIBERTY).expect("TODO unhandled");
                Some(model)
            }
        };
        Self {
            whitening: pca_model,
        }
    }

    pub fn descriptor_size(&self) -> u32 {
        match self.whitening {
            None => 238,
            Some(_) => 128,
        }
    }

    pub fn patch(&self, patch: &ArrayView2<f32>) -> Array1<f32> {
        let desc = mkd(patch);
        match &self.whitening {
            None => desc,
            Some(PCAModel {
                mean,
                eigvals,
                eigvecs,
            }) => {
                let t = 0.7;
                let m = -0.5 * t;
                let out_dims = 128;
                let in_dims = 238;
                let eigvecs = eigvecs.slice(s![.., ..out_dims]).to_owned();
                let eigvals = eigvals.slice(s![..out_dims]);
                let eigvecs = eigvecs * eigvals.powf(m).broadcast((in_dims, out_dims)).unwrap();
                let res = (desc - mean).dot(&eigvecs);
                normalize_l2(&res.view())
            }
        }
    }
}

/// Returns, 2x32x32 array: stacked [x_gradient, y_gradient]
/// Border mode: replicate
fn gradients(arr: &ArrayView2<f32>) -> Array3<f32> {
    assert_eq!(arr.shape(), [32, 32]);
    let mut grads = Array3::zeros((2, 32, 32));
    let w = arr.ncols();
    let h = arr.nrows();
    for y in 0..arr.nrows() {
        for x in 0..arr.ncols() {
            // clamping because reference implementation uses border mode replicate
            grads[(0, y, x)] = arr[(y, x.clamp(0, w - 2) + 1)] - arr[(y, x.clamp(1, w - 1) - 1)];
            grads[(1, y, x)] = arr[(y.clamp(1, h - 1) - 1, x)] - arr[(y.clamp(0, h - 2) + 1, x)];
        }
    }
    grads
}

/// Blur with 5x5 kernel, sigma=0.7
/// Border mode: replicate
fn gaussian_blur(arr: &ArrayView2<f32>) -> Array2<f32> {
    assert_eq!(arr.shape(), [32, 32]);
    let kernel = [0.0096_f32, 0.2054, 0.5699, 0.2054, 0.0096];
    assert_eq!(kernel.len() % 2, 1);
    let rad = (kernel.len() - 1) / 2;
    let mut res_vert = Array2::zeros((32, 32));
    let mut res = Array2::zeros((32, 32));
    let w = arr.ncols();
    let h = arr.nrows();
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (i, k) in kernel.iter().enumerate() {
                let yy = (y + i).saturating_sub(rad).min(h - 1);
                sum += k * arr[(yy, x)];
            }
            res_vert[(y, x)] = sum;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (i, k) in kernel.iter().enumerate() {
                let xx = (x + i).saturating_sub(rad).min(w - 1);
                sum += k * res_vert[(y, xx)];
            }
            res[(y, x)] = sum;
        }
    }
    res
}

/// arr: [(x, y), 32, 32]
/// returns [(mag, angle), 32, 32]
pub(crate) fn cart2pol(arr: &ArrayView3<f32>) -> Array3<f32> {
    assert_eq!(arr.shape(), [2, 32, 32]);
    let mut res = Array3::zeros((2, 32, 32));
    let eps = 1e-8;
    for y in 0..arr.shape()[1] {
        for x in 0..arr.shape()[2] {
            res[(0, y, x)] = (arr[(0, y, x)].powi(2) + arr[(1, y, x)].powi(2) + eps).sqrt();
            res[(1, y, x)] = -arr[(1, y, x)].atan2(arr[(0, y, x)]);
        }
    }
    res
}

fn von_mises(arr: &ArrayView2<f32>, coeffs: &[f32]) -> Array3<f32> {
    assert_eq!(arr.shape(), [32, 32]);
    let n = coeffs.len() - 1;
    let weights: Array1<f32> = coeffs
        .iter()
        .copied()
        .chain(coeffs.iter().copied().skip(1))
        .collect();
    assert_eq!(weights.len(), 2 * n + 1);
    // 1 * arr, 2 * arr, .., n * arr
    let frange = (1..=n).map(|a| arr * a as f32).collect_vec();
    let frange: Array3<f32> =
        stack(Axis(0), &frange.iter().map(|a| a.view()).collect_vec()).unwrap();
    assert_eq!(frange.shape(), [n, 32, 32]);
    let emb0 = Array3::ones([1, 32, 32]);
    //
    let emb1 = frange.cos();
    let emb2 = frange.sin();
    // coeff[0] ++ coeffs[1..] ++ coeffs[1..]
    let mut cat = concatenate(Axis(0), &[emb0.view(), emb1.view(), emb2.view()]).unwrap();
    assert_eq!(cat.shape(), [2 * n + 1, 32, 32]);
    cat.axis_iter_mut(Axis(0))
        .zip(weights)
        .for_each(|(mut v, w)| v *= w);
    cat
}

pub(crate) fn mesh_grid() -> Array3<f32> {
    let mut grid: Array3<f32> = Array3::zeros((2, 32, 32));
    let n = 32;
    for y in 0..n {
        for x in 0..n {
            let xf = x as f32;
            let yf = y as f32;
            grid[(0, y, x)] = 2. * xf / (n as f32 - 1.) - 1.;
            grid[(1, y, x)] = 2. * yf / (n as f32 - 1.) - 1.;
        }
    }
    grid
}

#[derive(Debug, Clone, Copy)]
enum RelativeGradients {
    Yes,
    No,
}

fn embed_gradients(grads: &ArrayView3<f32>, relative: RelativeGradients) -> Array3<f32> {
    assert_eq!(grads.shape(), [2, 32, 32]);
    let oris = match relative {
        RelativeGradients::Yes => {
            // PRECOMPUTE
            let kgrid = mesh_grid();
            let phi = cart2pol(&kgrid.view()).slice_move(s![1, .., ..]);

            &grads.slice(s![1, .., ..]) + &phi.view()
        }
        RelativeGradients::No => grads.slice(s![1, .., ..]).to_owned(),
    };
    // dbg!(&oris);
    let embedded_mags = ((grads + 1e-8).slice(s![0, .., ..])).sqrt();
    von_mises(&oris.view(), VM_FOURIER_N3_K8) * embedded_mags
}

pub(crate) fn spatial_kernel_embedding_cart() -> Array3<f32> {
    let grid = mesh_grid() * std::f32::consts::FRAC_PI_2;
    // x
    let emb_a = von_mises(&grid.slice(s![0, .., ..]), VM_FOURIER_N1_K1);
    assert_eq!(emb_a.shape(), [3, 32, 32]);
    // y
    let emb_b = von_mises(&grid.slice(s![1, .., ..]), VM_FOURIER_N1_K1);
    assert_eq!(emb_b.shape(), [3, 32, 32]);
    let mut spatial_kernel = Array3::zeros((emb_a.shape()[0] * emb_b.shape()[0], 32, 32));
    for i in 0..emb_a.shape()[0] {
        for j in 0..emb_b.shape()[0] {
            for r in 0..32 {
                for c in 0..32 {
                    spatial_kernel[(i * emb_b.shape()[0] + j, r, c)] =
                        emb_a[(i, r, c)] * emb_b[(j, r, c)];
                }
            }
        }
    }
    assert_eq!(spatial_kernel.shape(), [9, 32, 32]);
    spatial_kernel
}

pub(crate) fn spatial_kernel_embedding_polar() -> Array3<f32> {
    let grid = cart2pol(&mesh_grid().view());
    let rho =
        grid.slice(s![0, .., ..]).to_owned() * std::f32::consts::PI / std::f32::consts::SQRT_2;
    let phi = grid.slice(s![1, .., ..]).to_owned() * -1.;
    let emb_a = von_mises(&phi.view(), VM_FOURIER_N2_K8);
    assert_eq!(emb_a.shape(), [5, 32, 32]);
    let emb_b = von_mises(&rho.view(), VM_FOURIER_N2_K8);
    assert_eq!(emb_b.shape(), [5, 32, 32]);
    let mut spatial_kernel = Array3::zeros((emb_a.shape()[0] * emb_b.shape()[0], 32, 32));
    // 0..5
    for i in 0..emb_a.shape()[0] {
        // 0..5
        for j in 0..emb_b.shape()[0] {
            for r in 0..32 {
                for c in 0..32 {
                    spatial_kernel[(i * emb_b.shape()[0] + j, r, c)] =
                        emb_a[(i, r, c)] * emb_b[(j, r, c)];
                }
            }
        }
    }
    assert_eq!(spatial_kernel.shape(), [25, 32, 32]);
    spatial_kernel
}

pub(crate) fn gaussian_weighting() -> Array2<f32> {
    let grid = mesh_grid();
    let norm = grid.powi(2).sum_axis(Axis(0)).sqrt();
    assert_eq!(norm.shape(), [32, 32]);
    let max = norm.iter().copied().max_by(f32::total_cmp).unwrap();
    let norm = norm / max;
    let sigma = 1_f32;
    (norm.powi(2) / -(sigma.powi(2))).exp()
}

enum SpatialEncoding {
    Cartesian,
    Polar,
}
fn explicit_spatial_encoding(arr: &ArrayView3<f32>, enc: SpatialEncoding) -> Array1<f32> {
    // 7
    let in_dims = arr.shape()[0];
    let emb = match enc {
        // [9, 32, 32]
        SpatialEncoding::Cartesian => spatial_kernel_embedding_cart(),
        // [25, 32, 32]
        SpatialEncoding::Polar => spatial_kernel_embedding_polar(),
    };
    // 9 or 25
    let d_emb = emb.shape()[0];
    let emb = emb * gaussian_weighting();

    let mut out: Array1<f32> = Array1::zeros((d_emb * in_dims,));
    for r in 0..32 {
        for c in 0..32 {
            for i in 0..in_dims {
                for j in 0..d_emb {
                    out[i * d_emb + j] += arr[(i, r, c)] * emb[(j, r, c)];
                }
            }
        }
    }
    out
}

fn normalize_l2(arr: &ArrayView1<f32>) -> Array1<f32> {
    let norm = arr.powi(2).sum().sqrt();
    arr / norm
}

fn mkd(patch: &ArrayView2<f32>) -> Array1<f32> {
    let smoothed = gaussian_blur(&patch.view());
    let grads = -gradients(&smoothed.view());
    let grads = cart2pol(&grads.view());

    // polar parametrization
    let eg_rel = embed_gradients(&grads.view(), RelativeGradients::Yes);
    assert_eq!(eg_rel.shape(), [7, 32, 32]);

    let sp_enc_polar = explicit_spatial_encoding(&eg_rel.view(), SpatialEncoding::Polar);
    let sp_enc_polar = normalize_l2(&sp_enc_polar.view());
    assert_eq!(sp_enc_polar.shape(), [175]);

    // cartesian parametrization
    let eg_norel = embed_gradients(&grads.view(), RelativeGradients::No);
    assert_eq!(eg_norel.shape(), [7, 32, 32]);

    let sp_enc_cart = explicit_spatial_encoding(&eg_norel.view(), SpatialEncoding::Cartesian);
    let sp_enc_cart = normalize_l2(&sp_enc_cart.view());
    assert_eq!(sp_enc_cart.shape(), [63]);

    let out = concatenate(Axis(0), &[sp_enc_polar.view(), sp_enc_cart.view()]).unwrap();
    normalize_l2(&out.view())
}

fn mkd_pca_att(patch: &ArrayView2<f32>) -> Array1<f32> {
    let desc = mkd(patch);
    let PCAModel {
        mean,
        eigvals,
        eigvecs,
    } = PCAModel::from_safetensors(
        std::fs::read("models/mkd/concat-pca-liberty.safetensors")
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let t = 0.7;
    let m = -0.5 * t;
    let out_dims = 128;
    let in_dims = 238;
    let eigvecs = eigvecs.slice(s![.., ..out_dims]).to_owned();
    let eigvals = eigvals.slice(s![..out_dims]);
    let eigvecs = eigvecs * eigvals.powf(m).broadcast((in_dims, out_dims)).unwrap();
    let res = (desc - mean).dot(&eigvecs);
    normalize_l2(&res.view())
}

#[derive(Debug, Clone)]
pub(crate) struct PCAModel {
    pub mean: Array1<f32>,
    pub eigvals: Array1<f32>,
    pub eigvecs: Array2<f32>,
}

fn u8_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
    // TODO: handle errors
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes(ch.try_into().unwrap()))
        .collect_vec()
}

impl PCAModel {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, safetensors::SafeTensorError> {
        let model =
            safetensors::SafeTensors::deserialize(bytes).expect("error deserializing safetensors");
        let mean = model.tensor("mean")?;
        assert_eq!(mean.shape(), [238]);
        let mean_vec: Vec<f32> = u8_slice_to_f32(mean.data());
        let eigvals = model.tensor("eigvals")?;
        let eigvals_vec = u8_slice_to_f32(eigvals.data());
        let eigvecs = model.tensor("eigvecs")?;
        let eigvecs_vec = u8_slice_to_f32(eigvecs.data());
        Ok(PCAModel {
            mean: Array1::from_shape_vec(mean.shape()[0], mean_vec.to_owned())
                .expect("mean has wrong shape"),
            eigvecs: Array2::from_shape_vec(
                (eigvecs.shape()[0], eigvecs.shape()[1]),
                eigvecs_vec.to_owned(),
            )
            .expect("eigvecs has wrong shape"),
            eigvals: Array1::from_shape_vec(eigvals.shape()[0], eigvals_vec.to_owned())
                .expect("eigvals has wrong shape"),
        })
    }
}

// #[cfg(test)]
// mod tests {
//     use ndarray::{Array1, Array2};
//     use serde::Deserialize;
//
//     #[derive(Debug, Clone, Deserialize)]
//     struct TestCase {
//         pub patch: Vec<f32>,
//         pub descriptor: Vec<f32>,
//     }
//
//     #[test]
//
//     fn test_pcawt() {
//         let in_path = "test_vectors/mkd_pcawt.json";
//         let test_cases: Vec<TestCase> = serde_json::from_str(
//             &std::fs::read_to_string(in_path).expect("failed to read test case file"),
//         )
//         .unwrap();
//         let mkd = super::Mkd::new(&super::MkdOptions {
//             whitening: super::Whitening::PcaAttenutated,
//         });
//         for TestCase { patch, descriptor } in test_cases {
//             let patch = Array2::from_shape_vec((32, 32), patch).unwrap();
//             let expected_descriptor = Array1::from_vec(descriptor);
//             let actual_descriptor = mkd.patch(&patch.view());
//             let diff = expected_descriptor - actual_descriptor;
//             let max_diff = diff.abs().iter().copied().max_by(f32::total_cmp).unwrap();
//             let mse = diff.powi(2).mean().unwrap();
//             dbg!(max_diff);
//             dbg!(mse);
//             assert!(mse < 1e-8);
//             assert!(max_diff < 1e-4);
//         }
//     }
//
//     #[test]
//     fn test_no_whitening() {
//         let in_path = "test_vectors/mkd_no_whitening.json";
//         let test_cases: Vec<TestCase> = serde_json::from_str(
//             &std::fs::read_to_string(in_path).expect("failed to read test case file"),
//         )
//         .unwrap();
//         let mkd = super::Mkd::new(&super::MkdOptions {
//             whitening: super::Whitening::None,
//         });
//         for TestCase { patch, descriptor } in test_cases {
//             let patch = Array2::from_shape_vec((32, 32), patch).unwrap();
//             let expected_descriptor = Array1::from_vec(descriptor);
//             let actual_descriptor = mkd.patch(&patch.view());
//             assert_eq!(actual_descriptor.shape(), [238]);
//             let diff = expected_descriptor - actual_descriptor;
//             let max_diff = diff.abs().iter().copied().max_by(f32::total_cmp).unwrap();
//             let mse = diff.powi(2).mean().unwrap();
//             dbg!(max_diff);
//             dbg!(mse);
//             assert!(mse < 1e-5);
//             assert!(max_diff < 1e-5);
//         }
//     }
// }
