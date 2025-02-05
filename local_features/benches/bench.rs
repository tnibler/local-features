use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{buffer::ConvertBuffer as _, GrayImage};
use ndarray::Array2;
use nshare::IntoNdarray2 as _;
use std::{path::Path, time::Duration};

const PATH: &str = "../sample_data/houses.jpg";

fn open_image(path: impl AsRef<Path>, scale: f32) -> GrayImage {
    let img = match image::open(path).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => panic!("wrong image type"),
    };
    let (width, height) = (
        (img.width() as f32 * scale).round() as u32,
        (img.height() as f32 * scale).round() as u32,
    );

    image::imageops::resize(&img, width, height, image::imageops::FilterType::Lanczos3)
}

struct BenchmarkInput {
    pub scale: f32,
    pub max_features: i32,
}
impl std::fmt::Display for BenchmarkInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x-{}feats", self.scale, self.max_features)
    }
}

fn all_scales() -> Vec<f32> {
    let n = 4;
    (1..=n).map(|i| (i as f32) / (n as f32)).collect()
}

fn all_n_features() -> Vec<i32> {
    vec![100, 500, 1000, 2000]
}

fn bench(n_scales: u32, scales_feats: Vec<(f32, i32)>, group: &str, c: &mut Criterion) {
    let mut group = c.benchmark_group(group);
    group.sample_size(50);
    group.measurement_time(Duration::from_millis(30 * 40));
    for (scale, max_features) in scales_feats {
        let image = open_image(PATH, scale);
        let image: image::ImageBuffer<image::Luma<f32>, Vec<f32>> = image.convert();
        let image: Array2<f32> = image.into_ndarray2();

        let input = BenchmarkInput {
            scale,
            max_features,
        };
        let mut lf = local_features::new_vulkan(
            local_features::BuildTimeParams {
                n_scales,
                max_image_width: image.ncols() as u32,
                max_image_height: image.nrows() as u32,
                max_features: max_features as u32,
                max_blobs: 5 * max_features as u32,
                ..Default::default()
            },
            local_features::FeatureDetectParams::default(),
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(input), &scale, |b, _input| {
            b.iter(|| {
                lf.detect_top_n(&image.view(), max_features as u32, 0.)
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn do_benches_scale(c: &mut Criterion) {
    let n_features = 3000;
    bench(
        3,
        all_scales().into_iter().map(|s| (s, n_features)).collect(),
        "scale_scales=3",
        c,
    );
    bench(
        5,
        all_scales().into_iter().map(|s| (s, n_features)).collect(),
        "scale_scales=5",
        c,
    );
}

fn do_benches_nfeats(c: &mut Criterion) {
    let scale = 1.;
    bench(
        3,
        all_n_features()
            .into_iter()
            .map(|n| (scale, n.max(1)))
            .collect(),
        "feats_scales=3",
        c,
    );
    bench(
        5,
        all_n_features()
            .into_iter()
            .map(|n| (scale, n.max(1)))
            .collect(),
        "feats_scales=5",
        c,
    );
}

criterion_group!(benches, do_benches_nfeats, do_benches_scale,);
criterion_main!(benches);
