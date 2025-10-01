use std::{path::Path, time::Duration};

use benchmarks::{VulkanSift, VulkanSiftConfig};
use criterion::{Criterion, criterion_group, criterion_main};
use image::{GrayImage, buffer::ConvertBuffer as _};
use ndarray::Array2;
use nshare::IntoNdarray2;

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

fn vulkansift(c: &mut Criterion) {
    let mut group = c.benchmark_group("vulkansift");
    group.sample_size(30);

    let scale = 1.0;
    let image = open_image(PATH, scale);
    let image: Array2<u8> = image.into_ndarray2();
    let width = image.ncols() as u32;
    let height = image.nrows() as u32;
    let config = VulkanSiftConfig {
        max_image_width: width,
        max_image_height: height,
        do_upscale: true,
        max_features: 8000,
    };
    let mut vk_sift = unsafe { VulkanSift::new(config).expect("Error creating VulkanSift") };

    group.bench_function("vulkansift", |b| {
        b.iter(|| vk_sift.detect(image.as_slice().unwrap(), width, height));
    });
    group.finish();
}

fn local_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_features");
    group.sample_size(50);
    group.measurement_time(Duration::from_millis(6000));
    let scale = 1.0;
    let image = open_image(PATH, scale);
    let image: image::ImageBuffer<image::Luma<f32>, Vec<f32>> = image.convert();
    let image: Array2<f32> = image.into_ndarray2();
    let max_features = 8000;

    let vk = local_features::vulkan::Vulkan::new().expect("need vulkan");
    let mut lf = local_features::new_vulkan(
        &vk,
        local_features::BuildTimeParams {
            n_scales: 5,
            max_image_width: image.ncols() as u32,
            max_image_height: image.nrows() as u32,
            max_features: max_features as u32,
            max_blobs: 3 * max_features as u32,
            ..Default::default()
        },
        local_features::FeatureDetectParams::default(),
    )
    .unwrap();
    group.bench_function("local_features", |b| {
        b.iter(|| lf.detect_extract_all(&image.view()).unwrap())
    });
    group.finish();
}

criterion_group!(benches, vulkansift, local_features);
criterion_main!(benches);
