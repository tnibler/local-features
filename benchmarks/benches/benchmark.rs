use std::path::Path;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use image::{GrayImage, buffer::ConvertBuffer as _};
use ndarray::Array2;
use nshare::IntoNdarray2;

const PATH: &str = "../sample_data/houses.jpg";

fn open_image(path: impl AsRef<Path>) -> GrayImage {
    match image::open(path).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => panic!("wrong image type"),
    }
}

fn resize_image(img: &GrayImage, scale: f32) -> GrayImage {
    let (width, height) = (
        (img.width() as f32 * scale).round() as u32,
        (img.height() as f32 * scale).round() as u32,
    );

    image::imageops::resize(img, width, height, image::imageops::FilterType::Lanczos3)
}

fn image_size(c: &mut Criterion) {
    let n_features = 10;
    let scales_max_features = [
        (0.3, n_features),
        (0.5, n_features),
        (0.8, n_features),
        (1.0, n_features),
    ];
    do_combinations(c, "Image Size", &scales_max_features);
}

fn feature_count(c: &mut Criterion) {
    let scales_max_features = [
        (1.0, 1),
        (1.0, 250),
        (1.0, 500),
        (1.0, 1000),
        (1.0, 5000),
        (1.0, 10000),
    ];
    do_combinations(c, "Feature Count", &scales_max_features);
}

fn do_combinations(c: &mut Criterion, name: &str, scales_max_features: &[(f32, i32)]) {
    let image = open_image(PATH);

    let mut group = c.benchmark_group(name);
    group.sample_size(20);

    let vk = local_features::vulkan::Vulkan::new().expect("need vulkan");

    for (scale, max_features) in scales_max_features.iter().copied() {
        let image = resize_image(&image, scale);
        let width = image.width();
        let height = image.height();
        let image_f32: image::ImageBuffer<image::Luma<f32>, Vec<f32>> = image.convert();
        let image_f32: Array2<f32> = image_f32.into_ndarray2();

        let input_name = format!("{width},{height},{max_features}",);
        let input = (width * height, max_features);

        let mut lf = local_features::new_vulkan(
            &vk,
            local_features::BuildTimeParams {
                n_scales: 5,
                max_image_width: width as u32,
                max_image_height: height as u32,
                max_features: max_features as u32,
                max_blobs: 2 * max_features as u32,
                ..Default::default()
            },
            local_features::FeatureDetectParams::default(),
        )
        .unwrap();

        group.bench_with_input(BenchmarkId::new("Ours", &input_name), &input, |b, _| {
            b.iter(|| {
                lf.detect_extract_all(&image_f32.view(), &Default::default())
                    .unwrap()
            })
        });

        #[cfg(feature = "vulkansift")]
        {
            let image = resize_image(&image, scale);
            let width = image.width();
            let height = image.height();
            use benchmarks::{VulkanSift, VulkanSiftConfig};
            let config = VulkanSiftConfig {
                max_image_width: width,
                max_image_height: height,
                do_upscale: true,
                max_features: max_features as u32,
            };
            let mut vk_sift =
                unsafe { VulkanSift::new(config).expect("Error creating VulkanSift") };

            group.bench_with_input(
                BenchmarkId::new("VulkanSIFT", &input_name),
                &input,
                |b, _| {
                    b.iter(|| vk_sift.detect(image.as_raw(), width, height));
                },
            );
        }

        #[cfg(feature = "opencv")]
        {
            use opencv::prelude::Feature2DTrait;
            let image = resize_image(&image, scale);
            let width = image.width();
            let height = image.height();
            let cv_image = opencv::core::Mat::new_rows_cols_with_data(
                height as i32,
                width as i32,
                image.as_raw(),
            )
            .unwrap();
            let mut sift =
                opencv::features2d::SIFT::create(max_features, 3, 0.04, 10., 1.6, false).unwrap();
            group.bench_with_input(
                BenchmarkId::new("OpenCV SIFT", &input_name),
                &input,
                |b, _| {
                    b.iter(|| {
                        let mut cvkp1 = opencv::core::Vector::new();
                        let mut cvdesc1 = opencv::core::Mat::default();
                        sift.detect_and_compute_def(
                            &cv_image,
                            &opencv::core::no_array(),
                            &mut cvkp1,
                            &mut cvdesc1,
                        )
                        .unwrap();
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, image_size, feature_count);
criterion_main!(benches);
