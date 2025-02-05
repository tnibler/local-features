use image::{GenericImage as _, GrayImage, buffer::ConvertBuffer};
use itertools::Itertools as _;
use local_features::FeaturesResult;
use log::info;
use ndarray::ArrayView2;
use nshare::AsNdarray2 as _;

fn match_features(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Vec<(usize, usize)> {
    // Very slow n^2 brute force matching.
    let mut res = Vec::default();
    for (i, va) in a.rows().into_iter().enumerate() {
        assert_eq!(va.shape(), [128]);
        let sim = b.map_axis(ndarray::Axis(1), |vb| (&vb * &va).sum());
        assert_eq!(sim.len(), b.shape()[0]);
        let mut idxs = (0..b.shape()[0]).collect_vec();
        idxs.sort_by(|i, j| sim[*i].total_cmp(&sim[*j]));
        // Nearest and second-nearest neighbor. Filter using "Lowe's ratio"
        // (best match must be significantly better than the second best, otherwise it's thrown out).
        // This is why matching 1->2 is not the same as 2->1.
        let first = *idxs.last().unwrap();
        let second = idxs[idxs.len() - 2];
        if sim[first] * 0.8 > sim[second] {
            res.push((i, first));
        }
    }
    res
}
fn main() -> Result<(), ()> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "info");
        }
    }
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Required arguments: IMAGE_1 IMAGE_2 IMAGE_OUT");
        return Err(());
    }
    let path1 = &args[1];
    let path2 = &args[2];
    let out_path = &args[3];

    let img1 = match image::open(path1).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => {
            eprintln!("wrong image type");
            return Err(());
        }
    };
    let img2 = match image::open(path2).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => {
            eprintln!("wrong image type");
            return Err(());
        }
    };

    let img1_f32: image::ImageBuffer<image::Luma<f32>, Vec<f32>> = img1.convert();
    let img2_f32: image::ImageBuffer<image::Luma<f32>, Vec<f32>> = img2.convert();

    let mut feats = local_features::new_vulkan(
        local_features::BuildTimeParams {
            n_scales: 5,
            max_image_width: img1.width().max(img2.width()),
            max_image_height: img1.height().max(img2.height()),
            max_features: 3000,
            max_blobs: 8000,
            ..Default::default()
        },
        local_features::FeatureDetectParams::default(),
    )
    .unwrap();

    let min_size = 0.0;
    let top_n = 2000;
    info!("Limiting to best {top_n} features, minimum size {min_size}");
    let result1 = feats
        .detect_top_n(&img1_f32.as_ndarray2(), top_n, min_size)
        .unwrap();

    if result1.dropped_blobs > 0 || result1.dropped_features > 0 {
        info!(
            "Extracted {} keypoints. {} candidate blobs and {} keypoints did not fit in buffers",
            result1.keypoints.len(),
            result1.dropped_blobs,
            result1.dropped_features
        );
    } else {
        info!("Extracted {} keypoints", result1.keypoints.len());
    }

    let result2 = feats
        .detect_top_n(&img2_f32.as_ndarray2(), top_n, min_size)
        .unwrap();

    if result2.dropped_blobs > 0 || result2.dropped_features > 0 {
        info!(
            "Extracted {} keypoints. {} candidate blobs and {} keypoints did not fit in buffers",
            result2.keypoints.len(),
            result2.dropped_blobs,
            result2.dropped_features
        );
    } else {
        info!("Extracted {} keypoints", result2.keypoints.len());
    }

    let FeaturesResult {
        keypoints: kp1,
        descriptors: desc1,
        ..
    } = result1;
    let FeaturesResult {
        keypoints: kp2,
        descriptors: desc2,
        ..
    } = result2;

    let matches = match_features(&desc1.view(), &desc2.view());
    info!("Matching 1 -> 2: {} matches", matches.len(),);
    let matches_reverse = match_features(&desc2.view(), &desc1.view());
    info!("Matching 2 -> 1: {} matches", matches_reverse.len());

    let mut match_img = GrayImage::new(
        img1.width() + img2.width(),
        img1.height().max(img2.height()),
    );
    match_img.copy_from(&img1, 0, 0).unwrap();
    match_img.copy_from(&img2, img1.width(), 0).unwrap();
    for p in &kp1 {
        imageproc::drawing::draw_hollow_circle_mut(
            &mut match_img,
            (p.x as i32, p.y as i32),
            p.size as i32,
            image::Luma([255]),
        );
    }
    for p in &kp2 {
        imageproc::drawing::draw_hollow_circle_mut(
            &mut match_img,
            (img1.width() as i32 + p.x as i32, p.y as i32),
            p.size as i32,
            image::Luma([255]),
        );
    }
    for (i, j) in &matches {
        imageproc::drawing::draw_antialiased_line_segment_mut(
            &mut match_img,
            (kp1[*i].x as i32, kp1[*i].y as i32),
            (img1.width() as i32 + kp2[*j].x as i32, kp2[*j].y as i32),
            image::Luma([255]),
            imageproc::pixelops::interpolate,
        );
    }
    match_img
        .save(out_path)
        .expect("Failed to write output image");
    Ok(())
}
