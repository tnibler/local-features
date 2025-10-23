use std::ptr::null;

use image::buffer::ConvertBuffer as _;
use log::info;
use nshare::AsNdarray2 as _;

fn main() -> Result<(), ()> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "info");
        }
    }
    env_logger::init();

    let vulkan = match local_features::vulkan::Vulkan::new() {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Error creating Vulkan backend");
            eprintln!("{:?}", err);
            return Err(());
        }
    };

    let mut args = pico_args::Arguments::from_env();
    let mut do_renderdoc = if let Some("renderdoc") = args.subcommand().unwrap().as_deref() {
        let rd = renderdoc::RenderDoc::<renderdoc::V140>::new().unwrap();
        Some(rd)
    } else {
        None
    };
    let path: String = args.free_from_str().expect("need input image path");
    args.finish();

    let img = match image::open(path).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => {
            eprintln!("wrong image type");
            return Err(());
        }
    };
    let img_f32: image::ImageBuffer<image::Luma<f32>, Vec<f32>> = img.convert();

    let mut feats = local_features::new_vulkan(
        &vulkan,
        local_features::BuildTimeParams {
            n_scales: 5,
            max_image_width: img.width(),
            max_image_height: img.height(),
            max_features: 10000,
            max_blobs: 30000,
            ..Default::default()
        },
    )
    .unwrap();

    let start = std::time::Instant::now();
    if let Some(rd) = do_renderdoc.as_mut() {
        rd.start_frame_capture(null(), null())
    }

    let result = feats
        .detect_extract_all(&img_f32.as_ndarray2(), &Default::default())
        .unwrap();

    if let Some(rd) = do_renderdoc.as_mut() {
        rd.end_frame_capture(null(), null())
    }
    let time = start.elapsed();

    if result.dropped_blobs > 0 || result.dropped_features > 0 {
        info!(
            "Extracted {} keypoints in {:?}. {} candidate blobs and {} keypoints did not fit in buffers",
            result.keypoints.len(),
            time,
            result.dropped_blobs,
            result.dropped_features
        );
    } else {
        info!(
            "Extracted {} keypoints in {:?}",
            result.keypoints.len(),
            time
        );
    }
    Ok(())
}
