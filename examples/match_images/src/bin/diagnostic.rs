use std::ptr::null;

use image::{GrayImage, Luma, buffer::ConvertBuffer as _};
use imageproc::definitions::Image;
use log::info;
use ndarray::{Array2, Axis, s};
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
    let command = args.subcommand().unwrap().expect("need subcommand");
    let mut do_renderdoc = if command == "renderdoc" {
        let rd = renderdoc::RenderDoc::<renderdoc::V140>::new().unwrap();
        Some(rd)
    } else {
        None
    };
    let do_patches = command == "patches";
    let input_path: String = args.free_from_str().expect("need input image path");
    let out_path: Option<String> = if command == "patches" {
        Some(args.free_from_str().expect("need output image path"))
    } else {
        None
    };
    args.finish();

    let img = match image::open(input_path).unwrap().grayscale() {
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
            debug_readout_patches: do_patches,
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

    if do_patches {
        let patches = feats.debug_read_patches().unwrap();
        let n_patches = result.keypoints.len();

        let ps = 32;
        let grid_side = (n_patches as f32).sqrt().ceil() as usize;
        let width = grid_side * ps;
        let height = grid_side * ps;

        let mut patch_image = Array2::<f32>::zeros((height, width));

        for (i, patch) in patches.axis_iter(Axis(0)).take(n_patches).enumerate() {
            let r = i / grid_side;
            let c = i % grid_side;

            let start_row = r * ps;
            let end_row = start_row + ps;
            let start_col = c * ps;
            let end_col = start_col + ps;

            let mut slice = patch_image.slice_mut(s![start_row..end_row, start_col..end_col]);
            slice.assign(&patch);
        }
        // patch_image *= 255.0;
        let patch_image: Image<Luma<f32>> = Image::from_raw(
            width as u32,
            height as u32,
            patch_image.into_raw_vec_and_offset().0,
        )
        .unwrap();
        let patch_image: GrayImage = patch_image.convert();
        patch_image.save(out_path.unwrap()).unwrap();
    }

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
