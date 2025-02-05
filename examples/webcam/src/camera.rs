use anyhow::{Context, Result, anyhow, bail};
use eframe::epaint::ColorImage;
use image::{DynamicImage, GrayImage};
use log::{error, info};
use ouroboros::self_referencing;
use v4l::{
    Device, FourCC, buffer,
    context::{Node, enum_devices},
    io::traits::CaptureStream,
    prelude::{MmapStream, UserptrStream},
    video::Capture,
};

pub struct CameraImage {
    pub rgb_egui: ColorImage,
    pub gray: GrayImage,
}

pub enum CameraWithStream {
    UserPtr(Device, UserptrStream),
    Mmap(CameraMmapStream),
}

impl CameraWithStream {
    pub fn next_frame(&mut self) -> Result<CameraImage> {
        let frame = match self {
            CameraWithStream::UserPtr(_device, stream) => stream.next(),
            CameraWithStream::Mmap(camera_mmap_stream) => {
                camera_mmap_stream.with_stream_mut(|stream| stream.next())
            }
        }
        .context("Error reading frame from video device")?;

        let img = image::load_from_memory_with_format(frame.0, image::ImageFormat::Jpeg)
            .context("Error decoding frame")?;
        let size = [img.width() as _, img.height() as _];
        let egui_img =
            ColorImage::from_rgba_unmultiplied(size, img.to_rgba8().as_flat_samples().as_slice());
        let gray_img = match img.grayscale() {
            DynamicImage::ImageLuma8(img) => img,
            _ => anyhow::bail!("wrong image type"),
        };
        Ok(CameraImage {
            rgb_egui: egui_img,
            gray: gray_img,
        })
    }
}

#[self_referencing]
pub struct CameraMmapStream {
    device: Device,
    #[borrows(device)]
    #[covariant]
    stream: MmapStream<'this>,
}

pub fn open_device(index: usize) -> Result<CameraWithStream> {
    let device = Device::new(index).context("Error opening video device.")?;
    let mut format_want = device.format()?;
    format_want.fourcc = FourCC::new(b"MJPG");
    let format_have = device.set_format(&format_want)?;
    if format_have.fourcc != format_want.fourcc {
        bail!("Device {index} does not support MJPG");
    }

    let user_ptr_error = match UserptrStream::new(&device, buffer::Type::VideoCapture) {
        Ok(stream) => {
            return Ok(CameraWithStream::UserPtr(device, stream));
        }
        Err(err) => err,
    };
    let mmap_error = match MmapStream::new(&device, buffer::Type::VideoCapture) {
        Ok(stream) => {
            let cs = CameraMmapStreamBuilder {
                device,
                stream_builder: move |_dev: &Device| stream,
            }
            .build();
            return Ok(CameraWithStream::Mmap(cs));
        }
        Err(err) => err,
    };
    error!(
        "Device {index}: UserPtrStream failed\n{user_ptr_error})\nMmapStream failed\n{mmap_error}"
    );
    Err(anyhow!("Could not open stream for device"))
}

pub fn available_devices() -> Vec<Node> {
    enum_devices()
        .into_iter()
        .filter(|node| {
            let res = open_device(node.index());
            if res.is_err() {
                info!(
                    "Device {} '{}' is no good",
                    node.path().display(),
                    node.name().unwrap_or(String::new())
                );
            }
            res.is_ok()
        })
        .collect()
}
