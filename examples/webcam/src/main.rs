use anyhow::{Context, Result, bail};
use eframe::{
    App, NativeOptions,
    egui::{self, CentralPanel, Pos2, Response, Stroke, TextureOptions, TopBottomPanel},
};
use local_features::{FeaturesResult, Keypoint, LocalFeaturesVulkan};
use ndarray::Array2;
use nshare::AsNdarray2 as _;
use v4l::context::Node;

mod camera;
use crate::camera::{CameraImage, CameraWithStream, available_devices, open_device};

const MAX_FEATURES: u32 = 5000;

fn make_local_features(width: u32, height: u32) -> Result<LocalFeaturesVulkan> {
    local_features::new_vulkan(
        local_features::BuildTimeParams {
            n_scales: 4,
            max_image_width: width,
            max_image_height: height,
            max_features: MAX_FEATURES,
            max_blobs: 4 * MAX_FEATURES,
            ..Default::default()
        },
        Default::default(),
    )
    .context("Error creating LocalFeaturesVulkan")
}

fn main() -> Result<()> {
    env_logger::init();

    let available_devices: Vec<_> = available_devices();
    if available_devices.is_empty() {
        bail!("No suitable video device found");
    }

    let selected_device = 0;
    let mut camera = open_device(available_devices[0].index())?;
    let img = camera.next_frame()?;

    let (width, height) = (img.gray.width() as u32, img.gray.height() as u32);
    let feats = make_local_features(width, height)?;

    let app = WebcamDemo {
        available_cameras: available_devices,
        camera,
        selected_camera_index: selected_device,
        changed_camera_index: None,
        match_image: None,
        local_features: feats,
        min_feature_size: 0.0,
        limit_features: 2000,
        resolution: (width, height),
    };
    let window_opts = NativeOptions {
        viewport: egui::ViewportBuilder::default().with_maximized(true),
        ..Default::default()
    };

    eframe::run_native(
        "Webcam Feature Matching",
        window_opts,
        Box::new(|_| Ok(Box::new(app))),
    )
    .unwrap();

    Ok::<(), anyhow::Error>(())
}

struct WebcamDemo {
    available_cameras: Vec<Node>,
    camera: CameraWithStream,
    selected_camera_index: usize,
    changed_camera_index: Option<usize>,

    match_image: Option<MatchImage>,
    local_features: local_features::LocalFeaturesVulkan,
    min_feature_size: f32,
    limit_features: u32,
    resolution: (u32, u32),
}

struct MatchImage {
    texture: egui::TextureHandle,
    index: usearch::Index,
    image: egui::ColorImage,
    keypoints: Vec<Keypoint>,
}

fn make_match_image(
    img: egui::ColorImage,
    texture: egui::TextureHandle,
    feats: FeaturesResult,
) -> MatchImage {
    let index = usearch::Index::new(&usearch::IndexOptions {
        dimensions: 128,
        metric: usearch::MetricKind::IP,
        quantization: usearch::ScalarKind::F32,
        multi: true,
        ..Default::default()
    })
    .unwrap();
    index.reserve(feats.descriptors.shape()[0]).unwrap();
    for (i, v) in feats.descriptors.axis_iter(ndarray::Axis(0)).enumerate() {
        let v = v.as_slice().unwrap();
        index.add(i as u64, v).unwrap();
    }
    MatchImage {
        texture,
        image: img,
        index,
        keypoints: feats.keypoints,
    }
}

impl App for WebcamDemo {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if let Some(index) = &self.changed_camera_index {
            let device_index = self.available_cameras[*index].index();
            self.camera = open_device(device_index).unwrap();
            self.changed_camera_index = None;
        }

        let CameraImage { rgb_egui, gray } = self.camera.next_frame().unwrap();
        let res = (gray.width(), gray.height());
        if res != self.resolution {
            self.local_features = make_local_features(res.0, res.1).unwrap();
            self.resolution = res;
            // self.match_image = None;
        }

        let texture = ctx.load_texture("frame", rgb_egui.clone(), TextureOptions::LINEAR);
        let img_arr = gray.as_ndarray2();
        let img_arr: Array2<f32> = img_arr.mapv(|v| v as f32 / 255.);
        let extract_start = std::time::Instant::now();
        let feature_result = self
            .local_features
            .detect_top_n(&img_arr.view(), self.limit_features, self.min_feature_size)
            .unwrap();
        let extract_time = extract_start.elapsed();
        let keypoints_this_frame = feature_result.keypoints.clone();

        let space_pressed = ctx.input(|input| {
            input.events.iter().any(|ev| match ev {
                egui::Event::Key {
                    key,
                    physical_key: _,
                    pressed,
                    repeat: _,
                    modifiers: _,
                } => *key == egui::Key::Space && !pressed,
                _ => false,
            })
        });

        TopBottomPanel::top("Top panel").show(ctx, |panel| {
            if space_pressed || panel.button("Snap").clicked() {
                self.match_image = Some(make_match_image(
                    rgb_egui.clone(),
                    texture.clone(),
                    feature_result.clone(),
                ));
            }
            let mut selected = self.selected_camera_index;
            panel.vertical(|ui| {
                egui::ComboBox::from_label("Select Camera")
                    .selected_text(format!("{:?}", selected))
                    .show_ui(ui, |ui| {
                        for (i, info) in self.available_cameras.iter().enumerate() {
                            ui.selectable_value(
                                &mut selected,
                                i,
                                info.name().unwrap_or(info.path().display().to_string()),
                            );
                        }
                    });
                if selected != self.selected_camera_index {
                    self.changed_camera_index = Some(selected.clone());
                }
                ui.label(format!(
                    "Format: {}x{}",
                    self.resolution.0, self.resolution.1
                ));
            });
        });

        TopBottomPanel::bottom("bottom panel").show(ctx, |panel| {
            panel.vertical(|ui| {
                let slider = egui::Slider::new(&mut self.min_feature_size, 0f32..=32.)
                    .text("Min. Feature Size");
                ui.add(slider);

                let slider = egui::Slider::new(&mut self.limit_features, 0..=MAX_FEATURES)
                    .text("Max. Features");
                ui.add(slider);
            });
            panel.label(format!(
                "Feature extraction time: {}ms",
                extract_time.as_millis()
            ));
        });

        CentralPanel::default().show(ctx, |panel| {
            let painter = egui::Painter::new(
                panel.ctx().clone(),
                egui::LayerId::new(egui::Order::Foreground, egui::Id::new("lines")),
                panel.clip_rect(),
            );

            panel.columns(2, |columns| {
                let img = egui::Image::new(&texture)
                    .maintain_aspect_ratio(true)
                    .max_width(columns[0].available_width());
                let img_width = gray.width();

                let Response { rect: rect_cam, .. } = columns[0].add(img);

                let cam_scale = rect_cam.width() / img_width as f32;
                for kp in &keypoints_this_frame {
                    painter.circle_stroke(
                        Pos2::new(
                            rect_cam.left() + kp.x * cam_scale,
                            rect_cam.top() + kp.y * cam_scale,
                        ),
                        kp.size * cam_scale,
                        Stroke::new(1., egui::Color32::WHITE),
                    );
                }

                if let Some(match_image) = self.match_image.as_ref() {
                    let img = egui::Image::new(&match_image.texture)
                        .maintain_aspect_ratio(true)
                        .max_width(columns[1].available_width());
                    let Response {
                        rect: rect_snap, ..
                    } = columns[1].add(img);

                    let snap_scale = rect_snap.width() / match_image.image.width() as f32;
                    for kp in &match_image.keypoints {
                        painter.circle_stroke(
                            Pos2::new(
                                rect_snap.left() + kp.x * snap_scale,
                                rect_snap.top() + kp.y * snap_scale,
                            ),
                            kp.size * snap_scale,
                            Stroke::new(1., egui::Color32::WHITE),
                        );
                    }

                    let FeaturesResult {
                        keypoints,
                        descriptors,
                        ..
                    } = feature_result;
                    for (kp, desc) in keypoints
                        .iter()
                        .zip(descriptors.axis_iter(ndarray::Axis(0)))
                    {
                        let matches = match_image
                            .index
                            .search(desc.as_slice().unwrap(), 2)
                            .unwrap();
                        if matches.distances[0] < matches.distances[1] * 0.75 {
                            let kp_match = match_image.keypoints[matches.keys[0] as usize];

                            painter.line_segment(
                                [
                                    Pos2::new(
                                        rect_cam.left() + kp.x * cam_scale,
                                        rect_cam.top() + kp.y * cam_scale,
                                    ),
                                    Pos2::new(
                                        rect_snap.left() + kp_match.x * snap_scale,
                                        rect_snap.top() + kp_match.y * snap_scale,
                                    ),
                                ],
                                Stroke::new(1., egui::Color32::WHITE),
                            );
                        }
                    }

                    painter.line_segment(
                        [
                            Pos2::new(rect_cam.left() + 10., rect_cam.top() + 20.),
                            Pos2::new(rect_snap.left() + 10., rect_snap.bottom() - 20.),
                        ],
                        Stroke::new(1., egui::Color32::WHITE),
                    );
                }
            });
        });
        ctx.request_repaint();
    }
}
