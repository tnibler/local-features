# Fast Local Image Features

1920x1080 camera feed matched on Ryzen 7840HS Laptop integrated GPU:

https://private-user-images.githubusercontent.com/62287652/494117038-29c6ac37-7042-4d93-bf20-4764125518f2.mp4

Vulkan-based local image feature detector, combining a DoG variant described by [Ghahremani et. al](https://arxiv.org/abs/2012.00859) and Multi Kernel Descriptors by [Mukundan et. al](https://arxiv.org/abs/1811.11147).

Very experimental. Project goals:

 - Be very fast.
 - Usable on as much hardware as possible (Raspberry Pi 4/5 would be a goal)
 - With decent detecting/matching performance

Roadmap:

 - Top-K filtering for features on GPU, right now done on CPU
 - Compatibility: Vulkan 1.2 with no extra device features (Raspberry Pi), ideally Vulkan 1.1
 - Fancy descriptor dimensionality reduction, something like https://arxiv.org/pdf/2209.13586
   - Matryoshka, binary, float8 etc
   - Choice between reproject+truncate only or nicer but slower MLP
 - Need bigger patch dataset, pipeline to create your own
   - Very fancy: ARKitScenes or known camera poses + DepthAnything for patch covisibility/viewpoint ground truth
   - or just ELOFTR/ROMA as reference point correspondences
 - CPU fallback implementation
 - float16 and other missing stuff
 - Maybe better keypoint orientation estimate (SIFT histogram is almost free and quite good). Something like https://arxiv.org/abs/1511.04273 at most, everything else is way too expensive.

## Benchmarks

Comparison with OpenCV SIFT (CPU), and Maël Aubert's [VulkanSIFT](https://github.com/maelaubert/VulkanSift). Both use default SIFT parameters, including the initial 2x upscale (generally required to get good results). These are not definitive bulletproof publication-ready measurements, but to give an idea.

Run on a Ryzen 7840HS laptop, 64GB DDR5 5600MHz.

<img src="./benchmarks/images/image_size.svg">

<img src="./benchmarks/images/feature_count.svg">

Quality benchmarks: no data, no claims.

## Build

Requirements for library (nix devshell also contains everything):

 - Rust
 - Vulkan SDK

## Examples

### Simple

Extract features from two images and draw matches:

`cargo run --release --bin match_images -- IMAGE1 IMAGE2 IMAGE_OUT`

### Webcam (Linux only)

Requires `video4linux`. 

`cargo run --release --bin webcam`

Pressing space will save the current video frame and extracted features (displayed on the right). Features are then matched between the camera feed and the saved image.

## License

This library is available under the GNU General Public License v3.
