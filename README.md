# Fast Local Image Features

1920x1080 camera feed matched on Ryzen 7840HS Laptop integrated GPU:

https://private-user-images.githubusercontent.com/62287652/494117038-29c6ac37-7042-4d93-bf20-4764125518f2.mp4

Still experimental.

Project goals:

 - Be very fast.
 - Usable on as much hardware as possible
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

## Build

Non-linux might break any time. `remove-linuxonly` branch definitely works for the `match_images` example.

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
