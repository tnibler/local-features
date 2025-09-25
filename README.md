# Fast Local Image Features
https://private-user-images.githubusercontent.com/62287652/494117038-29c6ac37-7042-4d93-bf20-4764125518f2.mp4

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
