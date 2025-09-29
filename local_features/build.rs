fn main() {
    println!("cargo:rerun-if-changed=src/vulkan/shaders");
    println!("cargo:rerun-if-changed=src/vulkan/shaders_build");
    std::process::Command::new("make")
        .arg("-C")
        .arg("src/vulkan")
        .status()
        .expect("Failed to run make -C src/vulkan");
}
