fn main() {
    #[cfg(feature = "vulkansift")]
    {
        use std::env;
        use std::path::PathBuf;
        use std::str::FromStr;
        let vulkan_sift_include_path = std::env::var("VULKANSIFT_INCLUDE_PATH")
            .expect("VULKANSIFT_INCLUDE_PATH needs to be set");
        let vulkan_sift_lib_path =
            std::env::var("VULKANSIFT_LIB_PATH").expect("VULKANSIFT_LIB_PATH needs to be set");

        println!("cargo:rustc-link-search={}", vulkan_sift_lib_path);

        println!("cargo:rerun-if-env-changed=VULKANSIFT_INLUDE_PATH");
        println!("cargo:rerun-if-env-changed=VULKANSIFT_LIB_PATH");

        println!("cargo:rustc-link-lib=vulkansift");

        let bindings = bindgen::Builder::default()
            .header(
                PathBuf::from_str(&vulkan_sift_include_path)
                    .unwrap()
                    .join("vulkansift/vulkansift.h")
                    .to_str()
                    .unwrap(),
            )
            .clang_arg(format!("-I{}", &vulkan_sift_include_path))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings");
    }
}
