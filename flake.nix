{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [inputs.rust-overlay.overlays.default];
      };

      rust = pkgs.rust-bin.nightly.latest.default.override {
        extensions = ["rust-src" "rustfmt" "rust-analyzer"];
      };
    in rec {
      devShells = {
        default = pkgs.mkShell {
          packages = with pkgs; [
            rust
            vulkan-tools
            vulkan-tools-lunarg
            vulkan-validation-layers
            vulkan-extension-layer
            vulkan-loader
            glsl_analyzer
            shaderc
            maturin
            cargo-criterion

            xorg.libxcb
            xorg.libXau
            xorg.libXdmcp
            libxkbcommon
          ];

          env = {
            RUST_SRC_PATH = "${rust}/lib/rustlib/src/rust/library";
            VULKAN_SDK = "${pkgs.vulkan-headers}";
            SHADERC_LIB_DIR = "${pkgs.shaderc.lib}/lib";
            VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d:${pkgs.vulkan-tools-lunarg}/share/vulkan/explicit_layer.d";

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib"; # for v4l2 bindings, webcam example

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
              vulkan-loader
              vulkan-validation-layers
              vulkan-tools-lunarg
              xorg.libxcb
              xorg.libXdmcp
              xorg.libXau
              libxkbcommon
            ]);
          };
        };
      };
    });
}
