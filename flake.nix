{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    crane,
    rust-overlay,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        lib = pkgs.lib;

        buildTargets = {
          "x86_64-linux" = {
            crossSystemConfig = "x86_64-unknown-linux-gnu";
            rustTarget = "x86_64-unknown-linux-gnu";
          };
          "aarch64-linux" = {
            crossSystemConfig = "aarch64-unknown-linux-gnu";
            rustTarget = "aarch64-unknown-linux-gnu";
          };
        };
        cross-toolchain = p:
          p.rust-bin.stable.latest.minimal.override {
            targets = builtins.attrValues (builtins.mapAttrs (name: cfg: cfg.rustTarget) buildTargets);
          };

        buildForTarget = name: targetCfg: let
          targetPkgs = import nixpkgs {
            inherit system overlays;
            crossSystem = {
              config = targetCfg.crossSystemConfig;
            };
          };
          craneLib = (crane.mkLib targetPkgs).overrideToolchain (p: cross-toolchain p);
          src = craneLib.cleanCargoSource ./.;
          # TARGET_CC = "${targetPkgs.stdenv.cc}/bin/${targetPkgs.stdenv.cc.targetPrefix}cc";
          commonArgs = {
            version = "";
            doCheck = false;
            strictDeps = true;
            inherit src;
            buildInputs = with targetPkgs; [
              vulkan-loader
              vulkan-headers

              # for webcam example
              libv4l.dev
              libv4l
              linuxHeaders
            ];

            nativeBuildInputs = with pkgs; [
              vulkan-headers
              pkg-config
              shaderc
              llvmPackages.libclang.lib
              vulkan-headers
            ];

            VULKAN_SDK = "${targetPkgs.vulkan-headers}";
            SHADERC_LIB_DIR = "${pkgs.shaderc.lib}/lib";
            LINUX_HEADERS = "${pkgs.linuxHeaders}";

            CARGO_BUILD_TARGET = targetCfg.rustTarget;
            RUST_TARGET = targetCfg.rustTarget;
            CARGO_BUILD_RUSTFLAGS = [
              # "-C"
              # "linker=${TARGET_CC}"
              # "-C"
              # "link-args=-Wl,--dynamic-linker=/lib/ld-linux-aarch64.so.1"
            ];

            # For webcam example
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            BINDGEN_EXTRA_CLANG_ARGS = builtins.map (a: ''-I"${a}/include"'') [
              targetPkgs.glibc.dev
              targetPkgs.libv4l.dev
            ];
          };
          targetCargoArtifacts = craneLib.buildDepsOnly commonArgs;
          individualArgs =
            commonArgs
            // {
              cargoArtifacts = targetCargoArtifacts;
              doCheck = false;
              inherit (craneLib.crateNameFromCargoToml {inherit src;}) version;
            };
          fileSetForCrate = crate:
            lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./Cargo.lock
                ./Cargo.toml
                (lib.fileset.fileFilter (file: file.hasExt "glsl") ./.)
                (lib.fileset.fileFilter (file: file.hasExt "h") ./.)
                (lib.fileset.fileFilter (file: file.hasExt "safetensors") ./.)
                (craneLib.fileset.commonCargoSources ./.)
                (craneLib.fileset.commonCargoSources crate)
                # (craneLib.fileset.commonCargoSources ./examples)
                # (craneLib.fileset.commonCargoSources ./local_python)
              ];
            };

          vulkanSift = import ./benchmarks/VulkanSift.nix {inherit pkgs targetPkgs;};
        in {
          "match_images-${targetCfg.rustTarget}" = craneLib.buildPackage (
            individualArgs
            // {
              pname = "match_images";
              version = "no version";
              cargoExtraArgs = "-p match_images";

              src = fileSetForCrate ./examples/match_images;
            }
          );

          "webcam-${targetCfg.rustTarget}" = craneLib.buildPackage (
            individualArgs
            // {
              pname = "webcam";
              inherit (craneLib.crateNameFromCargoToml {inherit src;}) version;
              cargoExtraArgs = "-p webcam";

              src = fileSetForCrate ./examples/webcam;
              buildInputs = with targetPkgs; [
                libxkbcommon
                xorg.libxcb
                wayland
              ];
            }
          );

          "benchmarks-${targetCfg.rustTarget}" = craneLib.buildPackage (
            individualArgs
            // {
              pname = "benchmarks";
              cargoExtraArgs = "-p benchmarks";

              src = fileSetForCrate ./benchmarks;
              inherit (craneLib.crateNameFromCargoToml {inherit src;}) version;
              buildInputs = [vulkanSift];

              preConfigurePhase = ''
                export VULKANSIFT_LIB_PATH=${vulkanSift}/lib
                export VULKANSIFT_INCLUDE_PATH=${vulkanSift}/include
              '';
            }
          );
        };
        perTargetPackages = lib.mapAttrs buildForTarget buildTargets;
      in {
        packages = lib.concatMapAttrs (_: v: v) perTargetPackages;

        apps = {
          match_images = flake-utils.lib.mkApp {drv = perTargetPackages.${system}.match_images;};
        };

        devShells.default = let
          toolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-analyzer"
              "rust-src"
            ];
          };
          vulkanSift = import ./benchmarks/VulkanSift.nix {inherit pkgs;};
        in
          pkgs.mkShell {
            packages = with pkgs; [
              toolchain

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
              shader-slang
            ];

            buildInputs = [vulkanSift];

            env = {
              RUST_SRC_PATH = "${toolchain}/lib/rustlib/src/rust/library";
              VULKAN_SDK = "${pkgs.vulkan-headers}";
              SHADERC_LIB_DIR = "${pkgs.shaderc.lib}/lib";
              VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d:${pkgs.vulkan-tools-lunarg}/share/vulkan/explicit_layer.d";

              LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib"; # for v4l2 bindings, webcam example

              VULKANSIFT_LIB_PATH = "${vulkanSift}/lib";
              VULKANSIFT_INCLUDE_PATH = "${vulkanSift}/include";

              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
                shaderc
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
      }
    );
}
