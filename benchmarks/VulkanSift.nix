{
  pkgs,
  targetPkgs ? pkgs,
}: let
  vulkanSiftSrc = pkgs.fetchFromGitHub {
    owner = "maelaubert";
    repo = "VulkanSift";
    rev = "04245a5e194c9c1aa2ac07c2d257eedad3c8699d";
    hash = "sha256-Dq20pMf/16FUvCkNIOFRQSP0ZXAsCQ7Pt8JIs05ILK4=";
  };
in
  targetPkgs.stdenv.mkDerivation {
    pname = "VulkanSift";
    version = "0.1";
    src = vulkanSiftSrc;

    nativeBuildInputs = with pkgs; [cmake pkg-config shaderc python310];

    buildInputs = with targetPkgs; [
      vulkan-loader
      vulkan-headers
      xorg.libX11.dev
    ];

    cmakeFlags = ["-DBUILD_SHARED_LIBRARY=off"];
  }
