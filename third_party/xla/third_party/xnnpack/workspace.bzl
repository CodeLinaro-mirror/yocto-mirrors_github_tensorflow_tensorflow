"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "af44b5fa37dc2013f1ce0a98497753fca8b74b40a0470627bd2c38fce853b970",
        strip_prefix = "XNNPACK-a94de2f72f1336fe1c381b7c26318d3536f9e6b5",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/a94de2f72f1336fe1c381b7c26318d3536f9e6b5.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
