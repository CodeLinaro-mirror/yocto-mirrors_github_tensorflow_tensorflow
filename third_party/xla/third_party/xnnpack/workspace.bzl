"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "9d6517ff955c6efc44be76566b5a75b1659a49060867a147cdbbc1a01bb821bc",
        strip_prefix = "XNNPACK-5e5d26e8f34e5fa14cdbeb6a9619c3d9bce33e02",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/5e5d26e8f34e5fa14cdbeb6a9619c3d9bce33e02.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
