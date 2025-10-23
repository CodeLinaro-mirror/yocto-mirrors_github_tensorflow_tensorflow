"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "25d3d1ad1c5434f5b50d9693f239a581709e626365c261b537acbb6d9a916698",
        strip_prefix = "XNNPACK-240fc7757559ef783dced784754412ba3b35f981",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/240fc7757559ef783dced784754412ba3b35f981.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
