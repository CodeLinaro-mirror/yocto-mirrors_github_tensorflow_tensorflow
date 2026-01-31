"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-a3e59c3b9e264ac8cc4837a82c35f917cf992dac",
        sha256 = "62ba05e5f720c76c510d6ab2a77f8ccc17a76c5cea951bea47355a7dfa460449",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/a3e59c3b9e264ac8cc4837a82c35f917cf992dac.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
    )
