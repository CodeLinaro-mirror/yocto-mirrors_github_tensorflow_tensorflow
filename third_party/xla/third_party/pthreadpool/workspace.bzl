"""pthreadpool is a portable and efficient thread pool implementation."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "03524bb51d44e3f6c8cf3316de88cd8f4cec8ab5dde605451209a0051a5dd8c9",
        strip_prefix = "pthreadpool-bad85d364ff2be7a7fcf780f1fbaa7835d781c18",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/bad85d364ff2be7a7fcf780f1fbaa7835d781c18.zip"),
    )
