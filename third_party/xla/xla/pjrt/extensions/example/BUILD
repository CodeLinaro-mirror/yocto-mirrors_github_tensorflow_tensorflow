load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "example_extension_cpp",
    srcs = ["example_extension_cpp.cc"],
    hdrs = ["example_extension_cpp.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "example_extension",
    srcs = ["example_extension.cc"],
    hdrs = ["example_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":example_extension_cpp",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_absl//absl/status",
    ],
)
