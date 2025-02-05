// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=80 2>&1 | FileCheck %s --check-prefix=CHECK-AMPERE

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f16_to_f8_dot_operand(%f16_inp: tensor<32x32xf16, #dot_operand>) {
    // CHECK-LABEL: @f16_to_f8_dot_operand

    %f8 = tt.fp_to_fp %f16_inp, rounding = rtne : tensor<32x32xf16, #dot_operand> -> tensor<32x32xf8E5M2, #dot_operand>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 3072 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ampere_s8_to_fp16_conversion_opIdx1(%1 : tensor<16x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) attributes {noinline = false} {
    // CHECK-AMPERE-LABEL: ampere_s8_to_fp16_conversion_opIdx1
    // CHECK-AMPERE: llvm.sitofp %{{.*}} : i8 to f16
    %2 = arith.sitofp %1 : tensor<16x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> to tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 3072 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ampere_s8_to_fp16_conversion_opIdx0(%1 : tensor<32x16xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>) attributes {noinline = false} {
    // CHECK-AMPERE-LABEL: @ampere_s8_to_fp16_conversion_opIdx0
    // CHECK-AMPERE: llvm.sitofp %{{.*}} : i8 to f16
    %2 = arith.sitofp %1 : tensor<32x16xi8, #ttg.dot_op<{opIdx = 0 , parent = #mma, kWidth = 4}>> to tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}
