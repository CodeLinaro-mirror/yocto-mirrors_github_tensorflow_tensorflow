// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=partial-conversion=true %s | FileCheck %s --check-prefix CHECK-STABLEHLO-OP-UNCHANGED
// RUN: hlo-translate -mlir-to-hlo %s | FileCheck %s

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %Arg_0.1 = f32[4] parameter(0)
  // CHECK: %Arg_1.2 = f32[4] parameter(1)
  // CHECK: %add.3 = f32[4] add(%Arg_0.1, %Arg_1.2)
  // CHECK-STABLEHLO-OP-UNCHANGED: stablehlo.add
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
