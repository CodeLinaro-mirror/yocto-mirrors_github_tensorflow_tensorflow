/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

include "mlir/Pass/PassBase.td"

def FlattenTensorsPass : Pass<"xla-gpu-flatten-tensors", "mlir::ModuleOp"> {
  let summary = "Flatten tensors.";

  let description = [{
    Linearizes all tensors loads and stores.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "xla::gpu::XlaGpuDialect",
    "xla::XlaDialect",
  ];
  let constructor = "CreateFlattenTensorsPass()";
}

def ExpandFloatOpsPass : Pass<"xla-gpu-expand-float-ops", "mlir::ModuleOp"> {
  let summary = "Expands float ops that are not natively supported.";

  let description = [{
     Not all float ops are natively supported, either because they don't exist
     in hardware or they are too inaccurate.

     This pass replaces these ops with alternative implementations.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::math::MathDialect",
    "mlir::mhlo::MhloDialect"
  ];

  let constructor = "CreateExpandFloatOpsPass()";
}
