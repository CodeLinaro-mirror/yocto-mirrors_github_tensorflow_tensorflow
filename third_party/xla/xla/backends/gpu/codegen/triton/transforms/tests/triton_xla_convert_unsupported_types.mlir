// RUN: xla-opt --split-input-file --convert-unsupported-types --canonicalize  %s | FileCheck %s

#indexing_map = #xla.indexing_map<"(pid)[k] -> (pid * 16 + k), domain: pid in [0, 15], k in [0, 15]">
#indexing_map1 = #xla.indexing_map<"(pid_0) -> ((pid_0 floordiv 64) * 16), domain: pid_0 in [0, 255]">
#indexing_map2 = #xla.indexing_map<"(pid_0) -> ((pid_0 mod 16) * 32), domain: pid_0 in [0, 255]">
#indexing_map3 = #xla.indexing_map<"(pid_0) -> (pid_0 mod 16), domain: pid_0 in [0, 255]">
#indexing_map4 = #xla.indexing_map<"(pid_0) -> (((pid_0 floordiv 16) mod 4) * 16), domain: pid_0 in [0, 255]">
#indexing_map5 = #xla.indexing_map<"(pid_0) -> ((pid_0 floordiv 4) * 16), domain: pid_0 in [0, 15]">
#indexing_map6 = #xla.indexing_map<"(pid_0) -> ((pid_0 mod 4) * 16), domain: pid_0 in [0, 15]">
module {
  // CHECK:   func.func @triton_fn(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<f8E4M3FN>, %arg3: !tt.ptr<i8>, %arg4: !tt.ptr<f32>) {
  func.func @triton_fn(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E8M0FNU>, %arg2: !tt.ptr<f8E4M3FN>, %arg3: !tt.ptr<f8E8M0FNU>, %arg4: !tt.ptr<f32>) {
    %0 = arith.constant 0 : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_cast %1 : i64 to index
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %3 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %cst) -> (tensor<16x16xf32>) {
      %6 = xla.apply_indexing #indexing_map(%2)[%arg5]
      %7 = xla.apply_indexing #indexing_map1(%6)
      // CHECK: %[[i_7:.*]] = xla.apply_indexing #indexing_map
      %8 = xla.apply_indexing #indexing_map2(%6)
      // CHECK: %[[i_8:.*]] = xla.apply_indexing #indexing_map
      %extracted_tile = triton_xla.extract from %arg0 as memref<64x512xf8E4M3FN, #triton_xla.layout<[1, 0]>> [%7, %8] [16, 32] [1, 1] : tensor<16x32xf8E4M3FN>
      // CHECK: %[[arg_0:.*]] = triton_xla.extract from %arg0 as memref<64x512xf8E4M3FN, #triton_xla.layout<[1, 0]>> [%[[i_7]], %[[i_8]]] [16, 32] [1, 1] : tensor<16x32xf8E4M3FN>
      %9 = xla.apply_indexing #indexing_map1(%6)
      // CHECK: %[[i_9:.*]] = xla.apply_indexing #indexing_map
      %10 = xla.apply_indexing #indexing_map3(%6)
      // CHECK: %[[i_10:.*]] = xla.apply_indexing #indexing_map
      %extracted_tile_0 = triton_xla.extract from %arg1 as memref<64x16xf8E8M0FNU, #triton_xla.layout<[1, 0]>> [%9, %10] [16, 1] [1, 1] : tensor<16x1xf8E8M0FNU>
      // CHECK: %[[arg_1:.*]] = triton_xla.extract from %arg1 as memref<64x16xi8, #triton_xla.layout<[1, 0]>> [%[[i_9]], %[[i_10]]] [16, 1] [1, 1] : tensor<16x1xi8>
      %11 = xla.apply_indexing #indexing_map2(%6)
      // CHECK: %[[i_11:.*]] = xla.apply_indexing #indexing_map
      %12 = xla.apply_indexing #indexing_map4(%6)
      // CHECK: %[[i_12:.*]] = xla.apply_indexing #indexing_map
      %extracted_tile_1 = triton_xla.extract from %arg2 as memref<512x64xf8E4M3FN, #triton_xla.layout<[1, 0]>> [%11, %12] [32, 16] [1, 1] : tensor<32x16xf8E4M3FN>
      // CHECK: %[[arg_2:.*]] = triton_xla.extract from %arg2 as memref<512x64xf8E4M3FN, #triton_xla.layout<[1, 0]>> [%[[i_11]], %[[i_12]]] [32, 16] [1, 1] : tensor<32x16xf8E4M3FN>
      %13 = xla.apply_indexing #indexing_map3(%6)
      // CHECK: %[[i_13:.*]] = xla.apply_indexing #indexing_map
      %14 = xla.apply_indexing #indexing_map4(%6)
      // CHECK: %[[i_14:.*]] = xla.apply_indexing #indexing_map
      %extracted_tile_2 = triton_xla.extract from %arg3 as memref<16x64xf8E8M0FNU, #triton_xla.layout<[1, 0]>> [%13, %14] [1, 16] [1, 1] : tensor<1x16xf8E8M0FNU>
      // CHECK: %[[arg_3:.*]] = triton_xla.extract from %arg3 as memref<16x64xi8, #triton_xla.layout<[1, 0]>> [%[[i_13]], %[[i_14]]] [1, 16] [1, 1] : tensor<1x16xi8>
      %15 = arith.index_cast %arg5 : index to i32
      %16 = arith.bitcast %extracted_tile_0 : tensor<16x1xf8E8M0FNU> to tensor<16x1xi8>
      %17 = arith.bitcast %extracted_tile_2 : tensor<1x16xf8E8M0FNU> to tensor<1x16xi8>
      %18 = tt.dot_scaled %extracted_tile scale %16, %extracted_tile_1 scale %17, %arg6 lhs = e4m3 rhs = e4m3 {fastMath = true} : tensor<16x32xf8E4M3FN>, tensor<16x1xi8> * tensor<32x16xf8E4M3FN>, tensor<1x16xi8> -> tensor<16x16xf32>
      scf.yield %18 : tensor<16x16xf32>
    }
    %4 = xla.apply_indexing #indexing_map5(%2)
    %5 = xla.apply_indexing #indexing_map6(%2)
    triton_xla.insert %3 into %arg4 as memref<64x64xf32, #triton_xla.layout<[1, 0]>> [%4, %5] [16, 16] [1, 1] : tensor<16x16xf32>
    return
  }
}