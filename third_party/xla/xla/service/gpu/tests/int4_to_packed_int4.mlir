// RUN: xla-opt --int4-to-packed-int4-rewrite %s --mlir-print-ir-after-all

module {
  tt.func @gemm_fusion_dot_2_0_impl(%arg0: !tt.ptr<i4> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %0 = tt.get_program_id x : i32
    %c16_i32 = arith.constant 16 : i32
    %1 = arith.divsi %0, %c16_i32 : i32
    %c8_i32 = arith.constant 8 : i32
    %2 = arith.muli %1, %c8_i32 : i32
    %c64_i32 = arith.constant 64 : i32
    %3 = arith.subi %c64_i32, %2 : i32
    %4 = arith.cmpi slt, %3, %c8_i32 : i32
    %5 = arith.select %4, %3, %c8_i32 : i32
    %6 = arith.remsi %0, %5 : i32
    %7 = arith.addi %2, %6 : i32
    %c16_i32_0 = arith.constant 16 : i32
    %8 = arith.remsi %0, %c16_i32_0 : i32
    %9 = arith.divsi %8, %5 : i32
    %c128_i32 = arith.constant 128 : i32
    %10 = arith.muli %7, %c128_i32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %11 = arith.addi %10, %c0_i32 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i32_1 = arith.constant 0 : i32
    %c8192_i64_2 = arith.constant 8192 : i64
    %c0_i32_3 = arith.constant 0 : i32
    %c8192_i64_4 = arith.constant 8192 : i64
    %c0_i32_5 = arith.constant 0 : i32
    %12 = arith.addi %c0_i32_3, %c0_i32_5 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i32_6 = arith.constant 0 : i32
    %c4096_i64_7 = arith.constant 4096 : i64
    %c33554432_i32 = arith.constant 33554432 : i32
    %13 = tt.get_program_id y : i32
    %c0_i32_8 = arith.constant 0 : i32
    %14 = arith.addi %c0_i32_8, %13 : i32
    %15 = arith.muli %14, %c33554432_i32 : i32
    %16 = tt.addptr %arg0, %15 : !tt.ptr<i4>, i32
    %17 = tt.make_tensor_ptr %16, [%c8192_i64_2, %c4096_i64_7], [%c1_i64, %c8192_i64_4], [%c0_i32_1, %c0_i32_6] {order = array<i32: 1, 0>} : <tensor<128x32xi4>>
    %18 = tt.advance %17, [%10, %c0_i32_3] : <tensor<128x32xi4>>
    %c128_i32_9 = arith.constant 128 : i32
    %19 = arith.muli %7, %c128_i32_9 : i32
    %c1_i64_10 = arith.constant 1 : i64
    %c0_i32_11 = arith.constant 0 : i32
    %20 = arith.addi %19, %c0_i32_11 : i32
    %c8192_i64_12 = arith.constant 8192 : i64
    %c0_i32_13 = arith.constant 0 : i32
    %c8192_i64_14 = arith.constant 8192 : i64
    %c8192_i32 = arith.constant 8192 : i32
    %21 = tt.get_program_id y : i32
    %c0_i32_15 = arith.constant 0 : i32
    %22 = arith.addi %c0_i32_15, %21 : i32
    %23 = arith.muli %22, %c8192_i32 : i32
    %24 = tt.addptr %arg1, %23 : !tt.ptr<bf16>, i32
    %25 = tt.make_tensor_ptr %24, [%c8192_i64_14], [%c1_i64_10], [%c0_i32_13] {order = array<i32: 0>} : <tensor<128xbf16>>
    %26 = tt.advance %25, [%19] : <tensor<128xbf16>>
    %c0_i32_16 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c0_i32_17 = arith.constant 0 : i32
    %27 = arith.addi %c0_i32_16, %c0_i32_17 : i32
    %c4096_i64_18 = arith.constant 4096 : i64
    %c0_i32_19 = arith.constant 0 : i32
    %c4096_i64_20 = arith.constant 4096 : i64
    %c128_i32_21 = arith.constant 128 : i32
    %28 = arith.muli %9, %c128_i32_21 : i32
    %c1_i64_22 = arith.constant 1 : i64
    %c0_i32_23 = arith.constant 0 : i32
    %29 = arith.addi %28, %c0_i32_23 : i32
    %c256_i64_24 = arith.constant 256 : i64
    %c0_i32_25 = arith.constant 0 : i32
    %c256_i64_26 = arith.constant 256 : i64
    %c1048576_i32 = arith.constant 1048576 : i32
    %30 = tt.get_program_id y : i32
    %c0_i32_27 = arith.constant 0 : i32
    %31 = arith.addi %c0_i32_27, %30 : i32
    %32 = arith.muli %31, %c1048576_i32 : i32
    %33 = tt.addptr %arg2, %32 : !tt.ptr<bf16>, i32
    %34 = tt.make_tensor_ptr %33, [%c4096_i64_20, %c256_i64_26], [%c256_i64, %c1_i64_22], [%c0_i32_19, %c0_i32_25] {order = array<i32: 1, 0>} : <tensor<32x128xbf16>>
    %35 = tt.advance %34, [%c0_i32_16, %28] : <tensor<32x128xbf16>>
    %c0_i32_28 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c32_i32 = arith.constant 32 : i32
    %36:4 = scf.for %arg4 = %c0_i32_28 to %c4096_i32 step %c32_i32 iter_args(%arg5 = %18, %arg6 = %26, %arg7 = %35, %arg8 = %cst) -> (!tt.ptr<tensor<128x32xi4>>, !tt.ptr<tensor<128xbf16>>, !tt.ptr<tensor<32x128xbf16>>, tensor<128x128xf32>)  : i32 {
      %48 = tt.load %arg5 : !tt.ptr<tensor<128x32xi4>>
      %c0_i32_42 = arith.constant 0 : i32
      %c32_i32_43 = arith.constant 32 : i32
      %49 = tt.advance %arg5, [%c0_i32_42, %c32_i32_43] : <tensor<128x32xi4>>
      %50 = tt.load %arg6 : !tt.ptr<tensor<128xbf16>>
      %c0_i32_44 = arith.constant 0 : i32
      %51 = tt.advance %arg6, [%c0_i32_44] : <tensor<128xbf16>>
      %52 = tt.load %arg7 : !tt.ptr<tensor<32x128xbf16>>
      %c32_i32_45 = arith.constant 32 : i32
      %c0_i32_46 = arith.constant 0 : i32
      %53 = tt.advance %arg7, [%c32_i32_45, %c0_i32_46] : <tensor<32x128xbf16>>
      %99 = arith.extsi %48 : tensor<128x32xi4> to tensor<128x32xi8>
      %54 = arith.sitofp %99 : tensor<128x32xi8> to tensor<128x32xf32>
      %55 = tt.expand_dims %50 {axis = 1 : i32} : tensor<128xbf16> -> tensor<128x1xbf16>
      %56 = tt.broadcast %55 : tensor<128x1xbf16> -> tensor<128x32xbf16>
      %57 = arith.extf %56 : tensor<128x32xbf16> to tensor<128x32xf32>
      %58 = arith.mulf %54, %57 : tensor<128x32xf32>
      %59 = arith.truncf %58 : tensor<128x32xf32> to tensor<128x32xbf16>
      %60 = tt.dot %59, %52, %arg8 : tensor<128x32xbf16> * tensor<32x128xbf16> -> tensor<128x128xf32>
      scf.yield %49, %51, %53, %60 : !tt.ptr<tensor<128x32xi4>>, !tt.ptr<tensor<128xbf16>>, !tt.ptr<tensor<32x128xbf16>>, tensor<128x128xf32>
    }
    %37 = arith.truncf %36#3 : tensor<128x128xf32> to tensor<128x128xbf16>
    %c128_i32_29 = arith.constant 128 : i32
    %38 = arith.muli %7, %c128_i32_29 : i32
    %c256_i64_30 = arith.constant 256 : i64
    %c0_i32_31 = arith.constant 0 : i32
    %39 = arith.addi %38, %c0_i32_31 : i32
    %c8192_i64_32 = arith.constant 8192 : i64
    %c0_i32_33 = arith.constant 0 : i32
    %c8192_i64_34 = arith.constant 8192 : i64
    %c128_i32_35 = arith.constant 128 : i32
    %40 = arith.muli %9, %c128_i32_35 : i32
    %c1_i64_36 = arith.constant 1 : i64
    %c0_i32_37 = arith.constant 0 : i32
    %41 = arith.addi %40, %c0_i32_37 : i32
    %c256_i64_38 = arith.constant 256 : i64
    %c0_i32_39 = arith.constant 0 : i32
    %c256_i64_40 = arith.constant 256 : i64
    %c2097152_i32 = arith.constant 2097152 : i32
    %42 = tt.get_program_id y : i32
    %c0_i32_41 = arith.constant 0 : i32
    %43 = arith.addi %c0_i32_41, %42 : i32
    %44 = arith.muli %43, %c2097152_i32 : i32
    %45 = tt.addptr %arg3, %44 : !tt.ptr<bf16>, i32
    %46 = tt.make_tensor_ptr %45, [%c8192_i64_34, %c256_i64_40], [%c256_i64_30, %c1_i64_36], [%c0_i32_33, %c0_i32_39] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
    %47 = tt.advance %46, [%38, %40] : <tensor<128x128xbf16>>
    tt.store %47, %37 : !tt.ptr<tensor<128x128xbf16>>
    tt.return
  }
}