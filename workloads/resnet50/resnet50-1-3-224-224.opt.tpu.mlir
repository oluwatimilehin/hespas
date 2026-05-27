module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_1.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<1x2048x7x7xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<1x2048x7x7xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>) -> tensor<2048xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x2048x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<2048xf32>
    %4 = stablehlo.rsqrt %3 : tensor<2048xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x2048x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x2048x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x2048x7x7xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x2048x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2048x7x7xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x2048x7x7xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2048x7x7xf32>) -> tensor<2048x49xf32>
    %15 = stablehlo.reduce(%14 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<2048x49xf32>, tensor<f32>) -> tensor<2048xf32>
    return %15 : tensor<2048xf32>
  }
  func.func private @region_1.14.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<1000x2048xf32>, %arg1: tensor<2048xf32>) -> tensor<1000xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<2048xf32>) -> tensor<1x2048xf32>
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x2048xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x2048xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x2048xf32>) -> tensor<2048xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<2048xf32>) -> tensor<1000x2048xf32>
    %5 = stablehlo.multiply %4, %arg0 : tensor<1000x2048xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<1000x2048xf32>, tensor<f32>) -> tensor<1000xf32>
    return %6 : tensor<1000xf32>
  }
  func.func private @fused_maximum(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x56x56xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<1x256x56x56xf32>, %arg7: tensor<256xf32>, %arg8: tensor<256xf32>, %arg9: tensor<256xf32>) -> tensor<1x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<1x256x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg8, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x56x56xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %12 = stablehlo.subtract %arg2, %11 : tensor<1x256x56x56xf32>
    %13 = stablehlo.add %arg9, %2 : tensor<256xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %16 = stablehlo.multiply %12, %15 : tensor<1x256x56x56xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x256x56x56xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x256x56x56xf32>
    %21 = stablehlo.add %10, %20 : tensor<1x256x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x56x56xf32>
    %23 = stablehlo.maximum %21, %22 : tensor<1x256x56x56xf32>
    return %23 : tensor<1x256x56x56xf32>
  }
  func.func private @fused_maximum.1(%arg0: tensor<1x256x56x56xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1x256x56x56xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>) -> tensor<1x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x256x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x56x56xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x256x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x56x56xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x256x56x56xf32>
    return %13 : tensor<1x256x56x56xf32>
  }
  func.func private @fused_maximum.2(%arg0: tensor<1x256x56x56xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1x256x56x56xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>) -> tensor<1x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x256x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x56x56xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x256x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x56x56xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x256x56x56xf32>
    return %13 : tensor<1x256x56x56xf32>
  }
  func.func private @fused_maximum.3(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x28x28xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<1x512x28x28xf32>, %arg7: tensor<512xf32>, %arg8: tensor<512xf32>, %arg9: tensor<512xf32>) -> tensor<1x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg8, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x28x28xf32>
    %11 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %12 = stablehlo.subtract %arg6, %11 : tensor<1x512x28x28xf32>
    %13 = stablehlo.add %arg9, %2 : tensor<512xf32>
    %14 = stablehlo.rsqrt %13 : tensor<512xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %16 = stablehlo.multiply %12, %15 : tensor<1x512x28x28xf32>
    %17 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x512x28x28xf32>
    %19 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x512x28x28xf32>
    %21 = stablehlo.add %10, %20 : tensor<1x512x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x28x28xf32>
    %23 = stablehlo.maximum %21, %22 : tensor<1x512x28x28xf32>
    return %23 : tensor<1x512x28x28xf32>
  }
  func.func private @fused_maximum.4(%arg0: tensor<1x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1x512x28x28xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>) -> tensor<1x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x512x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x28x28xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x512x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x28x28xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x512x28x28xf32>
    return %13 : tensor<1x512x28x28xf32>
  }
  func.func private @fused_maximum.5(%arg0: tensor<1x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1x512x28x28xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>) -> tensor<1x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x512x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x28x28xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x512x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x28x28xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x512x28x28xf32>
    return %13 : tensor<1x512x28x28xf32>
  }
  func.func private @fused_maximum.6(%arg0: tensor<1x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1x512x28x28xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>) -> tensor<1x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x512x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x28x28xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x512x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x28x28xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x512x28x28xf32>
    return %13 : tensor<1x512x28x28xf32>
  }
  func.func private @fused_maximum.7(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1x1024x14x14xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1x1024x14x14xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<1024xf32>, %arg9: tensor<1024xf32>) -> tensor<1x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<1x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg8, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x1024x14x14xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %12 = stablehlo.subtract %arg2, %11 : tensor<1x1024x14x14xf32>
    %13 = stablehlo.add %arg9, %2 : tensor<1024xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %16 = stablehlo.multiply %12, %15 : tensor<1x1024x14x14xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x1024x14x14xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x1024x14x14xf32>
    %21 = stablehlo.add %10, %20 : tensor<1x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
    %23 = stablehlo.maximum %21, %22 : tensor<1x1024x14x14xf32>
    return %23 : tensor<1x1024x14x14xf32>
  }
  func.func private @fused_maximum.8(%arg0: tensor<1x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<1x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x1024x14x14xf32>
    return %13 : tensor<1x1024x14x14xf32>
  }
  func.func private @fused_maximum.9(%arg0: tensor<1x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<1x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x1024x14x14xf32>
    return %13 : tensor<1x1024x14x14xf32>
  }
  func.func private @fused_maximum.10(%arg0: tensor<1x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<1x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x1024x14x14xf32>
    return %13 : tensor<1x1024x14x14xf32>
  }
  func.func private @fused_maximum.11(%arg0: tensor<1x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<1x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x1024x14x14xf32>
    return %13 : tensor<1x1024x14x14xf32>
  }
  func.func private @fused_maximum.12(%arg0: tensor<1x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<1x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x1024x14x14xf32>
    return %13 : tensor<1x1024x14x14xf32>
  }
  func.func private @fused_maximum.13(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
  func.func private @fused_maximum.14(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
  func.func private @fused_maximum.15(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
  func.func private @fused_maximum.16(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
  func.func private @fused_maximum.17(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
  func.func private @fused_maximum.18(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
  func.func private @region_0.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_window(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x112x112xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x112x112xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x112x112xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x112x112xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x112x112xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x112x112xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x64x112x112xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %13 = "stablehlo.reduce_window"(%12, %cst_1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %14 = stablehlo.maximum %arg5, %arg6 : tensor<f32>
      stablehlo.return %14 : tensor<f32>
    }) : (tensor<1x64x112x112xf32>, tensor<f32>) -> tensor<1x64x56x56xf32>
    return %13 : tensor<1x64x56x56xf32>
  }
  func.func private @fused_maximum.19(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x56x56xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x128x56x56xf32>
    return %12 : tensor<1x128x56x56xf32>
  }
  func.func private @fused_maximum.20(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x28x28xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x128x28x28xf32>
    return %12 : tensor<1x128x28x28xf32>
  }
  func.func private @fused_maximum.21(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x28x28xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x128x28x28xf32>
    return %12 : tensor<1x128x28x28xf32>
  }
  func.func private @fused_maximum.22(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x28x28xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x128x28x28xf32>
    return %12 : tensor<1x128x28x28xf32>
  }
  func.func private @fused_maximum.23(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x28x28xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x128x28x28xf32>
    return %12 : tensor<1x128x28x28xf32>
  }
  func.func private @fused_maximum.24(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x28x28xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x128x28x28xf32>
    return %12 : tensor<1x128x28x28xf32>
  }
  func.func private @fused_maximum.25(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x28x28xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x128x28x28xf32>
    return %12 : tensor<1x128x28x28xf32>
  }
  func.func private @fused_maximum.26(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x28x28xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x128x28x28xf32>
    return %12 : tensor<1x128x28x28xf32>
  }
  func.func private @fused_maximum.27(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.28(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.29(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.30(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.31(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.32(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.33(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.34(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.35(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.36(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.37(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x14x14xf32>
    return %12 : tensor<1x256x14x14xf32>
  }
  func.func private @fused_maximum.38(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x28x28xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x256x28x28xf32>
    return %12 : tensor<1x256x28x28xf32>
  }
  func.func private @fused_maximum.39(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<1x2048x7x7xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<1x2048x7x7xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<2048xf32>, %arg9: tensor<2048xf32>) -> tensor<1x2048x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x2048x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %3 = stablehlo.add %arg8, %2 : tensor<2048xf32>
    %4 = stablehlo.rsqrt %3 : tensor<2048xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x2048x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x2048x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x2048x7x7xf32>
    %11 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %12 = stablehlo.subtract %arg6, %11 : tensor<1x2048x7x7xf32>
    %13 = stablehlo.add %arg9, %2 : tensor<2048xf32>
    %14 = stablehlo.rsqrt %13 : tensor<2048xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %16 = stablehlo.multiply %12, %15 : tensor<1x2048x7x7xf32>
    %17 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x2048x7x7xf32>
    %19 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x2048x7x7xf32>
    %21 = stablehlo.add %10, %20 : tensor<1x2048x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2048x7x7xf32>
    %23 = stablehlo.maximum %21, %22 : tensor<1x2048x7x7xf32>
    return %23 : tensor<1x2048x7x7xf32>
  }
  func.func private @fused_maximum.40(%arg0: tensor<1x2048x7x7xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<1x2048x7x7xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>) -> tensor<1x2048x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x2048x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<2048xf32>
    %4 = stablehlo.rsqrt %3 : tensor<2048xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x2048x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x2048x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x2048x7x7xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<1x2048x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2048x7x7xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<1x2048x7x7xf32>
    return %13 : tensor<1x2048x7x7xf32>
  }
  func.func private @fused_maximum.41(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x14x14xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x512x14x14xf32>
    return %12 : tensor<1x512x14x14xf32>
  }
  func.func private @fused_maximum.42(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x512x7x7xf32>
    return %12 : tensor<1x512x7x7xf32>
  }
  func.func private @fused_maximum.43(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x512x7x7xf32>
    return %12 : tensor<1x512x7x7xf32>
  }
  func.func private @fused_maximum.44(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x512x7x7xf32>
    return %12 : tensor<1x512x7x7xf32>
  }
  func.func private @fused_maximum.45(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x512x7x7xf32>
    return %12 : tensor<1x512x7x7xf32>
  }
  func.func private @fused_maximum.46(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x512x7x7xf32>
    return %12 : tensor<1x512x7x7xf32>
  }
  func.func private @wrapped_add_computation(%arg0: tensor<1000xf32>, %arg1: tensor<1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func.func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64x1x1xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<64x64x1x1xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<256x64x1x1xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<64x256x1x1xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64x64x3x3xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<256x64x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<64x256x1x1xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<64x64x3x3xf32>, %arg28: tensor<64xf32>, %arg29: tensor<64xf32>, %arg30: tensor<256x64x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<512x256x1x1xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<128x256x1x1xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128x128x3x3xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<512x128x1x1xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<128x512x1x1xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128x128x3x3xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<512x128x1x1xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512xf32>, %arg54: tensor<128x512x1x1xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128x128x3x3xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<512x128x1x1xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512xf32>, %arg63: tensor<128x512x1x1xf32>, %arg64: tensor<128xf32>, %arg65: tensor<128xf32>, %arg66: tensor<128x128x3x3xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<512x128x1x1xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<1024x512x1x1xf32>, %arg73: tensor<1024xf32>, %arg74: tensor<1024xf32>, %arg75: tensor<256x512x1x1xf32>, %arg76: tensor<256xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x256x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<256xf32>, %arg81: tensor<1024x256x1x1xf32>, %arg82: tensor<1024xf32>, %arg83: tensor<1024xf32>, %arg84: tensor<256x1024x1x1xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256x256x3x3xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<1024x256x1x1xf32>, %arg91: tensor<1024xf32>, %arg92: tensor<1024xf32>, %arg93: tensor<256x1024x1x1xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256x256x3x3xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<1024x256x1x1xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<1024xf32>, %arg102: tensor<256x1024x1x1xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256x256x3x3xf32>, %arg106: tensor<256xf32>, %arg107: tensor<256xf32>, %arg108: tensor<1024x256x1x1xf32>, %arg109: tensor<1024xf32>, %arg110: tensor<1024xf32>, %arg111: tensor<256x1024x1x1xf32>, %arg112: tensor<256xf32>, %arg113: tensor<256xf32>, %arg114: tensor<256x256x3x3xf32>, %arg115: tensor<256xf32>, %arg116: tensor<256xf32>, %arg117: tensor<1024x256x1x1xf32>, %arg118: tensor<1024xf32>, %arg119: tensor<1024xf32>, %arg120: tensor<256x1024x1x1xf32>, %arg121: tensor<256xf32>, %arg122: tensor<256xf32>, %arg123: tensor<256x256x3x3xf32>, %arg124: tensor<256xf32>, %arg125: tensor<256xf32>, %arg126: tensor<1024x256x1x1xf32>, %arg127: tensor<1024xf32>, %arg128: tensor<1024xf32>, %arg129: tensor<2048x1024x1x1xf32>, %arg130: tensor<2048xf32>, %arg131: tensor<2048xf32>, %arg132: tensor<512x1024x1x1xf32>, %arg133: tensor<512xf32>, %arg134: tensor<512xf32>, %arg135: tensor<512x512x3x3xf32>, %arg136: tensor<512xf32>, %arg137: tensor<512xf32>, %arg138: tensor<2048x512x1x1xf32>, %arg139: tensor<2048xf32>, %arg140: tensor<2048xf32>, %arg141: tensor<512x2048x1x1xf32>, %arg142: tensor<512xf32>, %arg143: tensor<512xf32>, %arg144: tensor<512x512x3x3xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<2048x512x1x1xf32>, %arg148: tensor<2048xf32>, %arg149: tensor<2048xf32>, %arg150: tensor<512x2048x1x1xf32>, %arg151: tensor<512xf32>, %arg152: tensor<512xf32>, %arg153: tensor<512x512x3x3xf32>, %arg154: tensor<512xf32>, %arg155: tensor<512xf32>, %arg156: tensor<2048x512x1x1xf32>, %arg157: tensor<2048xf32>, %arg158: tensor<2048xf32>, %arg159: tensor<1000x2048xf32>, %arg160: tensor<1000xf32>, %arg161: tensor<64xf32>, %arg162: tensor<64xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<64xf32>, %arg166: tensor<64xf32>, %arg167: tensor<64xf32>, %arg168: tensor<64xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<64xf32>, %arg174: tensor<64xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<64xf32>, %arg178: tensor<64xf32>, %arg179: tensor<64xf32>, %arg180: tensor<64xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512xf32>, %arg184: tensor<512xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<128xf32>, %arg188: tensor<128xf32>, %arg189: tensor<512xf32>, %arg190: tensor<512xf32>, %arg191: tensor<128xf32>, %arg192: tensor<128xf32>, %arg193: tensor<128xf32>, %arg194: tensor<128xf32>, %arg195: tensor<512xf32>, %arg196: tensor<512xf32>, %arg197: tensor<128xf32>, %arg198: tensor<128xf32>, %arg199: tensor<128xf32>, %arg200: tensor<128xf32>, %arg201: tensor<512xf32>, %arg202: tensor<512xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<512xf32>, %arg208: tensor<512xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1024xf32>, %arg211: tensor<256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<1024xf32>, %arg216: tensor<1024xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<1024xf32>, %arg222: tensor<1024xf32>, %arg223: tensor<256xf32>, %arg224: tensor<256xf32>, %arg225: tensor<256xf32>, %arg226: tensor<256xf32>, %arg227: tensor<1024xf32>, %arg228: tensor<1024xf32>, %arg229: tensor<256xf32>, %arg230: tensor<256xf32>, %arg231: tensor<256xf32>, %arg232: tensor<256xf32>, %arg233: tensor<1024xf32>, %arg234: tensor<1024xf32>, %arg235: tensor<256xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<256xf32>, %arg239: tensor<1024xf32>, %arg240: tensor<1024xf32>, %arg241: tensor<256xf32>, %arg242: tensor<256xf32>, %arg243: tensor<256xf32>, %arg244: tensor<256xf32>, %arg245: tensor<1024xf32>, %arg246: tensor<1024xf32>, %arg247: tensor<2048xf32>, %arg248: tensor<2048xf32>, %arg249: tensor<512xf32>, %arg250: tensor<512xf32>, %arg251: tensor<512xf32>, %arg252: tensor<512xf32>, %arg253: tensor<2048xf32>, %arg254: tensor<2048xf32>, %arg255: tensor<512xf32>, %arg256: tensor<512xf32>, %arg257: tensor<512xf32>, %arg258: tensor<512xf32>, %arg259: tensor<2048xf32>, %arg260: tensor<2048xf32>, %arg261: tensor<512xf32>, %arg262: tensor<512xf32>, %arg263: tensor<512xf32>, %arg264: tensor<512xf32>, %arg265: tensor<2048xf32>, %arg266: tensor<2048xf32>, %arg267: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = stablehlo.custom_call @__cudnn$convForward(%arg267, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,112,112]{3,2,1,0}, u8[0]{0})"} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tuple<tensor<1x64x112x112xf32>, tensor<0xui8>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<1x64x112x112xf32>, tensor<0xui8>>) -> tensor<1x64x112x112xf32>
    %2 = "mhlo.fusion"(%arg2, %arg1, %1, %arg161, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<1x64x112x112xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<1x64x112x112xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x64x112x112xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<64xf32>
      %162 = stablehlo.rsqrt %161 : tensor<64xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<64xf32>) -> tensor<1x64x112x112xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x64x112x112xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<1x64x112x112xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x64x112x112xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<1x64x112x112xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x64x112x112xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x112x112xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x64x112x112xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %171 = "stablehlo.reduce_window"(%170, %cst_1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
      ^bb0(%arg273: tensor<f32>, %arg274: tensor<f32>):
        %172 = stablehlo.maximum %arg273, %arg274 : tensor<f32>
        stablehlo.return %172 : tensor<f32>
      }) : (tensor<1x64x112x112xf32>, tensor<f32>) -> tensor<1x64x56x56xf32>
      mhlo.return %171 : tensor<1x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %3 = stablehlo.custom_call @__cudnn$convForward(%2, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<1x256x56x56xf32>, tensor<0xui8>>
    %4 = stablehlo.get_tuple_element %3[0] : (tuple<tensor<1x256x56x56xf32>, tensor<0xui8>>) -> tensor<1x256x56x56xf32>
    %5 = stablehlo.custom_call @__cudnn$convForward(%2, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>) -> tensor<1x64x56x56xf32>
    %7 = "mhlo.fusion"(%arg8, %arg7, %6, %arg165, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<1x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<64xf32>
      %162 = stablehlo.rsqrt %161 : tensor<64xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x64x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x64x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x64x56x56xf32>
      mhlo.return %170 : tensor<1x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %8 = stablehlo.custom_call @__cudnn$convForward(%7, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>
    %9 = stablehlo.get_tuple_element %8[0] : (tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>) -> tensor<1x64x56x56xf32>
    %10 = "mhlo.fusion"(%arg11, %arg10, %9, %arg167, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<1x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<64xf32>
      %162 = stablehlo.rsqrt %161 : tensor<64xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x64x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x64x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x64x56x56xf32>
      mhlo.return %170 : tensor<1x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %11 = stablehlo.custom_call @__cudnn$convForward(%10, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<1x256x56x56xf32>, tensor<0xui8>>
    %12 = stablehlo.get_tuple_element %11[0] : (tuple<tensor<1x256x56x56xf32>, tensor<0xui8>>) -> tensor<1x256x56x56xf32>
    %13 = "mhlo.fusion"(%arg5, %arg4, %4, %arg163, %arg14, %arg13, %12, %arg169, %arg170, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x56x56xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<1x256x56x56xf32>, %arg275: tensor<256xf32>, %arg276: tensor<256xf32>, %arg277: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %159 = stablehlo.subtract %arg274, %158 : tensor<1x256x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg276, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x56x56xf32>
      %169 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %170 = stablehlo.subtract %arg270, %169 : tensor<1x256x56x56xf32>
      %171 = stablehlo.add %arg277, %160 : tensor<256xf32>
      %172 = stablehlo.rsqrt %171 : tensor<256xf32>
      %173 = stablehlo.broadcast_in_dim %172, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %174 = stablehlo.multiply %170, %173 : tensor<1x256x56x56xf32>
      %175 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %176 = stablehlo.multiply %174, %175 : tensor<1x256x56x56xf32>
      %177 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %178 = stablehlo.add %176, %177 : tensor<1x256x56x56xf32>
      %179 = stablehlo.add %168, %178 : tensor<1x256x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %180 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x56x56xf32>
      %181 = stablehlo.maximum %179, %180 : tensor<1x256x56x56xf32>
      mhlo.return %181 : tensor<1x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %14 = stablehlo.custom_call @__cudnn$convForward(%13, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>
    %15 = stablehlo.get_tuple_element %14[0] : (tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>) -> tensor<1x64x56x56xf32>
    %16 = "mhlo.fusion"(%arg17, %arg16, %15, %arg171, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<1x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<64xf32>
      %162 = stablehlo.rsqrt %161 : tensor<64xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x64x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x64x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x64x56x56xf32>
      mhlo.return %170 : tensor<1x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %17 = stablehlo.custom_call @__cudnn$convForward(%16, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>
    %18 = stablehlo.get_tuple_element %17[0] : (tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>) -> tensor<1x64x56x56xf32>
    %19 = "mhlo.fusion"(%arg20, %arg19, %18, %arg173, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<1x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<64xf32>
      %162 = stablehlo.rsqrt %161 : tensor<64xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x64x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x64x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x64x56x56xf32>
      mhlo.return %170 : tensor<1x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %20 = stablehlo.custom_call @__cudnn$convForward(%19, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<1x256x56x56xf32>, tensor<0xui8>>
    %21 = stablehlo.get_tuple_element %20[0] : (tuple<tensor<1x256x56x56xf32>, tensor<0xui8>>) -> tensor<1x256x56x56xf32>
    %22 = "mhlo.fusion"(%13, %arg23, %arg22, %21, %arg175, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x256x56x56xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<1x256x56x56xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x256x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x56x56xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x256x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x56x56xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x256x56x56xf32>
      mhlo.return %171 : tensor<1x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %23 = stablehlo.custom_call @__cudnn$convForward(%22, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>
    %24 = stablehlo.get_tuple_element %23[0] : (tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>) -> tensor<1x64x56x56xf32>
    %25 = "mhlo.fusion"(%arg26, %arg25, %24, %arg177, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<1x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<64xf32>
      %162 = stablehlo.rsqrt %161 : tensor<64xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x64x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x64x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x64x56x56xf32>
      mhlo.return %170 : tensor<1x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %26 = stablehlo.custom_call @__cudnn$convForward(%25, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>
    %27 = stablehlo.get_tuple_element %26[0] : (tuple<tensor<1x64x56x56xf32>, tensor<0xui8>>) -> tensor<1x64x56x56xf32>
    %28 = "mhlo.fusion"(%arg29, %arg28, %27, %arg179, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<1x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<64xf32>
      %162 = stablehlo.rsqrt %161 : tensor<64xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x64x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x64x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<1x64x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x56x56xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x64x56x56xf32>
      mhlo.return %170 : tensor<1x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %29 = stablehlo.custom_call @__cudnn$convForward(%28, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<1x256x56x56xf32>, tensor<0xui8>>
    %30 = stablehlo.get_tuple_element %29[0] : (tuple<tensor<1x256x56x56xf32>, tensor<0xui8>>) -> tensor<1x256x56x56xf32>
    %31 = "mhlo.fusion"(%22, %arg32, %arg31, %30, %arg181, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x256x56x56xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<1x256x56x56xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x256x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x56x56xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x256x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x56x56xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x256x56x56xf32>
      mhlo.return %171 : tensor<1x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %32 = stablehlo.custom_call @__cudnn$convForward(%31, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<1x128x56x56xf32>, tensor<0xui8>>
    %33 = stablehlo.get_tuple_element %32[0] : (tuple<tensor<1x128x56x56xf32>, tensor<0xui8>>) -> tensor<1x128x56x56xf32>
    %34 = "mhlo.fusion"(%arg38, %arg37, %33, %arg185, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<1x128x56x56xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<1x128x56x56xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x128x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<128xf32>
      %162 = stablehlo.rsqrt %161 : tensor<128xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<128xf32>) -> tensor<1x128x56x56xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x128x56x56xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<1x128x56x56xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x128x56x56xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<1x128x56x56xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x128x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x56x56xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x128x56x56xf32>
      mhlo.return %170 : tensor<1x128x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x56x56xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x56x56xf32>
    %35 = stablehlo.custom_call @__cudnn$convForward(%34, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>
    %36 = stablehlo.get_tuple_element %35[0] : (tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>) -> tensor<1x128x28x28xf32>
    %37 = "mhlo.fusion"(%arg41, %arg40, %36, %arg187, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<1x128x28x28xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<128xf32>
      %162 = stablehlo.rsqrt %161 : tensor<128xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x128x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x128x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x128x28x28xf32>
      mhlo.return %170 : tensor<1x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %38 = stablehlo.custom_call @__cudnn$convForward(%37, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>
    %39 = stablehlo.get_tuple_element %38[0] : (tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>) -> tensor<1x512x28x28xf32>
    %40 = stablehlo.custom_call @__cudnn$convForward(%31, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>
    %41 = stablehlo.get_tuple_element %40[0] : (tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>) -> tensor<1x512x28x28xf32>
    %42 = "mhlo.fusion"(%arg44, %arg43, %39, %arg189, %arg35, %arg34, %41, %arg183, %arg190, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<1x512x28x28xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>, %arg274: tensor<1x512x28x28xf32>, %arg275: tensor<512xf32>, %arg276: tensor<512xf32>, %arg277: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x512x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg276, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %170 = stablehlo.subtract %arg274, %169 : tensor<1x512x28x28xf32>
      %171 = stablehlo.add %arg277, %160 : tensor<512xf32>
      %172 = stablehlo.rsqrt %171 : tensor<512xf32>
      %173 = stablehlo.broadcast_in_dim %172, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %174 = stablehlo.multiply %170, %173 : tensor<1x512x28x28xf32>
      %175 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %176 = stablehlo.multiply %174, %175 : tensor<1x512x28x28xf32>
      %177 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %178 = stablehlo.add %176, %177 : tensor<1x512x28x28xf32>
      %179 = stablehlo.add %168, %178 : tensor<1x512x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %180 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x28x28xf32>
      %181 = stablehlo.maximum %179, %180 : tensor<1x512x28x28xf32>
      mhlo.return %181 : tensor<1x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %43 = stablehlo.custom_call @__cudnn$convForward(%42, %arg45) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>
    %44 = stablehlo.get_tuple_element %43[0] : (tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>) -> tensor<1x128x28x28xf32>
    %45 = "mhlo.fusion"(%arg47, %arg46, %44, %arg191, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<1x128x28x28xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<128xf32>
      %162 = stablehlo.rsqrt %161 : tensor<128xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x128x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x128x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x128x28x28xf32>
      mhlo.return %170 : tensor<1x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %46 = stablehlo.custom_call @__cudnn$convForward(%45, %arg48) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>
    %47 = stablehlo.get_tuple_element %46[0] : (tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>) -> tensor<1x128x28x28xf32>
    %48 = "mhlo.fusion"(%arg50, %arg49, %47, %arg193, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<1x128x28x28xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<128xf32>
      %162 = stablehlo.rsqrt %161 : tensor<128xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x128x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x128x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x128x28x28xf32>
      mhlo.return %170 : tensor<1x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %49 = stablehlo.custom_call @__cudnn$convForward(%48, %arg51) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>
    %50 = stablehlo.get_tuple_element %49[0] : (tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>) -> tensor<1x512x28x28xf32>
    %51 = "mhlo.fusion"(%42, %arg53, %arg52, %50, %arg195, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<1x512x28x28xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x512x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x28x28xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x512x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x28x28xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x512x28x28xf32>
      mhlo.return %171 : tensor<1x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %52 = stablehlo.custom_call @__cudnn$convForward(%51, %arg54) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>
    %53 = stablehlo.get_tuple_element %52[0] : (tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>) -> tensor<1x128x28x28xf32>
    %54 = "mhlo.fusion"(%arg56, %arg55, %53, %arg197, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<1x128x28x28xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<128xf32>
      %162 = stablehlo.rsqrt %161 : tensor<128xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x128x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x128x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x128x28x28xf32>
      mhlo.return %170 : tensor<1x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %55 = stablehlo.custom_call @__cudnn$convForward(%54, %arg57) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>
    %56 = stablehlo.get_tuple_element %55[0] : (tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>) -> tensor<1x128x28x28xf32>
    %57 = "mhlo.fusion"(%arg59, %arg58, %56, %arg199, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<1x128x28x28xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<128xf32>
      %162 = stablehlo.rsqrt %161 : tensor<128xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x128x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x128x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x128x28x28xf32>
      mhlo.return %170 : tensor<1x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %58 = stablehlo.custom_call @__cudnn$convForward(%57, %arg60) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>
    %59 = stablehlo.get_tuple_element %58[0] : (tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>) -> tensor<1x512x28x28xf32>
    %60 = "mhlo.fusion"(%51, %arg62, %arg61, %59, %arg201, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<1x512x28x28xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x512x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x28x28xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x512x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x28x28xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x512x28x28xf32>
      mhlo.return %171 : tensor<1x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %61 = stablehlo.custom_call @__cudnn$convForward(%60, %arg63) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>
    %62 = stablehlo.get_tuple_element %61[0] : (tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>) -> tensor<1x128x28x28xf32>
    %63 = "mhlo.fusion"(%arg65, %arg64, %62, %arg203, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<1x128x28x28xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<128xf32>
      %162 = stablehlo.rsqrt %161 : tensor<128xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x128x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x128x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x128x28x28xf32>
      mhlo.return %170 : tensor<1x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %64 = stablehlo.custom_call @__cudnn$convForward(%63, %arg66) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>
    %65 = stablehlo.get_tuple_element %64[0] : (tuple<tensor<1x128x28x28xf32>, tensor<0xui8>>) -> tensor<1x128x28x28xf32>
    %66 = "mhlo.fusion"(%arg68, %arg67, %65, %arg205, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<1x128x28x28xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<128xf32>
      %162 = stablehlo.rsqrt %161 : tensor<128xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x128x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x128x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<1x128x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x28x28xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x128x28x28xf32>
      mhlo.return %170 : tensor<1x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %67 = stablehlo.custom_call @__cudnn$convForward(%66, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>
    %68 = stablehlo.get_tuple_element %67[0] : (tuple<tensor<1x512x28x28xf32>, tensor<0xui8>>) -> tensor<1x512x28x28xf32>
    %69 = "mhlo.fusion"(%60, %arg71, %arg70, %68, %arg207, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<1x512x28x28xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x512x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x28x28xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x512x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x28x28xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x512x28x28xf32>
      mhlo.return %171 : tensor<1x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %70 = stablehlo.custom_call @__cudnn$convForward(%69, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>
    %71 = stablehlo.get_tuple_element %70[0] : (tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>) -> tensor<1x1024x14x14xf32>
    %72 = stablehlo.custom_call @__cudnn$convForward(%69, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<1x256x28x28xf32>, tensor<0xui8>>
    %73 = stablehlo.get_tuple_element %72[0] : (tuple<tensor<1x256x28x28xf32>, tensor<0xui8>>) -> tensor<1x256x28x28xf32>
    %74 = "mhlo.fusion"(%arg77, %arg76, %73, %arg211, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x28x28xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x28x28xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x28x28xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x28x28xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x28x28xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x28x28xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x28x28xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x28x28xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x28x28xf32>
      mhlo.return %170 : tensor<1x256x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x28x28xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x28x28xf32>
    %75 = stablehlo.custom_call @__cudnn$convForward(%74, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %76 = stablehlo.get_tuple_element %75[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %77 = "mhlo.fusion"(%arg80, %arg79, %76, %arg213, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %78 = stablehlo.custom_call @__cudnn$convForward(%77, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>
    %79 = stablehlo.get_tuple_element %78[0] : (tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>) -> tensor<1x1024x14x14xf32>
    %80 = "mhlo.fusion"(%arg74, %arg73, %71, %arg209, %arg83, %arg82, %79, %arg215, %arg216, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1x1024x14x14xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1x1024x14x14xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<1024xf32>, %arg277: tensor<1024xf32>):
      %158 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %159 = stablehlo.subtract %arg274, %158 : tensor<1x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %161 = stablehlo.add %arg276, %160 : tensor<1024xf32>
      %162 = stablehlo.rsqrt %161 : tensor<1024xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x1024x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x1024x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x1024x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %170 = stablehlo.subtract %arg270, %169 : tensor<1x1024x14x14xf32>
      %171 = stablehlo.add %arg277, %160 : tensor<1024xf32>
      %172 = stablehlo.rsqrt %171 : tensor<1024xf32>
      %173 = stablehlo.broadcast_in_dim %172, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %174 = stablehlo.multiply %170, %173 : tensor<1x1024x14x14xf32>
      %175 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %176 = stablehlo.multiply %174, %175 : tensor<1x1024x14x14xf32>
      %177 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %178 = stablehlo.add %176, %177 : tensor<1x1024x14x14xf32>
      %179 = stablehlo.add %168, %178 : tensor<1x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %180 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
      %181 = stablehlo.maximum %179, %180 : tensor<1x1024x14x14xf32>
      mhlo.return %181 : tensor<1x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %81 = stablehlo.custom_call @__cudnn$convForward(%80, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %82 = stablehlo.get_tuple_element %81[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %83 = "mhlo.fusion"(%arg86, %arg85, %82, %arg217, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %84 = stablehlo.custom_call @__cudnn$convForward(%83, %arg87) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %85 = stablehlo.get_tuple_element %84[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %86 = "mhlo.fusion"(%arg89, %arg88, %85, %arg219, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %87 = stablehlo.custom_call @__cudnn$convForward(%86, %arg90) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>
    %88 = stablehlo.get_tuple_element %87[0] : (tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>) -> tensor<1x1024x14x14xf32>
    %89 = "mhlo.fusion"(%80, %arg92, %arg91, %88, %arg221, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<1024xf32>
      %162 = stablehlo.rsqrt %161 : tensor<1024xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x1024x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x1024x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x1024x14x14xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x1024x14x14xf32>
      mhlo.return %171 : tensor<1x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %90 = stablehlo.custom_call @__cudnn$convForward(%89, %arg93) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %91 = stablehlo.get_tuple_element %90[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %92 = "mhlo.fusion"(%arg95, %arg94, %91, %arg223, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %93 = stablehlo.custom_call @__cudnn$convForward(%92, %arg96) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %94 = stablehlo.get_tuple_element %93[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %95 = "mhlo.fusion"(%arg98, %arg97, %94, %arg225, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %96 = stablehlo.custom_call @__cudnn$convForward(%95, %arg99) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>
    %97 = stablehlo.get_tuple_element %96[0] : (tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>) -> tensor<1x1024x14x14xf32>
    %98 = "mhlo.fusion"(%89, %arg101, %arg100, %97, %arg227, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<1024xf32>
      %162 = stablehlo.rsqrt %161 : tensor<1024xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x1024x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x1024x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x1024x14x14xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x1024x14x14xf32>
      mhlo.return %171 : tensor<1x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %99 = stablehlo.custom_call @__cudnn$convForward(%98, %arg102) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %100 = stablehlo.get_tuple_element %99[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %101 = "mhlo.fusion"(%arg104, %arg103, %100, %arg229, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %102 = stablehlo.custom_call @__cudnn$convForward(%101, %arg105) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %103 = stablehlo.get_tuple_element %102[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %104 = "mhlo.fusion"(%arg107, %arg106, %103, %arg231, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %105 = stablehlo.custom_call @__cudnn$convForward(%104, %arg108) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>
    %106 = stablehlo.get_tuple_element %105[0] : (tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>) -> tensor<1x1024x14x14xf32>
    %107 = "mhlo.fusion"(%98, %arg110, %arg109, %106, %arg233, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<1024xf32>
      %162 = stablehlo.rsqrt %161 : tensor<1024xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x1024x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x1024x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x1024x14x14xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x1024x14x14xf32>
      mhlo.return %171 : tensor<1x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %108 = stablehlo.custom_call @__cudnn$convForward(%107, %arg111) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %109 = stablehlo.get_tuple_element %108[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %110 = "mhlo.fusion"(%arg113, %arg112, %109, %arg235, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %111 = stablehlo.custom_call @__cudnn$convForward(%110, %arg114) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %112 = stablehlo.get_tuple_element %111[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %113 = "mhlo.fusion"(%arg116, %arg115, %112, %arg237, %arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %114 = stablehlo.custom_call @__cudnn$convForward(%113, %arg117) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>
    %115 = stablehlo.get_tuple_element %114[0] : (tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>) -> tensor<1x1024x14x14xf32>
    %116 = "mhlo.fusion"(%107, %arg119, %arg118, %115, %arg239, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<1024xf32>
      %162 = stablehlo.rsqrt %161 : tensor<1024xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x1024x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x1024x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x1024x14x14xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x1024x14x14xf32>
      mhlo.return %171 : tensor<1x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %117 = stablehlo.custom_call @__cudnn$convForward(%116, %arg120) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %118 = stablehlo.get_tuple_element %117[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %119 = "mhlo.fusion"(%arg122, %arg121, %118, %arg241, %arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %120 = stablehlo.custom_call @__cudnn$convForward(%119, %arg123) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>
    %121 = stablehlo.get_tuple_element %120[0] : (tuple<tensor<1x256x14x14xf32>, tensor<0xui8>>) -> tensor<1x256x14x14xf32>
    %122 = "mhlo.fusion"(%arg125, %arg124, %121, %arg243, %arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<1x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<256xf32>
      %162 = stablehlo.rsqrt %161 : tensor<256xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x256x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x256x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<1x256x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x256x14x14xf32>
      mhlo.return %170 : tensor<1x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %123 = stablehlo.custom_call @__cudnn$convForward(%122, %arg126) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>
    %124 = stablehlo.get_tuple_element %123[0] : (tuple<tensor<1x1024x14x14xf32>, tensor<0xui8>>) -> tensor<1x1024x14x14xf32>
    %125 = "mhlo.fusion"(%116, %arg128, %arg127, %124, %arg245, %arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<1024xf32>
      %162 = stablehlo.rsqrt %161 : tensor<1024xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x1024x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x1024x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x1024x14x14xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x14x14xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x1024x14x14xf32>
      mhlo.return %171 : tensor<1x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %126 = stablehlo.custom_call @__cudnn$convForward(%125, %arg132) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tuple<tensor<1x512x14x14xf32>, tensor<0xui8>>
    %127 = stablehlo.get_tuple_element %126[0] : (tuple<tensor<1x512x14x14xf32>, tensor<0xui8>>) -> tensor<1x512x14x14xf32>
    %128 = "mhlo.fusion"(%arg134, %arg133, %127, %arg249, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<1x512x14x14xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<1x512x14x14xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x512x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x14x14xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x14x14xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x14x14xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x14x14xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<1x512x14x14xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x14x14xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x512x14x14xf32>
      mhlo.return %170 : tensor<1x512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x14x14xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x14x14xf32>
    %129 = stablehlo.custom_call @__cudnn$convForward(%128, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>
    %130 = stablehlo.get_tuple_element %129[0] : (tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>) -> tensor<1x512x7x7xf32>
    %131 = "mhlo.fusion"(%arg137, %arg136, %130, %arg251, %arg252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<1x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x7x7xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x7x7xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x512x7x7xf32>
      mhlo.return %170 : tensor<1x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %132 = stablehlo.custom_call @__cudnn$convForward(%131, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<1x2048x7x7xf32>, tensor<0xui8>>
    %133 = stablehlo.get_tuple_element %132[0] : (tuple<tensor<1x2048x7x7xf32>, tensor<0xui8>>) -> tensor<1x2048x7x7xf32>
    %134 = stablehlo.custom_call @__cudnn$convForward(%125, %arg129) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tuple<tensor<1x2048x7x7xf32>, tensor<0xui8>>
    %135 = stablehlo.get_tuple_element %134[0] : (tuple<tensor<1x2048x7x7xf32>, tensor<0xui8>>) -> tensor<1x2048x7x7xf32>
    %136 = "mhlo.fusion"(%arg140, %arg139, %133, %arg253, %arg131, %arg130, %135, %arg247, %arg254, %arg248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<1x2048x7x7xf32>, %arg271: tensor<2048xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>, %arg274: tensor<1x2048x7x7xf32>, %arg275: tensor<2048xf32>, %arg276: tensor<2048xf32>, %arg277: tensor<2048xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x2048x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %161 = stablehlo.add %arg276, %160 : tensor<2048xf32>
      %162 = stablehlo.rsqrt %161 : tensor<2048xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x2048x7x7xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x2048x7x7xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x2048x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %170 = stablehlo.subtract %arg274, %169 : tensor<1x2048x7x7xf32>
      %171 = stablehlo.add %arg277, %160 : tensor<2048xf32>
      %172 = stablehlo.rsqrt %171 : tensor<2048xf32>
      %173 = stablehlo.broadcast_in_dim %172, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %174 = stablehlo.multiply %170, %173 : tensor<1x2048x7x7xf32>
      %175 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %176 = stablehlo.multiply %174, %175 : tensor<1x2048x7x7xf32>
      %177 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %178 = stablehlo.add %176, %177 : tensor<1x2048x7x7xf32>
      %179 = stablehlo.add %168, %178 : tensor<1x2048x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %180 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2048x7x7xf32>
      %181 = stablehlo.maximum %179, %180 : tensor<1x2048x7x7xf32>
      mhlo.return %181 : tensor<1x2048x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %137 = stablehlo.custom_call @__cudnn$convForward(%136, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>
    %138 = stablehlo.get_tuple_element %137[0] : (tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>) -> tensor<1x512x7x7xf32>
    %139 = "mhlo.fusion"(%arg143, %arg142, %138, %arg255, %arg256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<1x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x7x7xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x7x7xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x512x7x7xf32>
      mhlo.return %170 : tensor<1x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %140 = stablehlo.custom_call @__cudnn$convForward(%139, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>
    %141 = stablehlo.get_tuple_element %140[0] : (tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>) -> tensor<1x512x7x7xf32>
    %142 = "mhlo.fusion"(%arg146, %arg145, %141, %arg257, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<1x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x7x7xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x7x7xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x512x7x7xf32>
      mhlo.return %170 : tensor<1x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %143 = stablehlo.custom_call @__cudnn$convForward(%142, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<1x2048x7x7xf32>, tensor<0xui8>>
    %144 = stablehlo.get_tuple_element %143[0] : (tuple<tensor<1x2048x7x7xf32>, tensor<0xui8>>) -> tensor<1x2048x7x7xf32>
    %145 = "mhlo.fusion"(%136, %arg149, %arg148, %144, %arg259, %arg260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1x2048x7x7xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<1x2048x7x7xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x2048x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<2048xf32>
      %162 = stablehlo.rsqrt %161 : tensor<2048xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x2048x7x7xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x2048x7x7xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x2048x7x7xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x2048x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2048x7x7xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x2048x7x7xf32>
      mhlo.return %171 : tensor<1x2048x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %146 = stablehlo.custom_call @__cudnn$convForward(%145, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>
    %147 = stablehlo.get_tuple_element %146[0] : (tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>) -> tensor<1x512x7x7xf32>
    %148 = "mhlo.fusion"(%arg152, %arg151, %147, %arg261, %arg262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<1x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x7x7xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x7x7xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x512x7x7xf32>
      mhlo.return %170 : tensor<1x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %149 = stablehlo.custom_call @__cudnn$convForward(%148, %arg153) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>
    %150 = stablehlo.get_tuple_element %149[0] : (tuple<tensor<1x512x7x7xf32>, tensor<0xui8>>) -> tensor<1x512x7x7xf32>
    %151 = "mhlo.fusion"(%arg155, %arg154, %150, %arg263, %arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<1x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %158 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %159 = stablehlo.subtract %arg270, %158 : tensor<1x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %161 = stablehlo.add %arg272, %160 : tensor<512xf32>
      %162 = stablehlo.rsqrt %161 : tensor<512xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x512x7x7xf32>
      %165 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x512x7x7xf32>
      %167 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<1x512x7x7xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %169 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x7x7xf32>
      %170 = stablehlo.maximum %168, %169 : tensor<1x512x7x7xf32>
      mhlo.return %170 : tensor<1x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %152 = stablehlo.custom_call @__cudnn$convForward(%151, %arg156) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<1x2048x7x7xf32>, tensor<0xui8>>
    %153 = stablehlo.get_tuple_element %152[0] : (tuple<tensor<1x2048x7x7xf32>, tensor<0xui8>>) -> tensor<1x2048x7x7xf32>
    %154 = "mhlo.fusion"(%145, %arg158, %arg157, %153, %arg265, %arg266) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg268: tensor<1x2048x7x7xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<1x2048x7x7xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>):
      %158 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %159 = stablehlo.subtract %arg271, %158 : tensor<1x2048x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %160 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %161 = stablehlo.add %arg273, %160 : tensor<2048xf32>
      %162 = stablehlo.rsqrt %161 : tensor<2048xf32>
      %163 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %164 = stablehlo.multiply %159, %163 : tensor<1x2048x7x7xf32>
      %165 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %166 = stablehlo.multiply %164, %165 : tensor<1x2048x7x7xf32>
      %167 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
      %168 = stablehlo.add %166, %167 : tensor<1x2048x7x7xf32>
      %169 = stablehlo.add %168, %arg268 : tensor<1x2048x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2048x7x7xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<1x2048x7x7xf32>
      %172 = mhlo.bitcast %171 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2048x7x7xf32>) -> tensor<2048x49xf32>
      %173 = stablehlo.reduce(%172 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<2048x49xf32>, tensor<f32>) -> tensor<2048xf32>
      mhlo.return %173 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %155 = "mhlo.fusion"(%arg159, %154) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg268: tensor<1000x2048xf32>, %arg269: tensor<2048xf32>):
      %158 = mhlo.bitcast %arg269 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<2048xf32>) -> tensor<1x2048xf32>
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %159 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x2048xf32>
      %160 = stablehlo.multiply %158, %159 : tensor<1x2048xf32>
      %161 = mhlo.bitcast %160 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x2048xf32>) -> tensor<2048xf32>
      %162 = stablehlo.broadcast_in_dim %161, dims = [1] : (tensor<2048xf32>) -> tensor<1000x2048xf32>
      %163 = stablehlo.multiply %162, %arg268 : tensor<1000x2048xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %164 = stablehlo.reduce(%163 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<1000x2048xf32>, tensor<f32>) -> tensor<1000xf32>
      mhlo.return %164 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x2048xf32>, tensor<2048xf32>) -> tensor<1000xf32>
    %156 = "mhlo.fusion"(%arg160, %155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1000xf32>, %arg269: tensor<1000xf32>):
      %158 = stablehlo.add %arg268, %arg269 : tensor<1000xf32>
      mhlo.return %158 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %157 : tensor<1x1000xf32>
  }
}
