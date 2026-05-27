module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.1_computation(%arg0: tensor<8x2048xf32>, %arg1: tensor<1000x2048xf32>) -> tensor<1000x8xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x2048xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<8x2048xf32>
    %2 = stablehlo.dot_general %arg1, %1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1000,8]{0,1}"} : (tensor<1000x2048xf32>, tensor<8x2048xf32>) -> tensor<1000x8xf32>
    return %2 : tensor<1000x8xf32>
  }
  func.func private @region_1.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<8x2048x7x7xf32>, %arg1: tensor<8x2048x7x7xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>) -> tensor<8x2048xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %1 = stablehlo.subtract %arg1, %0 : tensor<8x2048x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<2048xf32>
    %4 = stablehlo.rsqrt %3 : tensor<2048xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x2048x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x2048x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x2048x7x7xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x2048x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x2048x7x7xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x2048x7x7xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2048x7x7xf32>) -> tensor<8x2048x49xf32>
    %15 = stablehlo.reduce(%14 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x2048x49xf32>, tensor<f32>) -> tensor<8x2048xf32>
    return %15 : tensor<8x2048xf32>
  }
  func.func private @fused_add(%arg0: tensor<1000x8xf32>, %arg1: tensor<1000xf32>) -> tensor<8x1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1000xf32>) -> tensor<8x1000xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x8xf32>) -> tensor<8x1000xf32>
    %2 = stablehlo.add %0, %1 : tensor<8x1000xf32>
    return %2 : tensor<8x1000xf32>
  }
  func.func private @fused_maximum(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8x256x56x56xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<8x256x56x56xf32>, %arg9: tensor<256xf32>) -> tensor<8x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x256x56x56xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<8x256x56x56xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<8x256x56x56xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<8x256x56x56xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %15 = stablehlo.add %13, %14 : tensor<8x256x56x56xf32>
    %16 = stablehlo.add %7, %15 : tensor<8x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x56x56xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<8x256x56x56xf32>
    return %18 : tensor<8x256x56x56xf32>
  }
  func.func private @fused_maximum.1(%arg0: tensor<8x256x56x56xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<8x256x56x56xf32>, %arg5: tensor<256xf32>) -> tensor<8x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<8x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x256x56x56xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<8x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x56x56xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<8x256x56x56xf32>
    return %10 : tensor<8x256x56x56xf32>
  }
  func.func private @fused_maximum.2(%arg0: tensor<8x256x56x56xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<8x256x56x56xf32>, %arg5: tensor<256xf32>) -> tensor<8x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<8x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x256x56x56xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<8x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x56x56xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<8x256x56x56xf32>
    return %10 : tensor<8x256x56x56xf32>
  }
  func.func private @fused_maximum.3(%arg0: tensor<8x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<8x512x28x28xf32>, %arg7: tensor<512xf32>, %arg8: tensor<512xf32>, %arg9: tensor<512xf32>) -> tensor<8x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x512x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg8, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x28x28xf32>
    %11 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %12 = stablehlo.subtract %arg6, %11 : tensor<8x512x28x28xf32>
    %13 = stablehlo.add %arg9, %2 : tensor<512xf32>
    %14 = stablehlo.rsqrt %13 : tensor<512xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %16 = stablehlo.multiply %12, %15 : tensor<8x512x28x28xf32>
    %17 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<8x512x28x28xf32>
    %19 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %20 = stablehlo.add %18, %19 : tensor<8x512x28x28xf32>
    %21 = stablehlo.add %10, %20 : tensor<8x512x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x28x28xf32>
    %23 = stablehlo.maximum %21, %22 : tensor<8x512x28x28xf32>
    return %23 : tensor<8x512x28x28xf32>
  }
  func.func private @fused_maximum.4(%arg0: tensor<8x512x28x28xf32>, %arg1: tensor<8x512x28x28xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>) -> tensor<8x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %1 = stablehlo.subtract %arg1, %0 : tensor<8x512x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x28x28xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x512x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x28x28xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x512x28x28xf32>
    return %13 : tensor<8x512x28x28xf32>
  }
  func.func private @fused_maximum.5(%arg0: tensor<8x512x28x28xf32>, %arg1: tensor<8x512x28x28xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>) -> tensor<8x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %1 = stablehlo.subtract %arg1, %0 : tensor<8x512x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x28x28xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x512x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x28x28xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x512x28x28xf32>
    return %13 : tensor<8x512x28x28xf32>
  }
  func.func private @fused_maximum.6(%arg0: tensor<8x512x28x28xf32>, %arg1: tensor<8x512x28x28xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>) -> tensor<8x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %1 = stablehlo.subtract %arg1, %0 : tensor<8x512x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x28x28xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x512x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x28x28xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x512x28x28xf32>
    return %13 : tensor<8x512x28x28xf32>
  }
  func.func private @fused_maximum.7(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<8x1024x14x14xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<8x1024x14x14xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<1024xf32>, %arg9: tensor<1024xf32>) -> tensor<8x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<8x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg8, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x1024x14x14xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %12 = stablehlo.subtract %arg2, %11 : tensor<8x1024x14x14xf32>
    %13 = stablehlo.add %arg9, %2 : tensor<1024xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %16 = stablehlo.multiply %12, %15 : tensor<8x1024x14x14xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<8x1024x14x14xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %20 = stablehlo.add %18, %19 : tensor<8x1024x14x14xf32>
    %21 = stablehlo.add %10, %20 : tensor<8x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
    %23 = stablehlo.maximum %21, %22 : tensor<8x1024x14x14xf32>
    return %23 : tensor<8x1024x14x14xf32>
  }
  func.func private @fused_maximum.8(%arg0: tensor<8x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<8x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<8x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x1024x14x14xf32>
    return %13 : tensor<8x1024x14x14xf32>
  }
  func.func private @fused_maximum.9(%arg0: tensor<8x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<8x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<8x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x1024x14x14xf32>
    return %13 : tensor<8x1024x14x14xf32>
  }
  func.func private @fused_maximum.10(%arg0: tensor<8x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<8x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<8x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x1024x14x14xf32>
    return %13 : tensor<8x1024x14x14xf32>
  }
  func.func private @fused_maximum.11(%arg0: tensor<8x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<8x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<8x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x1024x14x14xf32>
    return %13 : tensor<8x1024x14x14xf32>
  }
  func.func private @fused_maximum.12(%arg0: tensor<8x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<8x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>) -> tensor<8x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x1024x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x1024x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x1024x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x1024x14x14xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x1024x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x1024x14x14xf32>
    return %13 : tensor<8x1024x14x14xf32>
  }
  func.func private @fused_maximum.13(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<8x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x64x56x56xf32>
    return %12 : tensor<8x64x56x56xf32>
  }
  func.func private @fused_maximum.14(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<8x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x64x56x56xf32>
    return %12 : tensor<8x64x56x56xf32>
  }
  func.func private @fused_maximum.15(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<8x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x64x56x56xf32>
    return %12 : tensor<8x64x56x56xf32>
  }
  func.func private @fused_maximum.16(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<8x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x64x56x56xf32>
    return %12 : tensor<8x64x56x56xf32>
  }
  func.func private @fused_maximum.17(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<8x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x64x56x56xf32>
    return %12 : tensor<8x64x56x56xf32>
  }
  func.func private @fused_maximum.18(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<8x64x56x56xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x64x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x64x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x64x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x64x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x64x56x56xf32>
    return %12 : tensor<8x64x56x56xf32>
  }
  func.func private @region_0.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_window(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<8x64x112x112xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x112x112xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x64x112x112xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<8x64x112x112xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x64x112x112xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x112x112xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x64x112x112xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x112x112xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x64x112x112xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x64x112x112xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<8x64x112x112xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %10 = "stablehlo.reduce_window"(%9, %cst_0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %11 = stablehlo.maximum %arg5, %arg6 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) : (tensor<8x64x112x112xf32>, tensor<f32>) -> tensor<8x64x56x56xf32>
    return %10 : tensor<8x64x56x56xf32>
  }
  func.func private @fused_maximum.19(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<8x128x56x56xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x128x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x128x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x128x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x128x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x56x56xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x128x56x56xf32>
    return %12 : tensor<8x128x56x56xf32>
  }
  func.func private @fused_maximum.20(%arg0: tensor<8x128x28x28xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x128x28x28xf32>
    return %12 : tensor<8x128x28x28xf32>
  }
  func.func private @fused_maximum.21(%arg0: tensor<8x128x28x28xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x128x28x28xf32>
    return %12 : tensor<8x128x28x28xf32>
  }
  func.func private @fused_maximum.22(%arg0: tensor<8x128x28x28xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x128x28x28xf32>
    return %12 : tensor<8x128x28x28xf32>
  }
  func.func private @fused_maximum.23(%arg0: tensor<8x128x28x28xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x128x28x28xf32>
    return %12 : tensor<8x128x28x28xf32>
  }
  func.func private @fused_maximum.24(%arg0: tensor<8x128x28x28xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x128x28x28xf32>
    return %12 : tensor<8x128x28x28xf32>
  }
  func.func private @fused_maximum.25(%arg0: tensor<8x128x28x28xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x128x28x28xf32>
    return %12 : tensor<8x128x28x28xf32>
  }
  func.func private @fused_maximum.26(%arg0: tensor<8x128x28x28xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x128x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x128x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x128x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x128x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x128x28x28xf32>
    return %12 : tensor<8x128x28x28xf32>
  }
  func.func private @fused_maximum.27(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.28(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.29(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.30(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.31(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.32(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.33(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.34(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.35(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.36(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.37(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x14x14xf32>
    return %12 : tensor<8x256x14x14xf32>
  }
  func.func private @fused_maximum.38(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x28x28xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x28x28xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x256x28x28xf32>
    return %12 : tensor<8x256x28x28xf32>
  }
  func.func private @fused_maximum.39(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<8x2048x7x7xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<8x2048x7x7xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<2048xf32>, %arg9: tensor<2048xf32>) -> tensor<8x2048x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<8x2048x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %3 = stablehlo.add %arg8, %2 : tensor<2048xf32>
    %4 = stablehlo.rsqrt %3 : tensor<2048xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x2048x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x2048x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x2048x7x7xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %12 = stablehlo.subtract %arg2, %11 : tensor<8x2048x7x7xf32>
    %13 = stablehlo.add %arg9, %2 : tensor<2048xf32>
    %14 = stablehlo.rsqrt %13 : tensor<2048xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %16 = stablehlo.multiply %12, %15 : tensor<8x2048x7x7xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<8x2048x7x7xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %20 = stablehlo.add %18, %19 : tensor<8x2048x7x7xf32>
    %21 = stablehlo.add %10, %20 : tensor<8x2048x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x2048x7x7xf32>
    %23 = stablehlo.maximum %21, %22 : tensor<8x2048x7x7xf32>
    return %23 : tensor<8x2048x7x7xf32>
  }
  func.func private @fused_maximum.40(%arg0: tensor<8x2048x7x7xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<8x2048x7x7xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>) -> tensor<8x2048x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x2048x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<2048xf32>
    %4 = stablehlo.rsqrt %3 : tensor<2048xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x2048x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x2048x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x2048x7x7xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<8x2048x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x2048x7x7xf32>
    %13 = stablehlo.maximum %11, %12 : tensor<8x2048x7x7xf32>
    return %13 : tensor<8x2048x7x7xf32>
  }
  func.func private @fused_maximum.41(%arg0: tensor<8x512x14x14xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x14x14xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<8x512x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<8x512x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x512x14x14xf32>
    return %12 : tensor<8x512x14x14xf32>
  }
  func.func private @fused_maximum.42(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x512x7x7xf32>
    return %12 : tensor<8x512x7x7xf32>
  }
  func.func private @fused_maximum.43(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x512x7x7xf32>
    return %12 : tensor<8x512x7x7xf32>
  }
  func.func private @fused_maximum.44(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x512x7x7xf32>
    return %12 : tensor<8x512x7x7xf32>
  }
  func.func private @fused_maximum.45(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x512x7x7xf32>
    return %12 : tensor<8x512x7x7xf32>
  }
  func.func private @fused_maximum.46(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x512x7x7xf32>
    return %12 : tensor<8x512x7x7xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.1(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.2(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.3(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.4(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64x1x1xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<64x64x1x1xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<256x64x1x1xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<64x256x1x1xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64x64x3x3xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<256x64x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<64x256x1x1xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<64x64x3x3xf32>, %arg28: tensor<64xf32>, %arg29: tensor<64xf32>, %arg30: tensor<256x64x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<512x256x1x1xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<128x256x1x1xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128x128x3x3xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<512x128x1x1xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<128x512x1x1xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128x128x3x3xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<512x128x1x1xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512xf32>, %arg54: tensor<128x512x1x1xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128x128x3x3xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<512x128x1x1xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512xf32>, %arg63: tensor<128x512x1x1xf32>, %arg64: tensor<128xf32>, %arg65: tensor<128xf32>, %arg66: tensor<128x128x3x3xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<512x128x1x1xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<1024x512x1x1xf32>, %arg73: tensor<1024xf32>, %arg74: tensor<1024xf32>, %arg75: tensor<256x512x1x1xf32>, %arg76: tensor<256xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x256x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<256xf32>, %arg81: tensor<1024x256x1x1xf32>, %arg82: tensor<1024xf32>, %arg83: tensor<1024xf32>, %arg84: tensor<256x1024x1x1xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256x256x3x3xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<1024x256x1x1xf32>, %arg91: tensor<1024xf32>, %arg92: tensor<1024xf32>, %arg93: tensor<256x1024x1x1xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256x256x3x3xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<1024x256x1x1xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<1024xf32>, %arg102: tensor<256x1024x1x1xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256x256x3x3xf32>, %arg106: tensor<256xf32>, %arg107: tensor<256xf32>, %arg108: tensor<1024x256x1x1xf32>, %arg109: tensor<1024xf32>, %arg110: tensor<1024xf32>, %arg111: tensor<256x1024x1x1xf32>, %arg112: tensor<256xf32>, %arg113: tensor<256xf32>, %arg114: tensor<256x256x3x3xf32>, %arg115: tensor<256xf32>, %arg116: tensor<256xf32>, %arg117: tensor<1024x256x1x1xf32>, %arg118: tensor<1024xf32>, %arg119: tensor<1024xf32>, %arg120: tensor<256x1024x1x1xf32>, %arg121: tensor<256xf32>, %arg122: tensor<256xf32>, %arg123: tensor<256x256x3x3xf32>, %arg124: tensor<256xf32>, %arg125: tensor<256xf32>, %arg126: tensor<1024x256x1x1xf32>, %arg127: tensor<1024xf32>, %arg128: tensor<1024xf32>, %arg129: tensor<2048x1024x1x1xf32>, %arg130: tensor<2048xf32>, %arg131: tensor<2048xf32>, %arg132: tensor<512x1024x1x1xf32>, %arg133: tensor<512xf32>, %arg134: tensor<512xf32>, %arg135: tensor<512x512x3x3xf32>, %arg136: tensor<512xf32>, %arg137: tensor<512xf32>, %arg138: tensor<2048x512x1x1xf32>, %arg139: tensor<2048xf32>, %arg140: tensor<2048xf32>, %arg141: tensor<512x2048x1x1xf32>, %arg142: tensor<512xf32>, %arg143: tensor<512xf32>, %arg144: tensor<512x512x3x3xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<2048x512x1x1xf32>, %arg148: tensor<2048xf32>, %arg149: tensor<2048xf32>, %arg150: tensor<512x2048x1x1xf32>, %arg151: tensor<512xf32>, %arg152: tensor<512xf32>, %arg153: tensor<512x512x3x3xf32>, %arg154: tensor<512xf32>, %arg155: tensor<512xf32>, %arg156: tensor<2048x512x1x1xf32>, %arg157: tensor<2048xf32>, %arg158: tensor<2048xf32>, %arg159: tensor<1000x2048xf32>, %arg160: tensor<1000xf32>, %arg161: tensor<64xf32>, %arg162: tensor<64xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<64xf32>, %arg166: tensor<64xf32>, %arg167: tensor<64xf32>, %arg168: tensor<64xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<64xf32>, %arg174: tensor<64xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<64xf32>, %arg178: tensor<64xf32>, %arg179: tensor<64xf32>, %arg180: tensor<64xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512xf32>, %arg184: tensor<512xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<128xf32>, %arg188: tensor<128xf32>, %arg189: tensor<512xf32>, %arg190: tensor<512xf32>, %arg191: tensor<128xf32>, %arg192: tensor<128xf32>, %arg193: tensor<128xf32>, %arg194: tensor<128xf32>, %arg195: tensor<512xf32>, %arg196: tensor<512xf32>, %arg197: tensor<128xf32>, %arg198: tensor<128xf32>, %arg199: tensor<128xf32>, %arg200: tensor<128xf32>, %arg201: tensor<512xf32>, %arg202: tensor<512xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<512xf32>, %arg208: tensor<512xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1024xf32>, %arg211: tensor<256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<1024xf32>, %arg216: tensor<1024xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<1024xf32>, %arg222: tensor<1024xf32>, %arg223: tensor<256xf32>, %arg224: tensor<256xf32>, %arg225: tensor<256xf32>, %arg226: tensor<256xf32>, %arg227: tensor<1024xf32>, %arg228: tensor<1024xf32>, %arg229: tensor<256xf32>, %arg230: tensor<256xf32>, %arg231: tensor<256xf32>, %arg232: tensor<256xf32>, %arg233: tensor<1024xf32>, %arg234: tensor<1024xf32>, %arg235: tensor<256xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<256xf32>, %arg239: tensor<1024xf32>, %arg240: tensor<1024xf32>, %arg241: tensor<256xf32>, %arg242: tensor<256xf32>, %arg243: tensor<256xf32>, %arg244: tensor<256xf32>, %arg245: tensor<1024xf32>, %arg246: tensor<1024xf32>, %arg247: tensor<2048xf32>, %arg248: tensor<2048xf32>, %arg249: tensor<512xf32>, %arg250: tensor<512xf32>, %arg251: tensor<512xf32>, %arg252: tensor<512xf32>, %arg253: tensor<2048xf32>, %arg254: tensor<2048xf32>, %arg255: tensor<512xf32>, %arg256: tensor<512xf32>, %arg257: tensor<512xf32>, %arg258: tensor<512xf32>, %arg259: tensor<2048xf32>, %arg260: tensor<2048xf32>, %arg261: tensor<512xf32>, %arg262: tensor<512xf32>, %arg263: tensor<512xf32>, %arg264: tensor<512xf32>, %arg265: tensor<2048xf32>, %arg266: tensor<2048xf32>, %arg267: tensor<8x3x224x224xf32>) -> tensor<8x1000xf32> {
    %0 = "mhlo.fusion"(%arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %162 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %163 = stablehlo.add %arg268, %162 : tensor<256xf32>
      %164 = stablehlo.rsqrt %163 : tensor<256xf32>
      mhlo.return %164 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"(%arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %162 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %163 = stablehlo.add %arg268, %162 : tensor<64xf32>
      %164 = stablehlo.rsqrt %163 : tensor<64xf32>
      mhlo.return %164 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %2 = stablehlo.custom_call @__cudnn$convForward(%arg267, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,112,112]{3,2,1,0}, u8[0]{0})"} : (tensor<8x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tuple<tensor<8x64x112x112xf32>, tensor<0xui8>>
    %3 = stablehlo.get_tuple_element %2[0] : (tuple<tensor<8x64x112x112xf32>, tensor<0xui8>>) -> tensor<8x64x112x112xf32>
    %4 = "mhlo.fusion"(%arg2, %arg1, %1, %3, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<8x64x112x112xf32>, %arg272: tensor<64xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<8x64x112x112xf32>
      %163 = stablehlo.subtract %arg271, %162 : tensor<8x64x112x112xf32>
      %164 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<8x64x112x112xf32>
      %165 = stablehlo.multiply %163, %164 : tensor<8x64x112x112xf32>
      %166 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<8x64x112x112xf32>
      %167 = stablehlo.multiply %165, %166 : tensor<8x64x112x112xf32>
      %168 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<8x64x112x112xf32>
      %169 = stablehlo.add %167, %168 : tensor<8x64x112x112xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x64x112x112xf32>
      %171 = stablehlo.maximum %169, %170 : tensor<8x64x112x112xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %172 = "stablehlo.reduce_window"(%171, %cst_0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
      ^bb0(%arg273: tensor<f32>, %arg274: tensor<f32>):
        %173 = stablehlo.maximum %arg273, %arg274 : tensor<f32>
        stablehlo.return %173 : tensor<f32>
      }) : (tensor<8x64x112x112xf32>, tensor<f32>) -> tensor<8x64x56x56xf32>
      mhlo.return %172 : tensor<8x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<8x64x112x112xf32>, tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %5 = stablehlo.custom_call @__cudnn$convForward(%4, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>) -> tensor<8x64x56x56xf32>
    %7 = "mhlo.fusion"(%arg8, %arg7, %6, %arg165, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<8x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<64xf32>
      %166 = stablehlo.rsqrt %165 : tensor<64xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x64x56x56xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x64x56x56xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x64x56x56xf32>
      mhlo.return %174 : tensor<8x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %8 = stablehlo.custom_call @__cudnn$convForward(%7, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>
    %9 = stablehlo.get_tuple_element %8[0] : (tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>) -> tensor<8x64x56x56xf32>
    %10 = "mhlo.fusion"(%arg11, %arg10, %9, %arg167, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<8x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<64xf32>
      %166 = stablehlo.rsqrt %165 : tensor<64xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x64x56x56xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x64x56x56xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x64x56x56xf32>
      mhlo.return %174 : tensor<8x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %11 = stablehlo.custom_call @__cudnn$convForward(%10, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<8x256x56x56xf32>, tensor<0xui8>>
    %12 = stablehlo.get_tuple_element %11[0] : (tuple<tensor<8x256x56x56xf32>, tensor<0xui8>>) -> tensor<8x256x56x56xf32>
    %13 = "mhlo.fusion"(%arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %162 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %163 = stablehlo.add %arg268, %162 : tensor<256xf32>
      %164 = stablehlo.rsqrt %163 : tensor<256xf32>
      mhlo.return %164 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %14 = stablehlo.custom_call @__cudnn$convForward(%4, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<8x256x56x56xf32>, tensor<0xui8>>
    %15 = stablehlo.get_tuple_element %14[0] : (tuple<tensor<8x256x56x56xf32>, tensor<0xui8>>) -> tensor<8x256x56x56xf32>
    %16 = "mhlo.fusion"(%arg14, %arg13, %0, %12, %arg169, %arg5, %arg4, %13, %15, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<8x256x56x56xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<256xf32>, %arg276: tensor<8x256x56x56xf32>, %arg277: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %163 = stablehlo.subtract %arg271, %162 : tensor<8x256x56x56xf32>
      %164 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %165 = stablehlo.multiply %163, %164 : tensor<8x256x56x56xf32>
      %166 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %167 = stablehlo.multiply %165, %166 : tensor<8x256x56x56xf32>
      %168 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %169 = stablehlo.add %167, %168 : tensor<8x256x56x56xf32>
      %170 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %171 = stablehlo.subtract %arg276, %170 : tensor<8x256x56x56xf32>
      %172 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %173 = stablehlo.multiply %171, %172 : tensor<8x256x56x56xf32>
      %174 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %175 = stablehlo.multiply %173, %174 : tensor<8x256x56x56xf32>
      %176 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %177 = stablehlo.add %175, %176 : tensor<8x256x56x56xf32>
      %178 = stablehlo.add %169, %177 : tensor<8x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %179 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x56x56xf32>
      %180 = stablehlo.maximum %178, %179 : tensor<8x256x56x56xf32>
      mhlo.return %180 : tensor<8x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256x56x56xf32>, tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %17 = "mhlo.fusion"(%arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %162 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %163 = stablehlo.add %arg268, %162 : tensor<256xf32>
      %164 = stablehlo.rsqrt %163 : tensor<256xf32>
      mhlo.return %164 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %18 = stablehlo.custom_call @__cudnn$convForward(%16, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>
    %19 = stablehlo.get_tuple_element %18[0] : (tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>) -> tensor<8x64x56x56xf32>
    %20 = "mhlo.fusion"(%arg17, %arg16, %19, %arg171, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<8x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<64xf32>
      %166 = stablehlo.rsqrt %165 : tensor<64xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x64x56x56xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x64x56x56xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x64x56x56xf32>
      mhlo.return %174 : tensor<8x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %21 = stablehlo.custom_call @__cudnn$convForward(%20, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>
    %22 = stablehlo.get_tuple_element %21[0] : (tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>) -> tensor<8x64x56x56xf32>
    %23 = "mhlo.fusion"(%arg20, %arg19, %22, %arg173, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<8x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<64xf32>
      %166 = stablehlo.rsqrt %165 : tensor<64xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x64x56x56xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x64x56x56xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x64x56x56xf32>
      mhlo.return %174 : tensor<8x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %24 = stablehlo.custom_call @__cudnn$convForward(%23, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<8x256x56x56xf32>, tensor<0xui8>>
    %25 = stablehlo.get_tuple_element %24[0] : (tuple<tensor<8x256x56x56xf32>, tensor<0xui8>>) -> tensor<8x256x56x56xf32>
    %26 = "mhlo.fusion"(%16, %arg23, %arg22, %17, %25, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x256x56x56xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<8x256x56x56xf32>, %arg273: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %163 = stablehlo.subtract %arg272, %162 : tensor<8x256x56x56xf32>
      %164 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %165 = stablehlo.multiply %163, %164 : tensor<8x256x56x56xf32>
      %166 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %167 = stablehlo.multiply %165, %166 : tensor<8x256x56x56xf32>
      %168 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %169 = stablehlo.add %167, %168 : tensor<8x256x56x56xf32>
      %170 = stablehlo.add %169, %arg268 : tensor<8x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %171 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x56x56xf32>
      %172 = stablehlo.maximum %170, %171 : tensor<8x256x56x56xf32>
      mhlo.return %172 : tensor<8x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256x56x56xf32>, tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %27 = "mhlo.fusion"(%arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %162 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %163 = stablehlo.add %arg268, %162 : tensor<256xf32>
      %164 = stablehlo.rsqrt %163 : tensor<256xf32>
      mhlo.return %164 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %28 = stablehlo.custom_call @__cudnn$convForward(%26, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>
    %29 = stablehlo.get_tuple_element %28[0] : (tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>) -> tensor<8x64x56x56xf32>
    %30 = "mhlo.fusion"(%arg26, %arg25, %29, %arg177, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<8x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<64xf32>
      %166 = stablehlo.rsqrt %165 : tensor<64xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x64x56x56xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x64x56x56xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x64x56x56xf32>
      mhlo.return %174 : tensor<8x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %31 = stablehlo.custom_call @__cudnn$convForward(%30, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>
    %32 = stablehlo.get_tuple_element %31[0] : (tuple<tensor<8x64x56x56xf32>, tensor<0xui8>>) -> tensor<8x64x56x56xf32>
    %33 = "mhlo.fusion"(%arg29, %arg28, %32, %arg179, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<8x64x56x56xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x64x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<64xf32>
      %166 = stablehlo.rsqrt %165 : tensor<64xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x64x56x56xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x64x56x56xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<8x64x56x56xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x64x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x64x56x56xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x64x56x56xf32>
      mhlo.return %174 : tensor<8x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x64x56x56xf32>
    %34 = stablehlo.custom_call @__cudnn$convForward(%33, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<8x256x56x56xf32>, tensor<0xui8>>
    %35 = stablehlo.get_tuple_element %34[0] : (tuple<tensor<8x256x56x56xf32>, tensor<0xui8>>) -> tensor<8x256x56x56xf32>
    %36 = "mhlo.fusion"(%26, %arg32, %arg31, %27, %35, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x256x56x56xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<8x256x56x56xf32>, %arg273: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %163 = stablehlo.subtract %arg272, %162 : tensor<8x256x56x56xf32>
      %164 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %165 = stablehlo.multiply %163, %164 : tensor<8x256x56x56xf32>
      %166 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %167 = stablehlo.multiply %165, %166 : tensor<8x256x56x56xf32>
      %168 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x56x56xf32>
      %169 = stablehlo.add %167, %168 : tensor<8x256x56x56xf32>
      %170 = stablehlo.add %169, %arg268 : tensor<8x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %171 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x56x56xf32>
      %172 = stablehlo.maximum %170, %171 : tensor<8x256x56x56xf32>
      mhlo.return %172 : tensor<8x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256x56x56xf32>, tensor<256xf32>) -> tensor<8x256x56x56xf32>
    %37 = stablehlo.custom_call @__cudnn$convForward(%36, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<8x128x56x56xf32>, tensor<0xui8>>
    %38 = stablehlo.get_tuple_element %37[0] : (tuple<tensor<8x128x56x56xf32>, tensor<0xui8>>) -> tensor<8x128x56x56xf32>
    %39 = "mhlo.fusion"(%arg38, %arg37, %38, %arg185, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<8x128x56x56xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<8x128x56x56xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x128x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<128xf32>
      %166 = stablehlo.rsqrt %165 : tensor<128xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<128xf32>) -> tensor<8x128x56x56xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x128x56x56xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<8x128x56x56xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x128x56x56xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<8x128x56x56xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x128x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x56x56xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x128x56x56xf32>
      mhlo.return %174 : tensor<8x128x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<8x128x56x56xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x128x56x56xf32>
    %40 = stablehlo.custom_call @__cudnn$convForward(%39, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>
    %41 = stablehlo.get_tuple_element %40[0] : (tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>) -> tensor<8x128x28x28xf32>
    %42 = "mhlo.fusion"(%41, %arg41, %arg40, %arg187, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x128x28x28xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<128xf32>
      %166 = stablehlo.rsqrt %165 : tensor<128xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x128x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x128x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x128x28x28xf32>
      mhlo.return %174 : tensor<8x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %43 = stablehlo.custom_call @__cudnn$convForward(%42, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>
    %44 = stablehlo.get_tuple_element %43[0] : (tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>) -> tensor<8x512x28x28xf32>
    %45 = stablehlo.custom_call @__cudnn$convForward(%36, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>
    %46 = stablehlo.get_tuple_element %45[0] : (tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>) -> tensor<8x512x28x28xf32>
    %47 = "mhlo.fusion"(%44, %arg44, %arg43, %arg189, %arg35, %arg34, %46, %arg183, %arg190, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>, %arg274: tensor<8x512x28x28xf32>, %arg275: tensor<512xf32>, %arg276: tensor<512xf32>, %arg277: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x512x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg276, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x28x28xf32>
      %173 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %174 = stablehlo.subtract %arg274, %173 : tensor<8x512x28x28xf32>
      %175 = stablehlo.add %arg277, %164 : tensor<512xf32>
      %176 = stablehlo.rsqrt %175 : tensor<512xf32>
      %177 = stablehlo.broadcast_in_dim %176, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %178 = stablehlo.multiply %174, %177 : tensor<8x512x28x28xf32>
      %179 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %180 = stablehlo.multiply %178, %179 : tensor<8x512x28x28xf32>
      %181 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %182 = stablehlo.add %180, %181 : tensor<8x512x28x28xf32>
      %183 = stablehlo.add %172, %182 : tensor<8x512x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %184 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x28x28xf32>
      %185 = stablehlo.maximum %183, %184 : tensor<8x512x28x28xf32>
      mhlo.return %185 : tensor<8x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<8x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %48 = stablehlo.custom_call @__cudnn$convForward(%47, %arg45) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>
    %49 = stablehlo.get_tuple_element %48[0] : (tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>) -> tensor<8x128x28x28xf32>
    %50 = "mhlo.fusion"(%49, %arg47, %arg46, %arg191, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x128x28x28xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<128xf32>
      %166 = stablehlo.rsqrt %165 : tensor<128xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x128x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x128x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x128x28x28xf32>
      mhlo.return %174 : tensor<8x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %51 = stablehlo.custom_call @__cudnn$convForward(%50, %arg48) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>
    %52 = stablehlo.get_tuple_element %51[0] : (tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>) -> tensor<8x128x28x28xf32>
    %53 = "mhlo.fusion"(%52, %arg50, %arg49, %arg193, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x128x28x28xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<128xf32>
      %166 = stablehlo.rsqrt %165 : tensor<128xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x128x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x128x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x128x28x28xf32>
      mhlo.return %174 : tensor<8x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %54 = stablehlo.custom_call @__cudnn$convForward(%53, %arg51) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>
    %55 = stablehlo.get_tuple_element %54[0] : (tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>) -> tensor<8x512x28x28xf32>
    %56 = "mhlo.fusion"(%47, %55, %arg53, %arg52, %arg195, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x512x28x28xf32>, %arg269: tensor<8x512x28x28xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %163 = stablehlo.subtract %arg269, %162 : tensor<8x512x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x28x28xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x512x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x28x28xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x512x28x28xf32>
      mhlo.return %175 : tensor<8x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512x28x28xf32>, tensor<8x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %57 = stablehlo.custom_call @__cudnn$convForward(%56, %arg54) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>
    %58 = stablehlo.get_tuple_element %57[0] : (tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>) -> tensor<8x128x28x28xf32>
    %59 = "mhlo.fusion"(%58, %arg56, %arg55, %arg197, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x128x28x28xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<128xf32>
      %166 = stablehlo.rsqrt %165 : tensor<128xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x128x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x128x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x128x28x28xf32>
      mhlo.return %174 : tensor<8x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %60 = stablehlo.custom_call @__cudnn$convForward(%59, %arg57) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>
    %61 = stablehlo.get_tuple_element %60[0] : (tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>) -> tensor<8x128x28x28xf32>
    %62 = "mhlo.fusion"(%61, %arg59, %arg58, %arg199, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x128x28x28xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<128xf32>
      %166 = stablehlo.rsqrt %165 : tensor<128xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x128x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x128x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x128x28x28xf32>
      mhlo.return %174 : tensor<8x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %63 = stablehlo.custom_call @__cudnn$convForward(%62, %arg60) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>
    %64 = stablehlo.get_tuple_element %63[0] : (tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>) -> tensor<8x512x28x28xf32>
    %65 = "mhlo.fusion"(%56, %64, %arg62, %arg61, %arg201, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x512x28x28xf32>, %arg269: tensor<8x512x28x28xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %163 = stablehlo.subtract %arg269, %162 : tensor<8x512x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x28x28xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x512x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x28x28xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x512x28x28xf32>
      mhlo.return %175 : tensor<8x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512x28x28xf32>, tensor<8x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %66 = stablehlo.custom_call @__cudnn$convForward(%65, %arg63) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>
    %67 = stablehlo.get_tuple_element %66[0] : (tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>) -> tensor<8x128x28x28xf32>
    %68 = "mhlo.fusion"(%67, %arg65, %arg64, %arg203, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x128x28x28xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<128xf32>
      %166 = stablehlo.rsqrt %165 : tensor<128xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x128x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x128x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x128x28x28xf32>
      mhlo.return %174 : tensor<8x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %69 = stablehlo.custom_call @__cudnn$convForward(%68, %arg66) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>
    %70 = stablehlo.get_tuple_element %69[0] : (tuple<tensor<8x128x28x28xf32>, tensor<0xui8>>) -> tensor<8x128x28x28xf32>
    %71 = "mhlo.fusion"(%70, %arg68, %arg67, %arg205, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x128x28x28xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x128x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<128xf32>
      %166 = stablehlo.rsqrt %165 : tensor<128xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x128x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x128x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<8x128x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x128x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x128x28x28xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x128x28x28xf32>
      mhlo.return %174 : tensor<8x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x128x28x28xf32>
    %72 = stablehlo.custom_call @__cudnn$convForward(%71, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>
    %73 = stablehlo.get_tuple_element %72[0] : (tuple<tensor<8x512x28x28xf32>, tensor<0xui8>>) -> tensor<8x512x28x28xf32>
    %74 = "mhlo.fusion"(%65, %73, %arg71, %arg70, %arg207, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x512x28x28xf32>, %arg269: tensor<8x512x28x28xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %163 = stablehlo.subtract %arg269, %162 : tensor<8x512x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<8x512x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x28x28xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x512x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x28x28xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x512x28x28xf32>
      mhlo.return %175 : tensor<8x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512x28x28xf32>, tensor<8x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x28x28xf32>
    %75 = stablehlo.custom_call @__cudnn$convForward(%74, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>
    %76 = stablehlo.get_tuple_element %75[0] : (tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>) -> tensor<8x1024x14x14xf32>
    %77 = stablehlo.custom_call @__cudnn$convForward(%74, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<8x256x28x28xf32>, tensor<0xui8>>
    %78 = stablehlo.get_tuple_element %77[0] : (tuple<tensor<8x256x28x28xf32>, tensor<0xui8>>) -> tensor<8x256x28x28xf32>
    %79 = "mhlo.fusion"(%arg77, %arg76, %78, %arg211, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x28x28xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x28x28xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x28x28xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x28x28xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x28x28xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x28x28xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x28x28xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x28x28xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x28x28xf32>
      mhlo.return %174 : tensor<8x256x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x28x28xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x28x28xf32>
    %80 = stablehlo.custom_call @__cudnn$convForward(%79, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %81 = stablehlo.get_tuple_element %80[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %82 = "mhlo.fusion"(%arg80, %arg79, %81, %arg213, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %83 = stablehlo.custom_call @__cudnn$convForward(%82, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>
    %84 = stablehlo.get_tuple_element %83[0] : (tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>) -> tensor<8x1024x14x14xf32>
    %85 = "mhlo.fusion"(%arg74, %arg73, %76, %arg209, %arg83, %arg82, %84, %arg215, %arg216, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<8x1024x14x14xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<8x1024x14x14xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<1024xf32>, %arg277: tensor<1024xf32>):
      %162 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %163 = stablehlo.subtract %arg274, %162 : tensor<8x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %165 = stablehlo.add %arg276, %164 : tensor<1024xf32>
      %166 = stablehlo.rsqrt %165 : tensor<1024xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x1024x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x1024x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x1024x14x14xf32>
      %173 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %174 = stablehlo.subtract %arg270, %173 : tensor<8x1024x14x14xf32>
      %175 = stablehlo.add %arg277, %164 : tensor<1024xf32>
      %176 = stablehlo.rsqrt %175 : tensor<1024xf32>
      %177 = stablehlo.broadcast_in_dim %176, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %178 = stablehlo.multiply %174, %177 : tensor<8x1024x14x14xf32>
      %179 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %180 = stablehlo.multiply %178, %179 : tensor<8x1024x14x14xf32>
      %181 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %182 = stablehlo.add %180, %181 : tensor<8x1024x14x14xf32>
      %183 = stablehlo.add %172, %182 : tensor<8x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %184 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
      %185 = stablehlo.maximum %183, %184 : tensor<8x1024x14x14xf32>
      mhlo.return %185 : tensor<8x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %86 = stablehlo.custom_call @__cudnn$convForward(%85, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %87 = stablehlo.get_tuple_element %86[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %88 = "mhlo.fusion"(%arg86, %arg85, %87, %arg217, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %89 = stablehlo.custom_call @__cudnn$convForward(%88, %arg87) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %90 = stablehlo.get_tuple_element %89[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %91 = "mhlo.fusion"(%arg89, %arg88, %90, %arg219, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %92 = stablehlo.custom_call @__cudnn$convForward(%91, %arg90) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>
    %93 = stablehlo.get_tuple_element %92[0] : (tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>) -> tensor<8x1024x14x14xf32>
    %94 = "mhlo.fusion"(%85, %arg92, %arg91, %93, %arg221, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<8x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %163 = stablehlo.subtract %arg271, %162 : tensor<8x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<1024xf32>
      %166 = stablehlo.rsqrt %165 : tensor<1024xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x1024x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x1024x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x1024x14x14xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x1024x14x14xf32>
      mhlo.return %175 : tensor<8x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %95 = stablehlo.custom_call @__cudnn$convForward(%94, %arg93) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %96 = stablehlo.get_tuple_element %95[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %97 = "mhlo.fusion"(%arg95, %arg94, %96, %arg223, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %98 = stablehlo.custom_call @__cudnn$convForward(%97, %arg96) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %99 = stablehlo.get_tuple_element %98[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %100 = "mhlo.fusion"(%arg98, %arg97, %99, %arg225, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %101 = stablehlo.custom_call @__cudnn$convForward(%100, %arg99) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>
    %102 = stablehlo.get_tuple_element %101[0] : (tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>) -> tensor<8x1024x14x14xf32>
    %103 = "mhlo.fusion"(%94, %arg101, %arg100, %102, %arg227, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<8x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %163 = stablehlo.subtract %arg271, %162 : tensor<8x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<1024xf32>
      %166 = stablehlo.rsqrt %165 : tensor<1024xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x1024x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x1024x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x1024x14x14xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x1024x14x14xf32>
      mhlo.return %175 : tensor<8x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %104 = stablehlo.custom_call @__cudnn$convForward(%103, %arg102) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %105 = stablehlo.get_tuple_element %104[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %106 = "mhlo.fusion"(%arg104, %arg103, %105, %arg229, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %107 = stablehlo.custom_call @__cudnn$convForward(%106, %arg105) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %108 = stablehlo.get_tuple_element %107[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %109 = "mhlo.fusion"(%arg107, %arg106, %108, %arg231, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %110 = stablehlo.custom_call @__cudnn$convForward(%109, %arg108) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>
    %111 = stablehlo.get_tuple_element %110[0] : (tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>) -> tensor<8x1024x14x14xf32>
    %112 = "mhlo.fusion"(%103, %arg110, %arg109, %111, %arg233, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<8x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %163 = stablehlo.subtract %arg271, %162 : tensor<8x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<1024xf32>
      %166 = stablehlo.rsqrt %165 : tensor<1024xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x1024x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x1024x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x1024x14x14xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x1024x14x14xf32>
      mhlo.return %175 : tensor<8x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %113 = stablehlo.custom_call @__cudnn$convForward(%112, %arg111) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %114 = stablehlo.get_tuple_element %113[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %115 = "mhlo.fusion"(%arg113, %arg112, %114, %arg235, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %116 = stablehlo.custom_call @__cudnn$convForward(%115, %arg114) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %117 = stablehlo.get_tuple_element %116[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %118 = "mhlo.fusion"(%arg116, %arg115, %117, %arg237, %arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %119 = stablehlo.custom_call @__cudnn$convForward(%118, %arg117) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>
    %120 = stablehlo.get_tuple_element %119[0] : (tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>) -> tensor<8x1024x14x14xf32>
    %121 = "mhlo.fusion"(%112, %arg119, %arg118, %120, %arg239, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<8x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %163 = stablehlo.subtract %arg271, %162 : tensor<8x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<1024xf32>
      %166 = stablehlo.rsqrt %165 : tensor<1024xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x1024x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x1024x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x1024x14x14xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x1024x14x14xf32>
      mhlo.return %175 : tensor<8x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %122 = stablehlo.custom_call @__cudnn$convForward(%121, %arg120) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %123 = stablehlo.get_tuple_element %122[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %124 = "mhlo.fusion"(%arg122, %arg121, %123, %arg241, %arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %125 = stablehlo.custom_call @__cudnn$convForward(%124, %arg123) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>
    %126 = stablehlo.get_tuple_element %125[0] : (tuple<tensor<8x256x14x14xf32>, tensor<0xui8>>) -> tensor<8x256x14x14xf32>
    %127 = "mhlo.fusion"(%arg125, %arg124, %126, %arg243, %arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<8x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<256xf32>
      %166 = stablehlo.rsqrt %165 : tensor<256xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x256x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x256x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<8x256x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x256x14x14xf32>
      mhlo.return %174 : tensor<8x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x14x14xf32>
    %128 = stablehlo.custom_call @__cudnn$convForward(%127, %arg126) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>
    %129 = stablehlo.get_tuple_element %128[0] : (tuple<tensor<8x1024x14x14xf32>, tensor<0xui8>>) -> tensor<8x1024x14x14xf32>
    %130 = "mhlo.fusion"(%121, %arg128, %arg127, %129, %arg245, %arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<8x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %163 = stablehlo.subtract %arg271, %162 : tensor<8x1024x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<1024xf32>
      %166 = stablehlo.rsqrt %165 : tensor<1024xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x1024x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x1024x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x1024x14x14xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x1024x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1024x14x14xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x1024x14x14xf32>
      mhlo.return %175 : tensor<8x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<8x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<8x1024x14x14xf32>
    %131 = stablehlo.custom_call @__cudnn$convForward(%130, %arg129) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tuple<tensor<8x2048x7x7xf32>, tensor<0xui8>>
    %132 = stablehlo.get_tuple_element %131[0] : (tuple<tensor<8x2048x7x7xf32>, tensor<0xui8>>) -> tensor<8x2048x7x7xf32>
    %133 = stablehlo.custom_call @__cudnn$convForward(%130, %arg132) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<8x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tuple<tensor<8x512x14x14xf32>, tensor<0xui8>>
    %134 = stablehlo.get_tuple_element %133[0] : (tuple<tensor<8x512x14x14xf32>, tensor<0xui8>>) -> tensor<8x512x14x14xf32>
    %135 = "mhlo.fusion"(%134, %arg134, %arg133, %arg249, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x512x14x14xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x14x14xf32>
      %163 = stablehlo.subtract %arg268, %162 : tensor<8x512x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x14x14xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x14x14xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<8x512x14x14xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x14x14xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<8x512x14x14xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x14x14xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x512x14x14xf32>
      mhlo.return %174 : tensor<8x512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512x14x14xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x14x14xf32>
    %136 = stablehlo.custom_call @__cudnn$convForward(%135, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>
    %137 = stablehlo.get_tuple_element %136[0] : (tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>) -> tensor<8x512x7x7xf32>
    %138 = "mhlo.fusion"(%arg137, %arg136, %137, %arg251, %arg252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<8x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x7x7xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x512x7x7xf32>
      mhlo.return %174 : tensor<8x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %139 = stablehlo.custom_call @__cudnn$convForward(%138, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<8x2048x7x7xf32>, tensor<0xui8>>
    %140 = stablehlo.get_tuple_element %139[0] : (tuple<tensor<8x2048x7x7xf32>, tensor<0xui8>>) -> tensor<8x2048x7x7xf32>
    %141 = "mhlo.fusion"(%arg131, %arg130, %132, %arg247, %arg140, %arg139, %140, %arg253, %arg254, %arg248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<8x2048x7x7xf32>, %arg271: tensor<2048xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>, %arg274: tensor<8x2048x7x7xf32>, %arg275: tensor<2048xf32>, %arg276: tensor<2048xf32>, %arg277: tensor<2048xf32>):
      %162 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %163 = stablehlo.subtract %arg274, %162 : tensor<8x2048x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %165 = stablehlo.add %arg276, %164 : tensor<2048xf32>
      %166 = stablehlo.rsqrt %165 : tensor<2048xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x2048x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x2048x7x7xf32>
      %171 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x2048x7x7xf32>
      %173 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %174 = stablehlo.subtract %arg270, %173 : tensor<8x2048x7x7xf32>
      %175 = stablehlo.add %arg277, %164 : tensor<2048xf32>
      %176 = stablehlo.rsqrt %175 : tensor<2048xf32>
      %177 = stablehlo.broadcast_in_dim %176, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %178 = stablehlo.multiply %174, %177 : tensor<8x2048x7x7xf32>
      %179 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %180 = stablehlo.multiply %178, %179 : tensor<8x2048x7x7xf32>
      %181 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %182 = stablehlo.add %180, %181 : tensor<8x2048x7x7xf32>
      %183 = stablehlo.add %172, %182 : tensor<8x2048x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %184 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x2048x7x7xf32>
      %185 = stablehlo.maximum %183, %184 : tensor<8x2048x7x7xf32>
      mhlo.return %185 : tensor<8x2048x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<8x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<8x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %142 = stablehlo.custom_call @__cudnn$convForward(%141, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>
    %143 = stablehlo.get_tuple_element %142[0] : (tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>) -> tensor<8x512x7x7xf32>
    %144 = "mhlo.fusion"(%arg143, %arg142, %143, %arg255, %arg256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<8x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x7x7xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x512x7x7xf32>
      mhlo.return %174 : tensor<8x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %145 = stablehlo.custom_call @__cudnn$convForward(%144, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>
    %146 = stablehlo.get_tuple_element %145[0] : (tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>) -> tensor<8x512x7x7xf32>
    %147 = "mhlo.fusion"(%arg146, %arg145, %146, %arg257, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<8x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x7x7xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x512x7x7xf32>
      mhlo.return %174 : tensor<8x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %148 = stablehlo.custom_call @__cudnn$convForward(%147, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<8x2048x7x7xf32>, tensor<0xui8>>
    %149 = stablehlo.get_tuple_element %148[0] : (tuple<tensor<8x2048x7x7xf32>, tensor<0xui8>>) -> tensor<8x2048x7x7xf32>
    %150 = "mhlo.fusion"(%141, %arg149, %arg148, %149, %arg259, %arg260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<8x2048x7x7xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<8x2048x7x7xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %163 = stablehlo.subtract %arg271, %162 : tensor<8x2048x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<2048xf32>
      %166 = stablehlo.rsqrt %165 : tensor<2048xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x2048x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x2048x7x7xf32>
      %171 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x2048x7x7xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x2048x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x2048x7x7xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x2048x7x7xf32>
      mhlo.return %175 : tensor<8x2048x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<8x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
    %151 = stablehlo.custom_call @__cudnn$convForward(%150, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>
    %152 = stablehlo.get_tuple_element %151[0] : (tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>) -> tensor<8x512x7x7xf32>
    %153 = "mhlo.fusion"(%arg152, %arg151, %152, %arg261, %arg262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<8x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x7x7xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x512x7x7xf32>
      mhlo.return %174 : tensor<8x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %154 = stablehlo.custom_call @__cudnn$convForward(%153, %arg153) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>
    %155 = stablehlo.get_tuple_element %154[0] : (tuple<tensor<8x512x7x7xf32>, tensor<0xui8>>) -> tensor<8x512x7x7xf32>
    %156 = "mhlo.fusion"(%arg155, %arg154, %155, %arg263, %arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<8x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %162 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %163 = stablehlo.subtract %arg270, %162 : tensor<8x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %165 = stablehlo.add %arg272, %164 : tensor<512xf32>
      %166 = stablehlo.rsqrt %165 : tensor<512xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x512x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x512x7x7xf32>
      %171 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<8x512x7x7xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x7x7xf32>
      %174 = stablehlo.maximum %172, %173 : tensor<8x512x7x7xf32>
      mhlo.return %174 : tensor<8x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x7x7xf32>
    %157 = stablehlo.custom_call @__cudnn$convForward(%156, %arg156) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<8x2048x7x7xf32>, tensor<0xui8>>
    %158 = stablehlo.get_tuple_element %157[0] : (tuple<tensor<8x2048x7x7xf32>, tensor<0xui8>>) -> tensor<8x2048x7x7xf32>
    %159 = "mhlo.fusion"(%150, %158, %arg158, %arg157, %arg265, %arg266) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg268: tensor<8x2048x7x7xf32>, %arg269: tensor<8x2048x7x7xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<2048xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>):
      %162 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %163 = stablehlo.subtract %arg269, %162 : tensor<8x2048x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %165 = stablehlo.add %arg273, %164 : tensor<2048xf32>
      %166 = stablehlo.rsqrt %165 : tensor<2048xf32>
      %167 = stablehlo.broadcast_in_dim %166, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %168 = stablehlo.multiply %163, %167 : tensor<8x2048x7x7xf32>
      %169 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<8x2048x7x7xf32>
      %171 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<8x2048x7x7xf32>
      %172 = stablehlo.add %170, %171 : tensor<8x2048x7x7xf32>
      %173 = stablehlo.add %172, %arg268 : tensor<8x2048x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %174 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x2048x7x7xf32>
      %175 = stablehlo.maximum %173, %174 : tensor<8x2048x7x7xf32>
      %176 = mhlo.bitcast %175 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2048x7x7xf32>) -> tensor<8x2048x49xf32>
      %177 = stablehlo.reduce(%176 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x2048x49xf32>, tensor<f32>) -> tensor<8x2048xf32>
      mhlo.return %177 : tensor<8x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2048x7x7xf32>, tensor<8x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<8x2048xf32>
    %160 = "mhlo.fusion"(%159, %arg159) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg268: tensor<8x2048xf32>, %arg269: tensor<1000x2048xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %162 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x2048xf32>
      %163 = stablehlo.multiply %arg268, %162 : tensor<8x2048xf32>
      %164 = stablehlo.dot_general %arg269, %163, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1000,8]{0,1}"} : (tensor<1000x2048xf32>, tensor<8x2048xf32>) -> tensor<1000x8xf32>
      mhlo.return %164 : tensor<1000x8xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1000,8]{0,1}"} : (tensor<8x2048xf32>, tensor<1000x2048xf32>) -> tensor<1000x8xf32>
    %161 = "mhlo.fusion"(%160, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1000x8xf32>, %arg269: tensor<1000xf32>):
      %162 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1000xf32>) -> tensor<8x1000xf32>
      %163 = mhlo.bitcast %arg268 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x8xf32>) -> tensor<8x1000xf32>
      %164 = stablehlo.add %162, %163 : tensor<8x1000xf32>
      mhlo.return %164 : tensor<8x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x8xf32>, tensor<1000xf32>) -> tensor<8x1000xf32>
    return %161 : tensor<8x1000xf32>
  }
}
