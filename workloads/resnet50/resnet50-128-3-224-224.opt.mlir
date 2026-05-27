module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.1_computation(%arg0: tensor<128x2048xf32>, %arg1: tensor<1000x2048xf32>) -> tensor<128x1000xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x2048xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x2048xf32>, tensor<1000x2048xf32>) -> tensor<128x1000xf32>
    return %2 : tensor<128x1000xf32>
  }
  func.func private @region_1.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<128x2048x7x7xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<128x2048x7x7xf32>, %arg5: tensor<2048xf32>) -> tensor<128x2048xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x2048x7x7xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x2048x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x2048x7x7xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x2048x7x7xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x2048x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048x7x7xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x2048x7x7xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2048x7x7xf32>) -> tensor<128x2048x49xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x2048x49xf32>, tensor<f32>) -> tensor<128x2048xf32>
    return %12 : tensor<128x2048xf32>
  }
  func.func private @fused_maximum(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x56x56xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<128x256x56x56xf32>, %arg9: tensor<256xf32>) -> tensor<128x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x56x56xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x256x56x56xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x256x56x56xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x256x56x56xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x256x56x56xf32>
    %16 = stablehlo.add %7, %15 : tensor<128x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x56x56xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<128x256x56x56xf32>
    return %18 : tensor<128x256x56x56xf32>
  }
  func.func private @fused_maximum.1(%arg0: tensor<128x256x56x56xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<128x256x56x56xf32>, %arg5: tensor<256xf32>) -> tensor<128x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x56x56xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x56x56xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x256x56x56xf32>
    return %10 : tensor<128x256x56x56xf32>
  }
  func.func private @fused_maximum.2(%arg0: tensor<128x256x56x56xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<128x256x56x56xf32>, %arg5: tensor<256xf32>) -> tensor<128x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x56x56xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x56x56xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x256x56x56xf32>
    return %10 : tensor<128x256x56x56xf32>
  }
  func.func private @fused_maximum.3(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x512x28x28xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512xf32>, %arg8: tensor<128x512x28x28xf32>, %arg9: tensor<512xf32>) -> tensor<128x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<128x512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x512x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x512x28x28xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<128x512x28x28xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x512x28x28xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x512x28x28xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x512x28x28xf32>
    %16 = stablehlo.add %7, %15 : tensor<128x512x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x28x28xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<128x512x28x28xf32>
    return %18 : tensor<128x512x28x28xf32>
  }
  func.func private @fused_maximum.4(%arg0: tensor<128x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<128x512x28x28xf32>, %arg5: tensor<512xf32>) -> tensor<128x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x512x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x512x28x28xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x512x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x28x28xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x512x28x28xf32>
    return %10 : tensor<128x512x28x28xf32>
  }
  func.func private @fused_maximum.5(%arg0: tensor<128x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<128x512x28x28xf32>, %arg5: tensor<512xf32>) -> tensor<128x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x512x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x512x28x28xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x512x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x28x28xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x512x28x28xf32>
    return %10 : tensor<128x512x28x28xf32>
  }
  func.func private @fused_maximum.6(%arg0: tensor<128x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<128x512x28x28xf32>, %arg5: tensor<512xf32>) -> tensor<128x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x512x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x512x28x28xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x512x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x28x28xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x512x28x28xf32>
    return %10 : tensor<128x512x28x28xf32>
  }
  func.func private @fused_maximum.7(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<128x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<128x1024x14x14xf32>, %arg9: tensor<1024xf32>) -> tensor<128x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x1024x14x14xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x1024x14x14xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x1024x14x14xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x1024x14x14xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x1024x14x14xf32>
    %16 = stablehlo.add %7, %15 : tensor<128x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<128x1024x14x14xf32>
    return %18 : tensor<128x1024x14x14xf32>
  }
  func.func private @fused_maximum.8(%arg0: tensor<128x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<128x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<128x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x1024x14x14xf32>
    return %10 : tensor<128x1024x14x14xf32>
  }
  func.func private @fused_maximum.9(%arg0: tensor<128x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<128x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<128x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x1024x14x14xf32>
    return %10 : tensor<128x1024x14x14xf32>
  }
  func.func private @fused_maximum.10(%arg0: tensor<128x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<128x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<128x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x1024x14x14xf32>
    return %10 : tensor<128x1024x14x14xf32>
  }
  func.func private @fused_maximum.11(%arg0: tensor<128x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<128x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<128x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x1024x14x14xf32>
    return %10 : tensor<128x1024x14x14xf32>
  }
  func.func private @fused_maximum.12(%arg0: tensor<128x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<128x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<128x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x1024x14x14xf32>
    return %10 : tensor<128x1024x14x14xf32>
  }
  func.func private @fused_maximum.13(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<128x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x64x56x56xf32>
    return %9 : tensor<128x64x56x56xf32>
  }
  func.func private @fused_maximum.14(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<128x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x64x56x56xf32>
    return %9 : tensor<128x64x56x56xf32>
  }
  func.func private @fused_maximum.15(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<128x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x64x56x56xf32>
    return %9 : tensor<128x64x56x56xf32>
  }
  func.func private @fused_maximum.16(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<128x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x64x56x56xf32>
    return %9 : tensor<128x64x56x56xf32>
  }
  func.func private @fused_maximum.17(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<128x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x64x56x56xf32>
    return %9 : tensor<128x64x56x56xf32>
  }
  func.func private @fused_maximum.18(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<128x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x64x56x56xf32>
    return %9 : tensor<128x64x56x56xf32>
  }
  func.func private @region_0.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_window(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x64x112x112xf32>, %arg4: tensor<64xf32>) -> tensor<128x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x64x112x112xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x64x112x112xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x64x112x112xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x64x112x112xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x112x112xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x64x112x112xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %10 = "stablehlo.reduce_window"(%9, %cst_0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %11 = stablehlo.maximum %arg5, %arg6 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x56x56xf32>
    return %10 : tensor<128x64x56x56xf32>
  }
  func.func private @fused_maximum.19(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128x56x56xf32>, %arg4: tensor<128xf32>) -> tensor<128x128x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<128x128x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x128x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x128x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x128x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<128x128x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x128x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x128x56x56xf32>
    return %9 : tensor<128x128x56x56xf32>
  }
  func.func private @fused_maximum.20(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x128x28x28xf32>
    return %9 : tensor<128x128x28x28xf32>
  }
  func.func private @fused_maximum.21(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x128x28x28xf32>
    return %9 : tensor<128x128x28x28xf32>
  }
  func.func private @fused_maximum.22(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x128x28x28xf32>
    return %9 : tensor<128x128x28x28xf32>
  }
  func.func private @fused_maximum.23(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x128x28x28xf32>
    return %9 : tensor<128x128x28x28xf32>
  }
  func.func private @fused_maximum.24(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x128x28x28xf32>
    return %9 : tensor<128x128x28x28xf32>
  }
  func.func private @fused_maximum.25(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x128x28x28xf32>
    return %9 : tensor<128x128x28x28xf32>
  }
  func.func private @fused_maximum.26(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x128x28x28xf32>
    return %9 : tensor<128x128x28x28xf32>
  }
  func.func private @fused_maximum.27(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.28(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.29(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.30(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.31(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.32(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.33(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.34(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.35(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.36(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.37(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x14x14xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x14x14xf32>
    return %9 : tensor<128x256x14x14xf32>
  }
  func.func private @fused_maximum.38(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x28x28xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x28x28xf32>
    return %9 : tensor<128x256x28x28xf32>
  }
  func.func private @fused_maximum.39(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<128x2048x7x7xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<2048xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<128x2048x7x7xf32>, %arg9: tensor<2048xf32>) -> tensor<128x2048x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x2048x7x7xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x2048x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x2048x7x7xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x2048x7x7xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x2048x7x7xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x2048x7x7xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x2048x7x7xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x2048x7x7xf32>
    %16 = stablehlo.add %7, %15 : tensor<128x2048x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048x7x7xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<128x2048x7x7xf32>
    return %18 : tensor<128x2048x7x7xf32>
  }
  func.func private @fused_maximum.40(%arg0: tensor<128x2048x7x7xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<128x2048x7x7xf32>, %arg5: tensor<2048xf32>) -> tensor<128x2048x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<128x2048x7x7xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x2048x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x2048x7x7xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x2048x7x7xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<128x2048x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048x7x7xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<128x2048x7x7xf32>
    return %10 : tensor<128x2048x7x7xf32>
  }
  func.func private @fused_maximum.41(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x512x14x14xf32>, %arg4: tensor<512xf32>) -> tensor<128x512x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<128x512x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x512x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<128x512x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x512x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<128x512x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x512x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x512x14x14xf32>
    return %9 : tensor<128x512x14x14xf32>
  }
  func.func private @fused_maximum.42(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<128x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<128x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<128x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<128x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<128x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<128x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<128x512x7x7xf32>
    return %12 : tensor<128x512x7x7xf32>
  }
  func.func private @fused_maximum.43(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<128x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<128x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<128x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<128x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<128x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<128x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<128x512x7x7xf32>
    return %12 : tensor<128x512x7x7xf32>
  }
  func.func private @fused_maximum.44(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<128x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<128x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<128x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<128x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<128x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<128x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<128x512x7x7xf32>
    return %12 : tensor<128x512x7x7xf32>
  }
  func.func private @fused_maximum.45(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<128x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<128x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<128x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<128x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<128x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<128x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<128x512x7x7xf32>
    return %12 : tensor<128x512x7x7xf32>
  }
  func.func private @fused_maximum.46(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<128x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<128x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<128x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<128x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<128x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<128x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<128x512x7x7xf32>
    return %12 : tensor<128x512x7x7xf32>
  }
  func.func private @fused_add(%arg0: tensor<128x1000xf32>, %arg1: tensor<1000xf32>) -> tensor<128x1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1000xf32>) -> tensor<128x1000xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<128x1000xf32>
    return %1 : tensor<128x1000xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
  func.func private @fused_rsqrt.1(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
  func.func private @fused_rsqrt.2(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
  func.func private @fused_rsqrt.3(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
  func.func private @fused_rsqrt.4(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.5(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.6(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.7(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.8(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.9(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.10(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.11(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.12(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.13(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.14(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.15(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.16(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.17(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.18(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.19(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.20(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.21(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.22(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.23(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.24(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.25(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.26(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.27(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.28(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.29(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.30(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.31(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.32(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.33(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.34(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.35(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.36(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.37(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.38(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.39(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.40(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.41(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.42(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.43(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.44(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.45(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.46(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.47(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64x1x1xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<64x64x1x1xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<256x64x1x1xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<64x256x1x1xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64x64x3x3xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<256x64x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<64x256x1x1xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<64x64x3x3xf32>, %arg28: tensor<64xf32>, %arg29: tensor<64xf32>, %arg30: tensor<256x64x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<512x256x1x1xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<128x256x1x1xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128x128x3x3xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<512x128x1x1xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<128x512x1x1xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128x128x3x3xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<512x128x1x1xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512xf32>, %arg54: tensor<128x512x1x1xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128x128x3x3xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<512x128x1x1xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512xf32>, %arg63: tensor<128x512x1x1xf32>, %arg64: tensor<128xf32>, %arg65: tensor<128xf32>, %arg66: tensor<128x128x3x3xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<512x128x1x1xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<1024x512x1x1xf32>, %arg73: tensor<1024xf32>, %arg74: tensor<1024xf32>, %arg75: tensor<256x512x1x1xf32>, %arg76: tensor<256xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x256x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<256xf32>, %arg81: tensor<1024x256x1x1xf32>, %arg82: tensor<1024xf32>, %arg83: tensor<1024xf32>, %arg84: tensor<256x1024x1x1xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256x256x3x3xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<1024x256x1x1xf32>, %arg91: tensor<1024xf32>, %arg92: tensor<1024xf32>, %arg93: tensor<256x1024x1x1xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256x256x3x3xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<1024x256x1x1xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<1024xf32>, %arg102: tensor<256x1024x1x1xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256x256x3x3xf32>, %arg106: tensor<256xf32>, %arg107: tensor<256xf32>, %arg108: tensor<1024x256x1x1xf32>, %arg109: tensor<1024xf32>, %arg110: tensor<1024xf32>, %arg111: tensor<256x1024x1x1xf32>, %arg112: tensor<256xf32>, %arg113: tensor<256xf32>, %arg114: tensor<256x256x3x3xf32>, %arg115: tensor<256xf32>, %arg116: tensor<256xf32>, %arg117: tensor<1024x256x1x1xf32>, %arg118: tensor<1024xf32>, %arg119: tensor<1024xf32>, %arg120: tensor<256x1024x1x1xf32>, %arg121: tensor<256xf32>, %arg122: tensor<256xf32>, %arg123: tensor<256x256x3x3xf32>, %arg124: tensor<256xf32>, %arg125: tensor<256xf32>, %arg126: tensor<1024x256x1x1xf32>, %arg127: tensor<1024xf32>, %arg128: tensor<1024xf32>, %arg129: tensor<2048x1024x1x1xf32>, %arg130: tensor<2048xf32>, %arg131: tensor<2048xf32>, %arg132: tensor<512x1024x1x1xf32>, %arg133: tensor<512xf32>, %arg134: tensor<512xf32>, %arg135: tensor<512x512x3x3xf32>, %arg136: tensor<512xf32>, %arg137: tensor<512xf32>, %arg138: tensor<2048x512x1x1xf32>, %arg139: tensor<2048xf32>, %arg140: tensor<2048xf32>, %arg141: tensor<512x2048x1x1xf32>, %arg142: tensor<512xf32>, %arg143: tensor<512xf32>, %arg144: tensor<512x512x3x3xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<2048x512x1x1xf32>, %arg148: tensor<2048xf32>, %arg149: tensor<2048xf32>, %arg150: tensor<512x2048x1x1xf32>, %arg151: tensor<512xf32>, %arg152: tensor<512xf32>, %arg153: tensor<512x512x3x3xf32>, %arg154: tensor<512xf32>, %arg155: tensor<512xf32>, %arg156: tensor<2048x512x1x1xf32>, %arg157: tensor<2048xf32>, %arg158: tensor<2048xf32>, %arg159: tensor<1000x2048xf32>, %arg160: tensor<1000xf32>, %arg161: tensor<64xf32>, %arg162: tensor<64xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<64xf32>, %arg166: tensor<64xf32>, %arg167: tensor<64xf32>, %arg168: tensor<64xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<64xf32>, %arg174: tensor<64xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<64xf32>, %arg178: tensor<64xf32>, %arg179: tensor<64xf32>, %arg180: tensor<64xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512xf32>, %arg184: tensor<512xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<128xf32>, %arg188: tensor<128xf32>, %arg189: tensor<512xf32>, %arg190: tensor<512xf32>, %arg191: tensor<128xf32>, %arg192: tensor<128xf32>, %arg193: tensor<128xf32>, %arg194: tensor<128xf32>, %arg195: tensor<512xf32>, %arg196: tensor<512xf32>, %arg197: tensor<128xf32>, %arg198: tensor<128xf32>, %arg199: tensor<128xf32>, %arg200: tensor<128xf32>, %arg201: tensor<512xf32>, %arg202: tensor<512xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<512xf32>, %arg208: tensor<512xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1024xf32>, %arg211: tensor<256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<1024xf32>, %arg216: tensor<1024xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<1024xf32>, %arg222: tensor<1024xf32>, %arg223: tensor<256xf32>, %arg224: tensor<256xf32>, %arg225: tensor<256xf32>, %arg226: tensor<256xf32>, %arg227: tensor<1024xf32>, %arg228: tensor<1024xf32>, %arg229: tensor<256xf32>, %arg230: tensor<256xf32>, %arg231: tensor<256xf32>, %arg232: tensor<256xf32>, %arg233: tensor<1024xf32>, %arg234: tensor<1024xf32>, %arg235: tensor<256xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<256xf32>, %arg239: tensor<1024xf32>, %arg240: tensor<1024xf32>, %arg241: tensor<256xf32>, %arg242: tensor<256xf32>, %arg243: tensor<256xf32>, %arg244: tensor<256xf32>, %arg245: tensor<1024xf32>, %arg246: tensor<1024xf32>, %arg247: tensor<2048xf32>, %arg248: tensor<2048xf32>, %arg249: tensor<512xf32>, %arg250: tensor<512xf32>, %arg251: tensor<512xf32>, %arg252: tensor<512xf32>, %arg253: tensor<2048xf32>, %arg254: tensor<2048xf32>, %arg255: tensor<512xf32>, %arg256: tensor<512xf32>, %arg257: tensor<512xf32>, %arg258: tensor<512xf32>, %arg259: tensor<2048xf32>, %arg260: tensor<2048xf32>, %arg261: tensor<512xf32>, %arg262: tensor<512xf32>, %arg263: tensor<512xf32>, %arg264: tensor<512xf32>, %arg265: tensor<2048xf32>, %arg266: tensor<2048xf32>, %arg267: tensor<128x3x224x224xf32>) -> tensor<128x1000xf32> {
    %0 = "mhlo.fusion"(%arg254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<2048xf32>
      %207 = stablehlo.rsqrt %206 : tensor<2048xf32>
      mhlo.return %207 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %1 = "mhlo.fusion"(%arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<512xf32>
      %207 = stablehlo.rsqrt %206 : tensor<512xf32>
      mhlo.return %207 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %2 = "mhlo.fusion"(%arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<1024xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1024xf32>
      mhlo.return %207 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %3 = "mhlo.fusion"(%arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %4 = "mhlo.fusion"(%arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %5 = "mhlo.fusion"(%arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<512xf32>
      %207 = stablehlo.rsqrt %206 : tensor<512xf32>
      mhlo.return %207 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %6 = "mhlo.fusion"(%arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %7 = "mhlo.fusion"(%arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<64xf32>
      %207 = stablehlo.rsqrt %206 : tensor<64xf32>
      mhlo.return %207 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %8 = "mhlo.fusion"(%arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<64xf32>
      %207 = stablehlo.rsqrt %206 : tensor<64xf32>
      mhlo.return %207 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %9 = "mhlo.fusion"(%arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<64xf32>
      %207 = stablehlo.rsqrt %206 : tensor<64xf32>
      mhlo.return %207 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %10 = stablehlo.custom_call @__cudnn$convForward(%arg267, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,112,112]{3,2,1,0}, u8[0]{0})"} : (tensor<128x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tuple<tensor<128x64x112x112xf32>, tensor<0xui8>>
    %11 = stablehlo.get_tuple_element %10[0] : (tuple<tensor<128x64x112x112xf32>, tensor<0xui8>>) -> tensor<128x64x112x112xf32>
    %12 = "mhlo.fusion"(%arg2, %arg1, %9, %11, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x64x112x112xf32>, %arg272: tensor<64xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x64x112x112xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x64x112x112xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x64x112x112xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<128x64x112x112xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x64x112x112xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x112x112xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x64x112x112xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %215 = "stablehlo.reduce_window"(%214, %cst_0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
      ^bb0(%arg273: tensor<f32>, %arg274: tensor<f32>):
        %216 = stablehlo.maximum %arg273, %arg274 : tensor<f32>
        stablehlo.return %216 : tensor<f32>
      }) : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x56x56xf32>
      mhlo.return %215 : tensor<128x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x112x112xf32>, tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %13 = stablehlo.custom_call @__cudnn$convForward(%12, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>
    %14 = stablehlo.get_tuple_element %13[0] : (tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>) -> tensor<128x64x56x56xf32>
    %15 = "mhlo.fusion"(%8, %arg8, %arg7, %14, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x64x56x56xf32>, %arg272: tensor<64xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x64x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x64x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x64x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x64x56x56xf32>
      mhlo.return %214 : tensor<128x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x56x56xf32>, tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %16 = stablehlo.custom_call @__cudnn$convForward(%15, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>
    %17 = stablehlo.get_tuple_element %16[0] : (tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>) -> tensor<128x64x56x56xf32>
    %18 = "mhlo.fusion"(%7, %arg11, %arg10, %17, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x64x56x56xf32>, %arg272: tensor<64xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x64x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x64x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x64x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x64x56x56xf32>
      mhlo.return %214 : tensor<128x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x56x56xf32>, tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %19 = stablehlo.custom_call @__cudnn$convForward(%18, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<128x256x56x56xf32>, tensor<0xui8>>
    %20 = stablehlo.get_tuple_element %19[0] : (tuple<tensor<128x256x56x56xf32>, tensor<0xui8>>) -> tensor<128x256x56x56xf32>
    %21 = "mhlo.fusion"(%arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %22 = stablehlo.custom_call @__cudnn$convForward(%12, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<128x256x56x56xf32>, tensor<0xui8>>
    %23 = stablehlo.get_tuple_element %22[0] : (tuple<tensor<128x256x56x56xf32>, tensor<0xui8>>) -> tensor<128x256x56x56xf32>
    %24 = "mhlo.fusion"(%arg14, %arg13, %6, %20, %arg169, %arg5, %arg4, %21, %23, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x56x56xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<256xf32>, %arg276: tensor<128x256x56x56xf32>, %arg277: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x56x56xf32>
      %213 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %214 = stablehlo.subtract %arg276, %213 : tensor<128x256x56x56xf32>
      %215 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %216 = stablehlo.multiply %214, %215 : tensor<128x256x56x56xf32>
      %217 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %218 = stablehlo.multiply %216, %217 : tensor<128x256x56x56xf32>
      %219 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %220 = stablehlo.add %218, %219 : tensor<128x256x56x56xf32>
      %221 = stablehlo.add %212, %220 : tensor<128x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x56x56xf32>
      %223 = stablehlo.maximum %221, %222 : tensor<128x256x56x56xf32>
      mhlo.return %223 : tensor<128x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x56x56xf32>, tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %25 = "mhlo.fusion"(%arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %26 = "mhlo.fusion"(%arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<64xf32>
      %207 = stablehlo.rsqrt %206 : tensor<64xf32>
      mhlo.return %207 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %27 = "mhlo.fusion"(%arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<64xf32>
      %207 = stablehlo.rsqrt %206 : tensor<64xf32>
      mhlo.return %207 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %28 = stablehlo.custom_call @__cudnn$convForward(%24, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>
    %29 = stablehlo.get_tuple_element %28[0] : (tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>) -> tensor<128x64x56x56xf32>
    %30 = "mhlo.fusion"(%27, %arg17, %arg16, %29, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x64x56x56xf32>, %arg272: tensor<64xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x64x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x64x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x64x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x64x56x56xf32>
      mhlo.return %214 : tensor<128x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x56x56xf32>, tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %31 = stablehlo.custom_call @__cudnn$convForward(%30, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>
    %32 = stablehlo.get_tuple_element %31[0] : (tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>) -> tensor<128x64x56x56xf32>
    %33 = "mhlo.fusion"(%26, %arg20, %arg19, %32, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x64x56x56xf32>, %arg272: tensor<64xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x64x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x64x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x64x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x64x56x56xf32>
      mhlo.return %214 : tensor<128x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x56x56xf32>, tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %34 = stablehlo.custom_call @__cudnn$convForward(%33, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<128x256x56x56xf32>, tensor<0xui8>>
    %35 = stablehlo.get_tuple_element %34[0] : (tuple<tensor<128x256x56x56xf32>, tensor<0xui8>>) -> tensor<128x256x56x56xf32>
    %36 = "mhlo.fusion"(%24, %arg23, %arg22, %25, %35, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x256x56x56xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<128x256x56x56xf32>, %arg273: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x256x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x56x56xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x56x56xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x256x56x56xf32>
      mhlo.return %215 : tensor<128x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x56x56xf32>, tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %37 = "mhlo.fusion"(%arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %38 = "mhlo.fusion"(%arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<64xf32>
      %207 = stablehlo.rsqrt %206 : tensor<64xf32>
      mhlo.return %207 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %39 = "mhlo.fusion"(%arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<64xf32>
      %207 = stablehlo.rsqrt %206 : tensor<64xf32>
      mhlo.return %207 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %40 = stablehlo.custom_call @__cudnn$convForward(%36, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>
    %41 = stablehlo.get_tuple_element %40[0] : (tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>) -> tensor<128x64x56x56xf32>
    %42 = "mhlo.fusion"(%39, %arg26, %arg25, %41, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x64x56x56xf32>, %arg272: tensor<64xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x64x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x64x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x64x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x64x56x56xf32>
      mhlo.return %214 : tensor<128x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x56x56xf32>, tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %43 = stablehlo.custom_call @__cudnn$convForward(%42, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>
    %44 = stablehlo.get_tuple_element %43[0] : (tuple<tensor<128x64x56x56xf32>, tensor<0xui8>>) -> tensor<128x64x56x56xf32>
    %45 = "mhlo.fusion"(%38, %arg29, %arg28, %44, %arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x64x56x56xf32>, %arg272: tensor<64xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x64x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x64x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x64x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<128x64x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x64x56x56xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x64x56x56xf32>
      mhlo.return %214 : tensor<128x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x56x56xf32>, tensor<64xf32>) -> tensor<128x64x56x56xf32>
    %46 = stablehlo.custom_call @__cudnn$convForward(%45, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<128x256x56x56xf32>, tensor<0xui8>>
    %47 = stablehlo.get_tuple_element %46[0] : (tuple<tensor<128x256x56x56xf32>, tensor<0xui8>>) -> tensor<128x256x56x56xf32>
    %48 = "mhlo.fusion"(%36, %arg32, %arg31, %37, %47, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x256x56x56xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<128x256x56x56xf32>, %arg273: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x256x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x56x56xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x56x56xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x256x56x56xf32>
      mhlo.return %215 : tensor<128x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x56x56xf32>, tensor<256xf32>) -> tensor<128x256x56x56xf32>
    %49 = stablehlo.custom_call @__cudnn$convForward(%48, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>
    %50 = stablehlo.get_tuple_element %49[0] : (tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>) -> tensor<128x512x28x28xf32>
    %51 = "mhlo.fusion"(%arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<512xf32>
      %207 = stablehlo.rsqrt %206 : tensor<512xf32>
      mhlo.return %207 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %52 = "mhlo.fusion"(%arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<128xf32>
      mhlo.return %207 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %53 = "mhlo.fusion"(%arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<128xf32>
      mhlo.return %207 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %54 = stablehlo.custom_call @__cudnn$convForward(%48, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<128x128x56x56xf32>, tensor<0xui8>>
    %55 = stablehlo.get_tuple_element %54[0] : (tuple<tensor<128x128x56x56xf32>, tensor<0xui8>>) -> tensor<128x128x56x56xf32>
    %56 = "mhlo.fusion"(%arg38, %arg37, %53, %55, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x128x56x56xf32>, %arg272: tensor<128xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x56x56xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x128x56x56xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<128x128x56x56xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x128x56x56xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<128x128x56x56xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x128x56x56xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<128x128x56x56xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x128x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x56x56xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x128x56x56xf32>
      mhlo.return %214 : tensor<128x128x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x56x56xf32>, tensor<128xf32>) -> tensor<128x128x56x56xf32>
    %57 = stablehlo.custom_call @__cudnn$convForward(%56, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>
    %58 = stablehlo.get_tuple_element %57[0] : (tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>) -> tensor<128x128x28x28xf32>
    %59 = "mhlo.fusion"(%arg41, %arg40, %52, %58, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x128x28x28xf32>, %arg272: tensor<128xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x128x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x128x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x128x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x128x28x28xf32>
      mhlo.return %214 : tensor<128x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x28x28xf32>, tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %60 = stablehlo.custom_call @__cudnn$convForward(%59, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>
    %61 = stablehlo.get_tuple_element %60[0] : (tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>) -> tensor<128x512x28x28xf32>
    %62 = "mhlo.fusion"(%arg35, %arg34, %5, %50, %arg183, %arg44, %arg43, %51, %61, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x512x28x28xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>, %arg274: tensor<512xf32>, %arg275: tensor<512xf32>, %arg276: tensor<128x512x28x28xf32>, %arg277: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %206 = stablehlo.subtract %arg276, %205 : tensor<128x512x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x512x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x512x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x512x28x28xf32>
      %213 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %214 = stablehlo.subtract %arg271, %213 : tensor<128x512x28x28xf32>
      %215 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %216 = stablehlo.multiply %214, %215 : tensor<128x512x28x28xf32>
      %217 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %218 = stablehlo.multiply %216, %217 : tensor<128x512x28x28xf32>
      %219 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %220 = stablehlo.add %218, %219 : tensor<128x512x28x28xf32>
      %221 = stablehlo.add %212, %220 : tensor<128x512x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x28x28xf32>
      %223 = stablehlo.maximum %221, %222 : tensor<128x512x28x28xf32>
      mhlo.return %223 : tensor<128x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x28x28xf32>, tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %63 = "mhlo.fusion"(%arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<512xf32>
      %207 = stablehlo.rsqrt %206 : tensor<512xf32>
      mhlo.return %207 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %64 = "mhlo.fusion"(%arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<128xf32>
      mhlo.return %207 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %65 = "mhlo.fusion"(%arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<128xf32>
      mhlo.return %207 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %66 = stablehlo.custom_call @__cudnn$convForward(%62, %arg45) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>
    %67 = stablehlo.get_tuple_element %66[0] : (tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>) -> tensor<128x128x28x28xf32>
    %68 = "mhlo.fusion"(%arg47, %arg46, %65, %67, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x128x28x28xf32>, %arg272: tensor<128xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x128x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x128x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x128x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x128x28x28xf32>
      mhlo.return %214 : tensor<128x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x28x28xf32>, tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %69 = stablehlo.custom_call @__cudnn$convForward(%68, %arg48) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>
    %70 = stablehlo.get_tuple_element %69[0] : (tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>) -> tensor<128x128x28x28xf32>
    %71 = "mhlo.fusion"(%arg50, %arg49, %64, %70, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x128x28x28xf32>, %arg272: tensor<128xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x128x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x128x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x128x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x128x28x28xf32>
      mhlo.return %214 : tensor<128x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x28x28xf32>, tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %72 = stablehlo.custom_call @__cudnn$convForward(%71, %arg51) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>
    %73 = stablehlo.get_tuple_element %72[0] : (tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>) -> tensor<128x512x28x28xf32>
    %74 = "mhlo.fusion"(%62, %arg53, %arg52, %63, %73, %arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<128x512x28x28xf32>, %arg273: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x512x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x512x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x512x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x512x28x28xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x512x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x28x28xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x512x28x28xf32>
      mhlo.return %215 : tensor<128x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x28x28xf32>, tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %75 = "mhlo.fusion"(%arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<512xf32>
      %207 = stablehlo.rsqrt %206 : tensor<512xf32>
      mhlo.return %207 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %76 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<128xf32>
      mhlo.return %207 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %77 = "mhlo.fusion"(%arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<128xf32>
      mhlo.return %207 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %78 = stablehlo.custom_call @__cudnn$convForward(%74, %arg54) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>
    %79 = stablehlo.get_tuple_element %78[0] : (tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>) -> tensor<128x128x28x28xf32>
    %80 = "mhlo.fusion"(%arg56, %arg55, %77, %79, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x128x28x28xf32>, %arg272: tensor<128xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x128x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x128x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x128x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x128x28x28xf32>
      mhlo.return %214 : tensor<128x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x28x28xf32>, tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %81 = stablehlo.custom_call @__cudnn$convForward(%80, %arg57) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>
    %82 = stablehlo.get_tuple_element %81[0] : (tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>) -> tensor<128x128x28x28xf32>
    %83 = "mhlo.fusion"(%arg59, %arg58, %76, %82, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x128x28x28xf32>, %arg272: tensor<128xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x128x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x128x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x128x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x128x28x28xf32>
      mhlo.return %214 : tensor<128x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x28x28xf32>, tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %84 = stablehlo.custom_call @__cudnn$convForward(%83, %arg60) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>
    %85 = stablehlo.get_tuple_element %84[0] : (tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>) -> tensor<128x512x28x28xf32>
    %86 = "mhlo.fusion"(%74, %arg62, %arg61, %75, %85, %arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<128x512x28x28xf32>, %arg273: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x512x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x512x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x512x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x512x28x28xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x512x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x28x28xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x512x28x28xf32>
      mhlo.return %215 : tensor<128x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x28x28xf32>, tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %87 = "mhlo.fusion"(%arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<512xf32>
      %207 = stablehlo.rsqrt %206 : tensor<512xf32>
      mhlo.return %207 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %88 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<128xf32>
      mhlo.return %207 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %89 = "mhlo.fusion"(%arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<128xf32>
      mhlo.return %207 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %90 = stablehlo.custom_call @__cudnn$convForward(%86, %arg63) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>
    %91 = stablehlo.get_tuple_element %90[0] : (tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>) -> tensor<128x128x28x28xf32>
    %92 = "mhlo.fusion"(%arg65, %arg64, %89, %91, %arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x128x28x28xf32>, %arg272: tensor<128xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x128x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x128x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x128x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x128x28x28xf32>
      mhlo.return %214 : tensor<128x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x28x28xf32>, tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %93 = stablehlo.custom_call @__cudnn$convForward(%92, %arg66) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>
    %94 = stablehlo.get_tuple_element %93[0] : (tuple<tensor<128x128x28x28xf32>, tensor<0xui8>>) -> tensor<128x128x28x28xf32>
    %95 = "mhlo.fusion"(%arg68, %arg67, %88, %94, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x128x28x28xf32>, %arg272: tensor<128xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x128x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x128x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x128x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<128x128x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x28x28xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x128x28x28xf32>
      mhlo.return %214 : tensor<128x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x28x28xf32>, tensor<128xf32>) -> tensor<128x128x28x28xf32>
    %96 = stablehlo.custom_call @__cudnn$convForward(%95, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>
    %97 = stablehlo.get_tuple_element %96[0] : (tuple<tensor<128x512x28x28xf32>, tensor<0xui8>>) -> tensor<128x512x28x28xf32>
    %98 = "mhlo.fusion"(%86, %arg71, %arg70, %87, %97, %arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<128x512x28x28xf32>, %arg273: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x512x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x512x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x512x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x512x28x28xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x512x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x28x28xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x512x28x28xf32>
      mhlo.return %215 : tensor<128x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x28x28xf32>, tensor<512xf32>) -> tensor<128x512x28x28xf32>
    %99 = stablehlo.custom_call @__cudnn$convForward(%98, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<128x256x28x28xf32>, tensor<0xui8>>
    %100 = stablehlo.get_tuple_element %99[0] : (tuple<tensor<128x256x28x28xf32>, tensor<0xui8>>) -> tensor<128x256x28x28xf32>
    %101 = "mhlo.fusion"(%4, %arg77, %arg76, %100, %arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x28x28xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x28x28xf32>
      %207 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x28x28xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x28x28xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x28x28xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x28x28xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x28x28xf32>
      mhlo.return %214 : tensor<128x256x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x28x28xf32>, tensor<256xf32>) -> tensor<128x256x28x28xf32>
    %102 = stablehlo.custom_call @__cudnn$convForward(%101, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %103 = stablehlo.get_tuple_element %102[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %104 = "mhlo.fusion"(%arg80, %arg79, %3, %103, %arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %105 = stablehlo.custom_call @__cudnn$convForward(%104, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>
    %106 = stablehlo.get_tuple_element %105[0] : (tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>) -> tensor<128x1024x14x14xf32>
    %107 = "mhlo.fusion"(%arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<1024xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1024xf32>
      mhlo.return %207 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %108 = stablehlo.custom_call @__cudnn$convForward(%98, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>
    %109 = stablehlo.get_tuple_element %108[0] : (tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>) -> tensor<128x1024x14x14xf32>
    %110 = "mhlo.fusion"(%arg83, %arg82, %2, %106, %arg215, %arg74, %arg73, %107, %109, %arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<128x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<128x1024x14x14xf32>, %arg277: tensor<1024xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x1024x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x1024x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x1024x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x1024x14x14xf32>
      %213 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %214 = stablehlo.subtract %arg276, %213 : tensor<128x1024x14x14xf32>
      %215 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %216 = stablehlo.multiply %214, %215 : tensor<128x1024x14x14xf32>
      %217 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %218 = stablehlo.multiply %216, %217 : tensor<128x1024x14x14xf32>
      %219 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %220 = stablehlo.add %218, %219 : tensor<128x1024x14x14xf32>
      %221 = stablehlo.add %212, %220 : tensor<128x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
      %223 = stablehlo.maximum %221, %222 : tensor<128x1024x14x14xf32>
      mhlo.return %223 : tensor<128x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x1024x14x14xf32>, tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %111 = "mhlo.fusion"(%arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<1024xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1024xf32>
      mhlo.return %207 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %112 = "mhlo.fusion"(%arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %113 = "mhlo.fusion"(%arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %114 = stablehlo.custom_call @__cudnn$convForward(%110, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %115 = stablehlo.get_tuple_element %114[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %116 = "mhlo.fusion"(%arg86, %arg85, %113, %115, %arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %117 = stablehlo.custom_call @__cudnn$convForward(%116, %arg87) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %118 = stablehlo.get_tuple_element %117[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %119 = "mhlo.fusion"(%arg89, %arg88, %112, %118, %arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %120 = stablehlo.custom_call @__cudnn$convForward(%119, %arg90) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>
    %121 = stablehlo.get_tuple_element %120[0] : (tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>) -> tensor<128x1024x14x14xf32>
    %122 = "mhlo.fusion"(%110, %arg92, %arg91, %111, %121, %arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<128x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x1024x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x1024x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x1024x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x1024x14x14xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x1024x14x14xf32>
      mhlo.return %215 : tensor<128x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x1024x14x14xf32>, tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %123 = "mhlo.fusion"(%arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<1024xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1024xf32>
      mhlo.return %207 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %124 = "mhlo.fusion"(%arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %125 = "mhlo.fusion"(%arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %126 = stablehlo.custom_call @__cudnn$convForward(%122, %arg93) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %127 = stablehlo.get_tuple_element %126[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %128 = "mhlo.fusion"(%arg95, %arg94, %125, %127, %arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %129 = stablehlo.custom_call @__cudnn$convForward(%128, %arg96) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %130 = stablehlo.get_tuple_element %129[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %131 = "mhlo.fusion"(%arg98, %arg97, %124, %130, %arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %132 = stablehlo.custom_call @__cudnn$convForward(%131, %arg99) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>
    %133 = stablehlo.get_tuple_element %132[0] : (tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>) -> tensor<128x1024x14x14xf32>
    %134 = "mhlo.fusion"(%122, %arg101, %arg100, %123, %133, %arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<128x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x1024x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x1024x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x1024x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x1024x14x14xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x1024x14x14xf32>
      mhlo.return %215 : tensor<128x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x1024x14x14xf32>, tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %135 = "mhlo.fusion"(%arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<1024xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1024xf32>
      mhlo.return %207 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %136 = "mhlo.fusion"(%arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %137 = "mhlo.fusion"(%arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %138 = stablehlo.custom_call @__cudnn$convForward(%134, %arg102) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %139 = stablehlo.get_tuple_element %138[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %140 = "mhlo.fusion"(%arg104, %arg103, %137, %139, %arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %141 = stablehlo.custom_call @__cudnn$convForward(%140, %arg105) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %142 = stablehlo.get_tuple_element %141[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %143 = "mhlo.fusion"(%arg107, %arg106, %136, %142, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %144 = stablehlo.custom_call @__cudnn$convForward(%143, %arg108) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>
    %145 = stablehlo.get_tuple_element %144[0] : (tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>) -> tensor<128x1024x14x14xf32>
    %146 = "mhlo.fusion"(%134, %arg110, %arg109, %135, %145, %arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<128x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x1024x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x1024x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x1024x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x1024x14x14xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x1024x14x14xf32>
      mhlo.return %215 : tensor<128x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x1024x14x14xf32>, tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %147 = "mhlo.fusion"(%arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<1024xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1024xf32>
      mhlo.return %207 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %148 = "mhlo.fusion"(%arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %149 = "mhlo.fusion"(%arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %150 = stablehlo.custom_call @__cudnn$convForward(%146, %arg111) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %151 = stablehlo.get_tuple_element %150[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %152 = "mhlo.fusion"(%arg113, %arg112, %149, %151, %arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %153 = stablehlo.custom_call @__cudnn$convForward(%152, %arg114) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %154 = stablehlo.get_tuple_element %153[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %155 = "mhlo.fusion"(%arg116, %arg115, %148, %154, %arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %156 = stablehlo.custom_call @__cudnn$convForward(%155, %arg117) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>
    %157 = stablehlo.get_tuple_element %156[0] : (tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>) -> tensor<128x1024x14x14xf32>
    %158 = "mhlo.fusion"(%146, %arg119, %arg118, %147, %157, %arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<128x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x1024x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x1024x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x1024x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x1024x14x14xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x1024x14x14xf32>
      mhlo.return %215 : tensor<128x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x1024x14x14xf32>, tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %159 = "mhlo.fusion"(%arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<1024xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1024xf32>
      mhlo.return %207 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %160 = "mhlo.fusion"(%arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %161 = "mhlo.fusion"(%arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<256xf32>
      mhlo.return %207 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %162 = stablehlo.custom_call @__cudnn$convForward(%158, %arg120) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %163 = stablehlo.get_tuple_element %162[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %164 = "mhlo.fusion"(%arg122, %arg121, %161, %163, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %165 = stablehlo.custom_call @__cudnn$convForward(%164, %arg123) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>
    %166 = stablehlo.get_tuple_element %165[0] : (tuple<tensor<128x256x14x14xf32>, tensor<0xui8>>) -> tensor<128x256x14x14xf32>
    %167 = "mhlo.fusion"(%arg125, %arg124, %160, %166, %arg243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x256x14x14xf32>, %arg272: tensor<256xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x256x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x256x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x256x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<128x256x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x256x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x256x14x14xf32>
      mhlo.return %214 : tensor<128x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x14x14xf32>, tensor<256xf32>) -> tensor<128x256x14x14xf32>
    %168 = stablehlo.custom_call @__cudnn$convForward(%167, %arg126) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>
    %169 = stablehlo.get_tuple_element %168[0] : (tuple<tensor<128x1024x14x14xf32>, tensor<0xui8>>) -> tensor<128x1024x14x14xf32>
    %170 = "mhlo.fusion"(%158, %arg128, %arg127, %159, %169, %arg245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<128x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x1024x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x1024x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x1024x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x1024x14x14xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x14x14xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x1024x14x14xf32>
      mhlo.return %215 : tensor<128x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x1024x14x14xf32>, tensor<1024xf32>) -> tensor<128x1024x14x14xf32>
    %171 = stablehlo.custom_call @__cudnn$convForward(%170, %arg132) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tuple<tensor<128x512x14x14xf32>, tensor<0xui8>>
    %172 = stablehlo.get_tuple_element %171[0] : (tuple<tensor<128x512x14x14xf32>, tensor<0xui8>>) -> tensor<128x512x14x14xf32>
    %173 = "mhlo.fusion"(%arg134, %arg133, %1, %172, %arg249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x512x14x14xf32>, %arg272: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<128x512x14x14xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x512x14x14xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<128x512x14x14xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x512x14x14xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x14x14xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x512x14x14xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<128x512x14x14xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x512x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %213 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512x14x14xf32>
      %214 = stablehlo.maximum %212, %213 : tensor<128x512x14x14xf32>
      mhlo.return %214 : tensor<128x512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x14x14xf32>, tensor<512xf32>) -> tensor<128x512x14x14xf32>
    %174 = stablehlo.custom_call @__cudnn$convForward(%173, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>
    %175 = stablehlo.get_tuple_element %174[0] : (tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>) -> tensor<128x512x7x7xf32>
    %176 = "mhlo.fusion"(%arg137, %arg136, %175, %arg251, %arg252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<128x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %206 = stablehlo.subtract %arg270, %205 : tensor<128x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %207 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %208 = stablehlo.add %arg272, %207 : tensor<512xf32>
      %209 = stablehlo.rsqrt %208 : tensor<512xf32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %211 = stablehlo.multiply %206, %210 : tensor<128x512x7x7xf32>
      %212 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %213 = stablehlo.multiply %211, %212 : tensor<128x512x7x7xf32>
      %214 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %215 = stablehlo.add %213, %214 : tensor<128x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %216 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
      %217 = stablehlo.maximum %215, %216 : tensor<128x512x7x7xf32>
      mhlo.return %217 : tensor<128x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<128x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %177 = stablehlo.custom_call @__cudnn$convForward(%176, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<128x2048x7x7xf32>, tensor<0xui8>>
    %178 = stablehlo.get_tuple_element %177[0] : (tuple<tensor<128x2048x7x7xf32>, tensor<0xui8>>) -> tensor<128x2048x7x7xf32>
    %179 = "mhlo.fusion"(%arg248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<2048xf32>
      %207 = stablehlo.rsqrt %206 : tensor<2048xf32>
      mhlo.return %207 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %180 = stablehlo.custom_call @__cudnn$convForward(%170, %arg129) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tuple<tensor<128x2048x7x7xf32>, tensor<0xui8>>
    %181 = stablehlo.get_tuple_element %180[0] : (tuple<tensor<128x2048x7x7xf32>, tensor<0xui8>>) -> tensor<128x2048x7x7xf32>
    %182 = "mhlo.fusion"(%arg140, %arg139, %0, %178, %arg253, %arg131, %arg130, %179, %181, %arg247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<128x2048x7x7xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>, %arg274: tensor<2048xf32>, %arg275: tensor<2048xf32>, %arg276: tensor<128x2048x7x7xf32>, %arg277: tensor<2048xf32>):
      %205 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %206 = stablehlo.subtract %arg271, %205 : tensor<128x2048x7x7xf32>
      %207 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x2048x7x7xf32>
      %209 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x2048x7x7xf32>
      %211 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x2048x7x7xf32>
      %213 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %214 = stablehlo.subtract %arg276, %213 : tensor<128x2048x7x7xf32>
      %215 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %216 = stablehlo.multiply %214, %215 : tensor<128x2048x7x7xf32>
      %217 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %218 = stablehlo.multiply %216, %217 : tensor<128x2048x7x7xf32>
      %219 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %220 = stablehlo.add %218, %219 : tensor<128x2048x7x7xf32>
      %221 = stablehlo.add %212, %220 : tensor<128x2048x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048x7x7xf32>
      %223 = stablehlo.maximum %221, %222 : tensor<128x2048x7x7xf32>
      mhlo.return %223 : tensor<128x2048x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x2048x7x7xf32>, tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %183 = "mhlo.fusion"(%arg260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<2048xf32>
      %207 = stablehlo.rsqrt %206 : tensor<2048xf32>
      mhlo.return %207 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %184 = stablehlo.custom_call @__cudnn$convForward(%182, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>
    %185 = stablehlo.get_tuple_element %184[0] : (tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>) -> tensor<128x512x7x7xf32>
    %186 = "mhlo.fusion"(%arg143, %arg142, %185, %arg255, %arg256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<128x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %206 = stablehlo.subtract %arg270, %205 : tensor<128x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %207 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %208 = stablehlo.add %arg272, %207 : tensor<512xf32>
      %209 = stablehlo.rsqrt %208 : tensor<512xf32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %211 = stablehlo.multiply %206, %210 : tensor<128x512x7x7xf32>
      %212 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %213 = stablehlo.multiply %211, %212 : tensor<128x512x7x7xf32>
      %214 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %215 = stablehlo.add %213, %214 : tensor<128x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %216 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
      %217 = stablehlo.maximum %215, %216 : tensor<128x512x7x7xf32>
      mhlo.return %217 : tensor<128x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<128x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %187 = stablehlo.custom_call @__cudnn$convForward(%186, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>
    %188 = stablehlo.get_tuple_element %187[0] : (tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>) -> tensor<128x512x7x7xf32>
    %189 = "mhlo.fusion"(%arg146, %arg145, %188, %arg257, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<128x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %206 = stablehlo.subtract %arg270, %205 : tensor<128x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %207 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %208 = stablehlo.add %arg272, %207 : tensor<512xf32>
      %209 = stablehlo.rsqrt %208 : tensor<512xf32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %211 = stablehlo.multiply %206, %210 : tensor<128x512x7x7xf32>
      %212 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %213 = stablehlo.multiply %211, %212 : tensor<128x512x7x7xf32>
      %214 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %215 = stablehlo.add %213, %214 : tensor<128x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %216 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
      %217 = stablehlo.maximum %215, %216 : tensor<128x512x7x7xf32>
      mhlo.return %217 : tensor<128x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<128x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %190 = stablehlo.custom_call @__cudnn$convForward(%189, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<128x2048x7x7xf32>, tensor<0xui8>>
    %191 = stablehlo.get_tuple_element %190[0] : (tuple<tensor<128x2048x7x7xf32>, tensor<0xui8>>) -> tensor<128x2048x7x7xf32>
    %192 = "mhlo.fusion"(%182, %arg149, %arg148, %183, %191, %arg259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x2048x7x7xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<2048xf32>, %arg272: tensor<128x2048x7x7xf32>, %arg273: tensor<2048xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x2048x7x7xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x2048x7x7xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x2048x7x7xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x2048x7x7xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x2048x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048x7x7xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x2048x7x7xf32>
      mhlo.return %215 : tensor<128x2048x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<128x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x2048x7x7xf32>, tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
    %193 = "mhlo.fusion"(%arg266) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %206 = stablehlo.add %arg268, %205 : tensor<2048xf32>
      %207 = stablehlo.rsqrt %206 : tensor<2048xf32>
      mhlo.return %207 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %194 = stablehlo.custom_call @__cudnn$convForward(%192, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>
    %195 = stablehlo.get_tuple_element %194[0] : (tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>) -> tensor<128x512x7x7xf32>
    %196 = "mhlo.fusion"(%arg152, %arg151, %195, %arg261, %arg262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<128x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %206 = stablehlo.subtract %arg270, %205 : tensor<128x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %207 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %208 = stablehlo.add %arg272, %207 : tensor<512xf32>
      %209 = stablehlo.rsqrt %208 : tensor<512xf32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %211 = stablehlo.multiply %206, %210 : tensor<128x512x7x7xf32>
      %212 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %213 = stablehlo.multiply %211, %212 : tensor<128x512x7x7xf32>
      %214 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %215 = stablehlo.add %213, %214 : tensor<128x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %216 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
      %217 = stablehlo.maximum %215, %216 : tensor<128x512x7x7xf32>
      mhlo.return %217 : tensor<128x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<128x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %197 = stablehlo.custom_call @__cudnn$convForward(%196, %arg153) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>
    %198 = stablehlo.get_tuple_element %197[0] : (tuple<tensor<128x512x7x7xf32>, tensor<0xui8>>) -> tensor<128x512x7x7xf32>
    %199 = "mhlo.fusion"(%arg155, %arg154, %198, %arg263, %arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<128x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %205 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %206 = stablehlo.subtract %arg270, %205 : tensor<128x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %207 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %208 = stablehlo.add %arg272, %207 : tensor<512xf32>
      %209 = stablehlo.rsqrt %208 : tensor<512xf32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %211 = stablehlo.multiply %206, %210 : tensor<128x512x7x7xf32>
      %212 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %213 = stablehlo.multiply %211, %212 : tensor<128x512x7x7xf32>
      %214 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<128x512x7x7xf32>
      %215 = stablehlo.add %213, %214 : tensor<128x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %216 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512x7x7xf32>
      %217 = stablehlo.maximum %215, %216 : tensor<128x512x7x7xf32>
      mhlo.return %217 : tensor<128x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<128x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<128x512x7x7xf32>
    %200 = stablehlo.custom_call @__cudnn$convForward(%199, %arg156) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<128x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<128x2048x7x7xf32>, tensor<0xui8>>
    %201 = stablehlo.get_tuple_element %200[0] : (tuple<tensor<128x2048x7x7xf32>, tensor<0xui8>>) -> tensor<128x2048x7x7xf32>
    %202 = "mhlo.fusion"(%192, %arg158, %arg157, %193, %201, %arg265) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg268: tensor<128x2048x7x7xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<2048xf32>, %arg272: tensor<128x2048x7x7xf32>, %arg273: tensor<2048xf32>):
      %205 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %206 = stablehlo.subtract %arg272, %205 : tensor<128x2048x7x7xf32>
      %207 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %208 = stablehlo.multiply %206, %207 : tensor<128x2048x7x7xf32>
      %209 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %210 = stablehlo.multiply %208, %209 : tensor<128x2048x7x7xf32>
      %211 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<128x2048x7x7xf32>
      %212 = stablehlo.add %210, %211 : tensor<128x2048x7x7xf32>
      %213 = stablehlo.add %212, %arg268 : tensor<128x2048x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %214 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048x7x7xf32>
      %215 = stablehlo.maximum %213, %214 : tensor<128x2048x7x7xf32>
      %216 = mhlo.bitcast %215 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2048x7x7xf32>) -> tensor<128x2048x49xf32>
      %217 = stablehlo.reduce(%216 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x2048x49xf32>, tensor<f32>) -> tensor<128x2048xf32>
      mhlo.return %217 : tensor<128x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<128x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x2048x7x7xf32>, tensor<2048xf32>) -> tensor<128x2048xf32>
    %203 = "mhlo.fusion"(%202, %arg159) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg268: tensor<128x2048xf32>, %arg269: tensor<1000x2048xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048xf32>
      %206 = stablehlo.multiply %arg268, %205 : tensor<128x2048xf32>
      %207 = stablehlo.dot_general %206, %arg269, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x2048xf32>, tensor<1000x2048xf32>) -> tensor<128x1000xf32>
      mhlo.return %207 : tensor<128x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<128x2048xf32>, tensor<1000x2048xf32>) -> tensor<128x1000xf32>
    %204 = "mhlo.fusion"(%203, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x1000xf32>, %arg269: tensor<1000xf32>):
      %205 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1000xf32>) -> tensor<128x1000xf32>
      %206 = stablehlo.add %205, %arg268 : tensor<128x1000xf32>
      mhlo.return %206 : tensor<128x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1000xf32>, tensor<1000xf32>) -> tensor<128x1000xf32>
    return %204 : tensor<128x1000xf32>
  }
}
