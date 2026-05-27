module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.1_computation(%arg0: tensor<64x2048xf32>, %arg1: tensor<1000x2048xf32>) -> tensor<64x1000xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x2048xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x2048xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<1000x2048xf32>) -> tensor<64x1000xf32>
    return %2 : tensor<64x1000xf32>
  }
  func.func private @region_1.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<64x2048x7x7xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<64x2048x7x7xf32>, %arg5: tensor<2048xf32>) -> tensor<64x2048xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x2048x7x7xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x2048x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x2048x7x7xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x2048x7x7xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x2048x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x2048x7x7xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x2048x7x7xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x49xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x2048x49xf32>, tensor<f32>) -> tensor<64x2048xf32>
    return %12 : tensor<64x2048xf32>
  }
  func.func private @fused_maximum(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x56x56xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<64x256x56x56xf32>, %arg9: tensor<256xf32>) -> tensor<64x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<64x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x56x56xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<64x256x56x56xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x256x56x56xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<64x256x56x56xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %15 = stablehlo.add %13, %14 : tensor<64x256x56x56xf32>
    %16 = stablehlo.add %7, %15 : tensor<64x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x56x56xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<64x256x56x56xf32>
    return %18 : tensor<64x256x56x56xf32>
  }
  func.func private @fused_maximum.1(%arg0: tensor<64x256x56x56xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<64x256x56x56xf32>, %arg5: tensor<256xf32>) -> tensor<64x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x56x56xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x56x56xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x256x56x56xf32>
    return %10 : tensor<64x256x56x56xf32>
  }
  func.func private @fused_maximum.2(%arg0: tensor<64x256x56x56xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<64x256x56x56xf32>, %arg5: tensor<256xf32>) -> tensor<64x256x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x256x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x56x56xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x256x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x56x56xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x256x56x56xf32>
    return %10 : tensor<64x256x56x56xf32>
  }
  func.func private @fused_maximum.3(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<64x512x28x28xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512xf32>, %arg8: tensor<64x512x28x28xf32>, %arg9: tensor<512xf32>) -> tensor<64x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x512x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x512x28x28xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<64x512x28x28xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x512x28x28xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<64x512x28x28xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %15 = stablehlo.add %13, %14 : tensor<64x512x28x28xf32>
    %16 = stablehlo.add %7, %15 : tensor<64x512x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x28x28xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<64x512x28x28xf32>
    return %18 : tensor<64x512x28x28xf32>
  }
  func.func private @fused_maximum.4(%arg0: tensor<64x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<64x512x28x28xf32>, %arg5: tensor<512xf32>) -> tensor<64x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x512x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x512x28x28xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x512x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x28x28xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x512x28x28xf32>
    return %10 : tensor<64x512x28x28xf32>
  }
  func.func private @fused_maximum.5(%arg0: tensor<64x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<64x512x28x28xf32>, %arg5: tensor<512xf32>) -> tensor<64x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x512x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x512x28x28xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x512x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x28x28xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x512x28x28xf32>
    return %10 : tensor<64x512x28x28xf32>
  }
  func.func private @fused_maximum.6(%arg0: tensor<64x512x28x28xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<64x512x28x28xf32>, %arg5: tensor<512xf32>) -> tensor<64x512x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x512x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x512x28x28xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x512x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x28x28xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x512x28x28xf32>
    return %10 : tensor<64x512x28x28xf32>
  }
  func.func private @fused_maximum.7(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<64x1024x14x14xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<64x1024x14x14xf32>, %arg9: tensor<1024xf32>) -> tensor<64x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x1024x14x14xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<64x1024x14x14xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x1024x14x14xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<64x1024x14x14xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %15 = stablehlo.add %13, %14 : tensor<64x1024x14x14xf32>
    %16 = stablehlo.add %7, %15 : tensor<64x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<64x1024x14x14xf32>
    return %18 : tensor<64x1024x14x14xf32>
  }
  func.func private @fused_maximum.8(%arg0: tensor<64x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<64x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<64x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x1024x14x14xf32>
    return %10 : tensor<64x1024x14x14xf32>
  }
  func.func private @fused_maximum.9(%arg0: tensor<64x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<64x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<64x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x1024x14x14xf32>
    return %10 : tensor<64x1024x14x14xf32>
  }
  func.func private @fused_maximum.10(%arg0: tensor<64x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<64x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<64x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x1024x14x14xf32>
    return %10 : tensor<64x1024x14x14xf32>
  }
  func.func private @fused_maximum.11(%arg0: tensor<64x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<64x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<64x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x1024x14x14xf32>
    return %10 : tensor<64x1024x14x14xf32>
  }
  func.func private @fused_maximum.12(%arg0: tensor<64x1024x14x14xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<64x1024x14x14xf32>, %arg5: tensor<1024xf32>) -> tensor<64x1024x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x1024x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x1024x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x1024x14x14xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x1024x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x1024x14x14xf32>
    return %10 : tensor<64x1024x14x14xf32>
  }
  func.func private @fused_maximum.13(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x64x56x56xf32>
    return %9 : tensor<64x64x56x56xf32>
  }
  func.func private @fused_maximum.14(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x64x56x56xf32>
    return %9 : tensor<64x64x56x56xf32>
  }
  func.func private @fused_maximum.15(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x64x56x56xf32>
    return %9 : tensor<64x64x56x56xf32>
  }
  func.func private @fused_maximum.16(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x64x56x56xf32>
    return %9 : tensor<64x64x56x56xf32>
  }
  func.func private @fused_maximum.17(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x64x56x56xf32>
    return %9 : tensor<64x64x56x56xf32>
  }
  func.func private @fused_maximum.18(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x56x56xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x64x56x56xf32>
    return %9 : tensor<64x64x56x56xf32>
  }
  func.func private @region_0.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_window(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x112x112xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x112x112xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x112x112xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x112x112xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x112x112xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x112x112xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x64x112x112xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %10 = "stablehlo.reduce_window"(%9, %cst_0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %11 = stablehlo.maximum %arg5, %arg6 : tensor<f32>
      stablehlo.return %11 : tensor<f32>
    }) : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x56x56xf32>
    return %10 : tensor<64x64x56x56xf32>
  }
  func.func private @fused_maximum.19(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x56x56xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x56x56xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x56x56xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x56x56xf32>
    return %9 : tensor<64x128x56x56xf32>
  }
  func.func private @fused_maximum.20(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x28x28xf32>
    return %9 : tensor<64x128x28x28xf32>
  }
  func.func private @fused_maximum.21(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x28x28xf32>
    return %9 : tensor<64x128x28x28xf32>
  }
  func.func private @fused_maximum.22(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x28x28xf32>
    return %9 : tensor<64x128x28x28xf32>
  }
  func.func private @fused_maximum.23(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x28x28xf32>
    return %9 : tensor<64x128x28x28xf32>
  }
  func.func private @fused_maximum.24(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x28x28xf32>
    return %9 : tensor<64x128x28x28xf32>
  }
  func.func private @fused_maximum.25(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x28x28xf32>
    return %9 : tensor<64x128x28x28xf32>
  }
  func.func private @fused_maximum.26(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x28x28xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x28x28xf32>
    return %9 : tensor<64x128x28x28xf32>
  }
  func.func private @fused_maximum.27(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.28(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.29(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.30(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.31(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.32(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.33(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.34(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.35(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.36(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.37(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x14x14xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x256x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x256x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x256x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x256x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x256x14x14xf32>
    return %12 : tensor<64x256x14x14xf32>
  }
  func.func private @fused_maximum.38(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x28x28xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x28x28xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x28x28xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x256x28x28xf32>
    return %9 : tensor<64x256x28x28xf32>
  }
  func.func private @fused_maximum.39(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<64x2048x7x7xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<2048xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<64x2048x7x7xf32>, %arg9: tensor<2048xf32>) -> tensor<64x2048x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x2048x7x7xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x2048x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x2048x7x7xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x2048x7x7xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<64x2048x7x7xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x2048x7x7xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<64x2048x7x7xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %15 = stablehlo.add %13, %14 : tensor<64x2048x7x7xf32>
    %16 = stablehlo.add %7, %15 : tensor<64x2048x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x2048x7x7xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<64x2048x7x7xf32>
    return %18 : tensor<64x2048x7x7xf32>
  }
  func.func private @fused_maximum.40(%arg0: tensor<64x2048x7x7xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<64x2048x7x7xf32>, %arg5: tensor<2048xf32>) -> tensor<64x2048x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x2048x7x7xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x2048x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x2048x7x7xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x2048x7x7xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<64x2048x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x2048x7x7xf32>
    %10 = stablehlo.maximum %8, %9 : tensor<64x2048x7x7xf32>
    return %10 : tensor<64x2048x7x7xf32>
  }
  func.func private @fused_maximum.41(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<64x512x14x14xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x512x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x512x14x14xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x512x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x14x14xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x512x14x14xf32>
    return %9 : tensor<64x512x14x14xf32>
  }
  func.func private @fused_maximum.42(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<64x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x512x7x7xf32>
    return %12 : tensor<64x512x7x7xf32>
  }
  func.func private @fused_maximum.43(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<64x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x512x7x7xf32>
    return %12 : tensor<64x512x7x7xf32>
  }
  func.func private @fused_maximum.44(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<64x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x512x7x7xf32>
    return %12 : tensor<64x512x7x7xf32>
  }
  func.func private @fused_maximum.45(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<64x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x512x7x7xf32>
    return %12 : tensor<64x512x7x7xf32>
  }
  func.func private @fused_maximum.46(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<64x512x7x7xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x512x7x7xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x512x7x7xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x512x7x7xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x512x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<64x512x7x7xf32>
    return %12 : tensor<64x512x7x7xf32>
  }
  func.func private @fused_add(%arg0: tensor<64x1000xf32>, %arg1: tensor<1000xf32>) -> tensor<64x1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<64x1000xf32>
    return %1 : tensor<64x1000xf32>
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
  func.func private @fused_rsqrt.22(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.23(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.24(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.25(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.26(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.27(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.28(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.29(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.30(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.31(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.32(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.33(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.34(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.35(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.36(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64x1x1xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<64x64x1x1xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<256x64x1x1xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<64x256x1x1xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64x64x3x3xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<256x64x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<64x256x1x1xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<64x64x3x3xf32>, %arg28: tensor<64xf32>, %arg29: tensor<64xf32>, %arg30: tensor<256x64x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<512x256x1x1xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<128x256x1x1xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128x128x3x3xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<512x128x1x1xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<128x512x1x1xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128x128x3x3xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<512x128x1x1xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512xf32>, %arg54: tensor<128x512x1x1xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128x128x3x3xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<512x128x1x1xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512xf32>, %arg63: tensor<128x512x1x1xf32>, %arg64: tensor<128xf32>, %arg65: tensor<128xf32>, %arg66: tensor<128x128x3x3xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<512x128x1x1xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<1024x512x1x1xf32>, %arg73: tensor<1024xf32>, %arg74: tensor<1024xf32>, %arg75: tensor<256x512x1x1xf32>, %arg76: tensor<256xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x256x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<256xf32>, %arg81: tensor<1024x256x1x1xf32>, %arg82: tensor<1024xf32>, %arg83: tensor<1024xf32>, %arg84: tensor<256x1024x1x1xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256x256x3x3xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<1024x256x1x1xf32>, %arg91: tensor<1024xf32>, %arg92: tensor<1024xf32>, %arg93: tensor<256x1024x1x1xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256x256x3x3xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<1024x256x1x1xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<1024xf32>, %arg102: tensor<256x1024x1x1xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256x256x3x3xf32>, %arg106: tensor<256xf32>, %arg107: tensor<256xf32>, %arg108: tensor<1024x256x1x1xf32>, %arg109: tensor<1024xf32>, %arg110: tensor<1024xf32>, %arg111: tensor<256x1024x1x1xf32>, %arg112: tensor<256xf32>, %arg113: tensor<256xf32>, %arg114: tensor<256x256x3x3xf32>, %arg115: tensor<256xf32>, %arg116: tensor<256xf32>, %arg117: tensor<1024x256x1x1xf32>, %arg118: tensor<1024xf32>, %arg119: tensor<1024xf32>, %arg120: tensor<256x1024x1x1xf32>, %arg121: tensor<256xf32>, %arg122: tensor<256xf32>, %arg123: tensor<256x256x3x3xf32>, %arg124: tensor<256xf32>, %arg125: tensor<256xf32>, %arg126: tensor<1024x256x1x1xf32>, %arg127: tensor<1024xf32>, %arg128: tensor<1024xf32>, %arg129: tensor<2048x1024x1x1xf32>, %arg130: tensor<2048xf32>, %arg131: tensor<2048xf32>, %arg132: tensor<512x1024x1x1xf32>, %arg133: tensor<512xf32>, %arg134: tensor<512xf32>, %arg135: tensor<512x512x3x3xf32>, %arg136: tensor<512xf32>, %arg137: tensor<512xf32>, %arg138: tensor<2048x512x1x1xf32>, %arg139: tensor<2048xf32>, %arg140: tensor<2048xf32>, %arg141: tensor<512x2048x1x1xf32>, %arg142: tensor<512xf32>, %arg143: tensor<512xf32>, %arg144: tensor<512x512x3x3xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<2048x512x1x1xf32>, %arg148: tensor<2048xf32>, %arg149: tensor<2048xf32>, %arg150: tensor<512x2048x1x1xf32>, %arg151: tensor<512xf32>, %arg152: tensor<512xf32>, %arg153: tensor<512x512x3x3xf32>, %arg154: tensor<512xf32>, %arg155: tensor<512xf32>, %arg156: tensor<2048x512x1x1xf32>, %arg157: tensor<2048xf32>, %arg158: tensor<2048xf32>, %arg159: tensor<1000x2048xf32>, %arg160: tensor<1000xf32>, %arg161: tensor<64xf32>, %arg162: tensor<64xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<64xf32>, %arg166: tensor<64xf32>, %arg167: tensor<64xf32>, %arg168: tensor<64xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<64xf32>, %arg174: tensor<64xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<64xf32>, %arg178: tensor<64xf32>, %arg179: tensor<64xf32>, %arg180: tensor<64xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512xf32>, %arg184: tensor<512xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<128xf32>, %arg188: tensor<128xf32>, %arg189: tensor<512xf32>, %arg190: tensor<512xf32>, %arg191: tensor<128xf32>, %arg192: tensor<128xf32>, %arg193: tensor<128xf32>, %arg194: tensor<128xf32>, %arg195: tensor<512xf32>, %arg196: tensor<512xf32>, %arg197: tensor<128xf32>, %arg198: tensor<128xf32>, %arg199: tensor<128xf32>, %arg200: tensor<128xf32>, %arg201: tensor<512xf32>, %arg202: tensor<512xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<512xf32>, %arg208: tensor<512xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1024xf32>, %arg211: tensor<256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<1024xf32>, %arg216: tensor<1024xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<1024xf32>, %arg222: tensor<1024xf32>, %arg223: tensor<256xf32>, %arg224: tensor<256xf32>, %arg225: tensor<256xf32>, %arg226: tensor<256xf32>, %arg227: tensor<1024xf32>, %arg228: tensor<1024xf32>, %arg229: tensor<256xf32>, %arg230: tensor<256xf32>, %arg231: tensor<256xf32>, %arg232: tensor<256xf32>, %arg233: tensor<1024xf32>, %arg234: tensor<1024xf32>, %arg235: tensor<256xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<256xf32>, %arg239: tensor<1024xf32>, %arg240: tensor<1024xf32>, %arg241: tensor<256xf32>, %arg242: tensor<256xf32>, %arg243: tensor<256xf32>, %arg244: tensor<256xf32>, %arg245: tensor<1024xf32>, %arg246: tensor<1024xf32>, %arg247: tensor<2048xf32>, %arg248: tensor<2048xf32>, %arg249: tensor<512xf32>, %arg250: tensor<512xf32>, %arg251: tensor<512xf32>, %arg252: tensor<512xf32>, %arg253: tensor<2048xf32>, %arg254: tensor<2048xf32>, %arg255: tensor<512xf32>, %arg256: tensor<512xf32>, %arg257: tensor<512xf32>, %arg258: tensor<512xf32>, %arg259: tensor<2048xf32>, %arg260: tensor<2048xf32>, %arg261: tensor<512xf32>, %arg262: tensor<512xf32>, %arg263: tensor<512xf32>, %arg264: tensor<512xf32>, %arg265: tensor<2048xf32>, %arg266: tensor<2048xf32>, %arg267: tensor<64x3x224x224xf32>) -> tensor<64x1000xf32> {
    %0 = "mhlo.fusion"(%arg254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<2048xf32>
      %196 = stablehlo.rsqrt %195 : tensor<2048xf32>
      mhlo.return %196 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %1 = "mhlo.fusion"(%arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<512xf32>
      %196 = stablehlo.rsqrt %195 : tensor<512xf32>
      mhlo.return %196 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %2 = "mhlo.fusion"(%arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<1024xf32>
      %196 = stablehlo.rsqrt %195 : tensor<1024xf32>
      mhlo.return %196 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %3 = "mhlo.fusion"(%arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<256xf32>
      %196 = stablehlo.rsqrt %195 : tensor<256xf32>
      mhlo.return %196 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %4 = "mhlo.fusion"(%arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<512xf32>
      %196 = stablehlo.rsqrt %195 : tensor<512xf32>
      mhlo.return %196 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %5 = "mhlo.fusion"(%arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<128xf32>
      %196 = stablehlo.rsqrt %195 : tensor<128xf32>
      mhlo.return %196 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %6 = "mhlo.fusion"(%arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<128xf32>
      %196 = stablehlo.rsqrt %195 : tensor<128xf32>
      mhlo.return %196 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %7 = "mhlo.fusion"(%arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<256xf32>
      %196 = stablehlo.rsqrt %195 : tensor<256xf32>
      mhlo.return %196 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %8 = "mhlo.fusion"(%arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<64xf32>
      %196 = stablehlo.rsqrt %195 : tensor<64xf32>
      mhlo.return %196 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %9 = stablehlo.custom_call @__cudnn$convForward(%arg267, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,112,112]{3,2,1,0}, u8[0]{0})"} : (tensor<64x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tuple<tensor<64x64x112x112xf32>, tensor<0xui8>>
    %10 = stablehlo.get_tuple_element %9[0] : (tuple<tensor<64x64x112x112xf32>, tensor<0xui8>>) -> tensor<64x64x112x112xf32>
    %11 = "mhlo.fusion"(%arg2, %arg1, %8, %10, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<64x64x112x112xf32>, %arg272: tensor<64xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x64x112x112xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x64x112x112xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x64x112x112xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x64x112x112xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x112x112xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x64x112x112xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %204 = "stablehlo.reduce_window"(%203, %cst_0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
      ^bb0(%arg273: tensor<f32>, %arg274: tensor<f32>):
        %205 = stablehlo.maximum %arg273, %arg274 : tensor<f32>
        stablehlo.return %205 : tensor<f32>
      }) : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x56x56xf32>
      mhlo.return %204 : tensor<64x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x112x112xf32>, tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %12 = stablehlo.custom_call @__cudnn$convForward(%11, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<64x256x56x56xf32>, tensor<0xui8>>
    %13 = stablehlo.get_tuple_element %12[0] : (tuple<tensor<64x256x56x56xf32>, tensor<0xui8>>) -> tensor<64x256x56x56xf32>
    %14 = "mhlo.fusion"(%arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<256xf32>
      %196 = stablehlo.rsqrt %195 : tensor<256xf32>
      mhlo.return %196 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %15 = "mhlo.fusion"(%arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<64xf32>
      %196 = stablehlo.rsqrt %195 : tensor<64xf32>
      mhlo.return %196 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %16 = "mhlo.fusion"(%arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<64xf32>
      %196 = stablehlo.rsqrt %195 : tensor<64xf32>
      mhlo.return %196 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %17 = stablehlo.custom_call @__cudnn$convForward(%11, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>
    %18 = stablehlo.get_tuple_element %17[0] : (tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>) -> tensor<64x64x56x56xf32>
    %19 = "mhlo.fusion"(%arg8, %arg7, %16, %18, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<64x64x56x56xf32>, %arg272: tensor<64xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x64x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x64x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x64x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x64x56x56xf32>
      mhlo.return %203 : tensor<64x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x56x56xf32>, tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %20 = stablehlo.custom_call @__cudnn$convForward(%19, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>
    %21 = stablehlo.get_tuple_element %20[0] : (tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>) -> tensor<64x64x56x56xf32>
    %22 = "mhlo.fusion"(%arg11, %arg10, %15, %21, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<64x64x56x56xf32>, %arg272: tensor<64xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x64x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x64x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x64x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x64x56x56xf32>
      mhlo.return %203 : tensor<64x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x56x56xf32>, tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %23 = stablehlo.custom_call @__cudnn$convForward(%22, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<64x256x56x56xf32>, tensor<0xui8>>
    %24 = stablehlo.get_tuple_element %23[0] : (tuple<tensor<64x256x56x56xf32>, tensor<0xui8>>) -> tensor<64x256x56x56xf32>
    %25 = "mhlo.fusion"(%arg5, %arg4, %7, %13, %arg163, %arg14, %arg13, %14, %24, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<64x256x56x56xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<256xf32>, %arg276: tensor<64x256x56x56xf32>, %arg277: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %195 = stablehlo.subtract %arg276, %194 : tensor<64x256x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x256x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x256x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x256x56x56xf32>
      %202 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %203 = stablehlo.subtract %arg271, %202 : tensor<64x256x56x56xf32>
      %204 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %205 = stablehlo.multiply %203, %204 : tensor<64x256x56x56xf32>
      %206 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %207 = stablehlo.multiply %205, %206 : tensor<64x256x56x56xf32>
      %208 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %209 = stablehlo.add %207, %208 : tensor<64x256x56x56xf32>
      %210 = stablehlo.add %201, %209 : tensor<64x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %211 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x56x56xf32>
      %212 = stablehlo.maximum %210, %211 : tensor<64x256x56x56xf32>
      mhlo.return %212 : tensor<64x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x56x56xf32>, tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %26 = "mhlo.fusion"(%arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<256xf32>
      %196 = stablehlo.rsqrt %195 : tensor<256xf32>
      mhlo.return %196 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %27 = "mhlo.fusion"(%arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<64xf32>
      %196 = stablehlo.rsqrt %195 : tensor<64xf32>
      mhlo.return %196 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %28 = "mhlo.fusion"(%arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<64xf32>
      %196 = stablehlo.rsqrt %195 : tensor<64xf32>
      mhlo.return %196 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %29 = stablehlo.custom_call @__cudnn$convForward(%25, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>
    %30 = stablehlo.get_tuple_element %29[0] : (tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>) -> tensor<64x64x56x56xf32>
    %31 = "mhlo.fusion"(%arg17, %arg16, %28, %30, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<64x64x56x56xf32>, %arg272: tensor<64xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x64x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x64x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x64x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x64x56x56xf32>
      mhlo.return %203 : tensor<64x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x56x56xf32>, tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %32 = stablehlo.custom_call @__cudnn$convForward(%31, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>
    %33 = stablehlo.get_tuple_element %32[0] : (tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>) -> tensor<64x64x56x56xf32>
    %34 = "mhlo.fusion"(%arg20, %arg19, %27, %33, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<64x64x56x56xf32>, %arg272: tensor<64xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x64x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x64x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x64x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x64x56x56xf32>
      mhlo.return %203 : tensor<64x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x56x56xf32>, tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %35 = stablehlo.custom_call @__cudnn$convForward(%34, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<64x256x56x56xf32>, tensor<0xui8>>
    %36 = stablehlo.get_tuple_element %35[0] : (tuple<tensor<64x256x56x56xf32>, tensor<0xui8>>) -> tensor<64x256x56x56xf32>
    %37 = "mhlo.fusion"(%25, %arg23, %arg22, %26, %36, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x256x56x56xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<64x256x56x56xf32>, %arg273: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x256x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x256x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x256x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x256x56x56xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x56x56xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x256x56x56xf32>
      mhlo.return %204 : tensor<64x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x56x56xf32>, tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %38 = "mhlo.fusion"(%arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<256xf32>
      %196 = stablehlo.rsqrt %195 : tensor<256xf32>
      mhlo.return %196 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %39 = "mhlo.fusion"(%arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<64xf32>
      %196 = stablehlo.rsqrt %195 : tensor<64xf32>
      mhlo.return %196 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %40 = "mhlo.fusion"(%arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<64xf32>
      %196 = stablehlo.rsqrt %195 : tensor<64xf32>
      mhlo.return %196 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %41 = stablehlo.custom_call @__cudnn$convForward(%37, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>
    %42 = stablehlo.get_tuple_element %41[0] : (tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>) -> tensor<64x64x56x56xf32>
    %43 = "mhlo.fusion"(%arg26, %arg25, %40, %42, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<64x64x56x56xf32>, %arg272: tensor<64xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x64x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x64x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x64x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x64x56x56xf32>
      mhlo.return %203 : tensor<64x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x56x56xf32>, tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %44 = stablehlo.custom_call @__cudnn$convForward(%43, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>
    %45 = stablehlo.get_tuple_element %44[0] : (tuple<tensor<64x64x56x56xf32>, tensor<0xui8>>) -> tensor<64x64x56x56xf32>
    %46 = "mhlo.fusion"(%arg29, %arg28, %39, %45, %arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<64x64x56x56xf32>, %arg272: tensor<64xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x64x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x64x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x64x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x64x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x56x56xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x64x56x56xf32>
      mhlo.return %203 : tensor<64x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x56x56xf32>, tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %47 = stablehlo.custom_call @__cudnn$convForward(%46, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tuple<tensor<64x256x56x56xf32>, tensor<0xui8>>
    %48 = stablehlo.get_tuple_element %47[0] : (tuple<tensor<64x256x56x56xf32>, tensor<0xui8>>) -> tensor<64x256x56x56xf32>
    %49 = "mhlo.fusion"(%37, %arg32, %arg31, %38, %48, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x256x56x56xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<64x256x56x56xf32>, %arg273: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x256x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x256x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x256x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x256x56x56xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x256x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x56x56xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x256x56x56xf32>
      mhlo.return %204 : tensor<64x256x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x56x56xf32>, tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %50 = stablehlo.custom_call @__cudnn$convForward(%49, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<64x128x56x56xf32>, tensor<0xui8>>
    %51 = stablehlo.get_tuple_element %50[0] : (tuple<tensor<64x128x56x56xf32>, tensor<0xui8>>) -> tensor<64x128x56x56xf32>
    %52 = "mhlo.fusion"(%6, %arg38, %arg37, %51, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<64x128x56x56xf32>, %arg272: tensor<128xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x128x56x56xf32>
      %196 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x128x56x56xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x128x56x56xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x128x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x56x56xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x128x56x56xf32>
      mhlo.return %203 : tensor<64x128x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x56x56xf32>, tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %53 = stablehlo.custom_call @__cudnn$convForward(%52, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>
    %54 = stablehlo.get_tuple_element %53[0] : (tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>) -> tensor<64x128x28x28xf32>
    %55 = "mhlo.fusion"(%arg41, %arg40, %5, %54, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<64x128x28x28xf32>, %arg272: tensor<128xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x128x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x128x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x128x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x128x28x28xf32>
      mhlo.return %203 : tensor<64x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x28x28xf32>, tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %56 = stablehlo.custom_call @__cudnn$convForward(%55, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>
    %57 = stablehlo.get_tuple_element %56[0] : (tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>) -> tensor<64x512x28x28xf32>
    %58 = "mhlo.fusion"(%arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<512xf32>
      %196 = stablehlo.rsqrt %195 : tensor<512xf32>
      mhlo.return %196 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %59 = stablehlo.custom_call @__cudnn$convForward(%49, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>
    %60 = stablehlo.get_tuple_element %59[0] : (tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>) -> tensor<64x512x28x28xf32>
    %61 = "mhlo.fusion"(%arg44, %arg43, %4, %57, %arg189, %arg35, %arg34, %58, %60, %arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<64x512x28x28xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>, %arg274: tensor<512xf32>, %arg275: tensor<512xf32>, %arg276: tensor<64x512x28x28xf32>, %arg277: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x512x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x512x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x512x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x512x28x28xf32>
      %202 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %203 = stablehlo.subtract %arg276, %202 : tensor<64x512x28x28xf32>
      %204 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %205 = stablehlo.multiply %203, %204 : tensor<64x512x28x28xf32>
      %206 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %207 = stablehlo.multiply %205, %206 : tensor<64x512x28x28xf32>
      %208 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %209 = stablehlo.add %207, %208 : tensor<64x512x28x28xf32>
      %210 = stablehlo.add %201, %209 : tensor<64x512x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %211 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x28x28xf32>
      %212 = stablehlo.maximum %210, %211 : tensor<64x512x28x28xf32>
      mhlo.return %212 : tensor<64x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x28x28xf32>, tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %62 = "mhlo.fusion"(%arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<512xf32>
      %196 = stablehlo.rsqrt %195 : tensor<512xf32>
      mhlo.return %196 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %63 = "mhlo.fusion"(%arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<128xf32>
      %196 = stablehlo.rsqrt %195 : tensor<128xf32>
      mhlo.return %196 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %64 = "mhlo.fusion"(%arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<128xf32>
      %196 = stablehlo.rsqrt %195 : tensor<128xf32>
      mhlo.return %196 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %65 = stablehlo.custom_call @__cudnn$convForward(%61, %arg45) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>
    %66 = stablehlo.get_tuple_element %65[0] : (tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>) -> tensor<64x128x28x28xf32>
    %67 = "mhlo.fusion"(%arg47, %arg46, %64, %66, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<64x128x28x28xf32>, %arg272: tensor<128xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x128x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x128x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x128x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x128x28x28xf32>
      mhlo.return %203 : tensor<64x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x28x28xf32>, tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %68 = stablehlo.custom_call @__cudnn$convForward(%67, %arg48) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>
    %69 = stablehlo.get_tuple_element %68[0] : (tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>) -> tensor<64x128x28x28xf32>
    %70 = "mhlo.fusion"(%arg50, %arg49, %63, %69, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<64x128x28x28xf32>, %arg272: tensor<128xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x128x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x128x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x128x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x128x28x28xf32>
      mhlo.return %203 : tensor<64x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x28x28xf32>, tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %71 = stablehlo.custom_call @__cudnn$convForward(%70, %arg51) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>
    %72 = stablehlo.get_tuple_element %71[0] : (tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>) -> tensor<64x512x28x28xf32>
    %73 = "mhlo.fusion"(%61, %arg53, %arg52, %62, %72, %arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<64x512x28x28xf32>, %arg273: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x512x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x512x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x512x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x512x28x28xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x512x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x28x28xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x512x28x28xf32>
      mhlo.return %204 : tensor<64x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<64x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x28x28xf32>, tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %74 = "mhlo.fusion"(%arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<512xf32>
      %196 = stablehlo.rsqrt %195 : tensor<512xf32>
      mhlo.return %196 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %75 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<128xf32>
      %196 = stablehlo.rsqrt %195 : tensor<128xf32>
      mhlo.return %196 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %76 = "mhlo.fusion"(%arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<128xf32>
      %196 = stablehlo.rsqrt %195 : tensor<128xf32>
      mhlo.return %196 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %77 = stablehlo.custom_call @__cudnn$convForward(%73, %arg54) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>
    %78 = stablehlo.get_tuple_element %77[0] : (tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>) -> tensor<64x128x28x28xf32>
    %79 = "mhlo.fusion"(%arg56, %arg55, %76, %78, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<64x128x28x28xf32>, %arg272: tensor<128xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x128x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x128x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x128x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x128x28x28xf32>
      mhlo.return %203 : tensor<64x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x28x28xf32>, tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %80 = stablehlo.custom_call @__cudnn$convForward(%79, %arg57) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>
    %81 = stablehlo.get_tuple_element %80[0] : (tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>) -> tensor<64x128x28x28xf32>
    %82 = "mhlo.fusion"(%arg59, %arg58, %75, %81, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<64x128x28x28xf32>, %arg272: tensor<128xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x128x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x128x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x128x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x128x28x28xf32>
      mhlo.return %203 : tensor<64x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x28x28xf32>, tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %83 = stablehlo.custom_call @__cudnn$convForward(%82, %arg60) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>
    %84 = stablehlo.get_tuple_element %83[0] : (tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>) -> tensor<64x512x28x28xf32>
    %85 = "mhlo.fusion"(%73, %arg62, %arg61, %74, %84, %arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<64x512x28x28xf32>, %arg273: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x512x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x512x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x512x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x512x28x28xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x512x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x28x28xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x512x28x28xf32>
      mhlo.return %204 : tensor<64x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<64x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x28x28xf32>, tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %86 = "mhlo.fusion"(%arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<512xf32>
      %196 = stablehlo.rsqrt %195 : tensor<512xf32>
      mhlo.return %196 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %87 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<128xf32>
      %196 = stablehlo.rsqrt %195 : tensor<128xf32>
      mhlo.return %196 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %88 = "mhlo.fusion"(%arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<128xf32>
      %196 = stablehlo.rsqrt %195 : tensor<128xf32>
      mhlo.return %196 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %89 = stablehlo.custom_call @__cudnn$convForward(%85, %arg63) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>
    %90 = stablehlo.get_tuple_element %89[0] : (tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>) -> tensor<64x128x28x28xf32>
    %91 = "mhlo.fusion"(%arg65, %arg64, %88, %90, %arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<64x128x28x28xf32>, %arg272: tensor<128xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x128x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x128x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x128x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x128x28x28xf32>
      mhlo.return %203 : tensor<64x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x28x28xf32>, tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %92 = stablehlo.custom_call @__cudnn$convForward(%91, %arg66) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>
    %93 = stablehlo.get_tuple_element %92[0] : (tuple<tensor<64x128x28x28xf32>, tensor<0xui8>>) -> tensor<64x128x28x28xf32>
    %94 = "mhlo.fusion"(%arg68, %arg67, %87, %93, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<64x128x28x28xf32>, %arg272: tensor<128xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x128x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x128x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x128x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x128x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x28x28xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x128x28x28xf32>
      mhlo.return %203 : tensor<64x128x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x28x28xf32>, tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %95 = stablehlo.custom_call @__cudnn$convForward(%94, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>
    %96 = stablehlo.get_tuple_element %95[0] : (tuple<tensor<64x512x28x28xf32>, tensor<0xui8>>) -> tensor<64x512x28x28xf32>
    %97 = "mhlo.fusion"(%85, %arg71, %arg70, %86, %96, %arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x512x28x28xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<512xf32>, %arg272: tensor<64x512x28x28xf32>, %arg273: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x512x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x512x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x512x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x512x28x28xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x512x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x28x28xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x512x28x28xf32>
      mhlo.return %204 : tensor<64x512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<64x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x28x28xf32>, tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %98 = stablehlo.custom_call @__cudnn$convForward(%97, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<64x256x28x28xf32>, tensor<0xui8>>
    %99 = stablehlo.get_tuple_element %98[0] : (tuple<tensor<64x256x28x28xf32>, tensor<0xui8>>) -> tensor<64x256x28x28xf32>
    %100 = "mhlo.fusion"(%arg77, %arg76, %3, %99, %arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<64x256x28x28xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x256x28x28xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x256x28x28xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x256x28x28xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x256x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x28x28xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x256x28x28xf32>
      mhlo.return %203 : tensor<64x256x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x28x28xf32>, tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %101 = stablehlo.custom_call @__cudnn$convForward(%100, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %102 = stablehlo.get_tuple_element %101[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %103 = "mhlo.fusion"(%arg80, %arg79, %102, %arg213, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %104 = stablehlo.custom_call @__cudnn$convForward(%103, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>
    %105 = stablehlo.get_tuple_element %104[0] : (tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>) -> tensor<64x1024x14x14xf32>
    %106 = "mhlo.fusion"(%arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<1024xf32>
      %196 = stablehlo.rsqrt %195 : tensor<1024xf32>
      mhlo.return %196 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %107 = stablehlo.custom_call @__cudnn$convForward(%97, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>
    %108 = stablehlo.get_tuple_element %107[0] : (tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>) -> tensor<64x1024x14x14xf32>
    %109 = "mhlo.fusion"(%arg83, %arg82, %2, %105, %arg215, %arg74, %arg73, %106, %108, %arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<64x1024x14x14xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<64x1024x14x14xf32>, %arg277: tensor<1024xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x1024x14x14xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x1024x14x14xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x1024x14x14xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x1024x14x14xf32>
      %202 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %203 = stablehlo.subtract %arg276, %202 : tensor<64x1024x14x14xf32>
      %204 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %205 = stablehlo.multiply %203, %204 : tensor<64x1024x14x14xf32>
      %206 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %207 = stablehlo.multiply %205, %206 : tensor<64x1024x14x14xf32>
      %208 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %209 = stablehlo.add %207, %208 : tensor<64x1024x14x14xf32>
      %210 = stablehlo.add %201, %209 : tensor<64x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %211 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
      %212 = stablehlo.maximum %210, %211 : tensor<64x1024x14x14xf32>
      mhlo.return %212 : tensor<64x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64x1024x14x14xf32>, tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %110 = "mhlo.fusion"(%arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<1024xf32>
      %196 = stablehlo.rsqrt %195 : tensor<1024xf32>
      mhlo.return %196 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %111 = stablehlo.custom_call @__cudnn$convForward(%109, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %112 = stablehlo.get_tuple_element %111[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %113 = "mhlo.fusion"(%arg86, %arg85, %112, %arg217, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %114 = stablehlo.custom_call @__cudnn$convForward(%113, %arg87) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %115 = stablehlo.get_tuple_element %114[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %116 = "mhlo.fusion"(%arg89, %arg88, %115, %arg219, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %117 = stablehlo.custom_call @__cudnn$convForward(%116, %arg90) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>
    %118 = stablehlo.get_tuple_element %117[0] : (tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>) -> tensor<64x1024x14x14xf32>
    %119 = "mhlo.fusion"(%109, %arg92, %arg91, %110, %118, %arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<64x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x1024x14x14xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x1024x14x14xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x1024x14x14xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x1024x14x14xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x1024x14x14xf32>
      mhlo.return %204 : tensor<64x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64x1024x14x14xf32>, tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %120 = "mhlo.fusion"(%arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<1024xf32>
      %196 = stablehlo.rsqrt %195 : tensor<1024xf32>
      mhlo.return %196 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %121 = stablehlo.custom_call @__cudnn$convForward(%119, %arg93) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %122 = stablehlo.get_tuple_element %121[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %123 = "mhlo.fusion"(%arg95, %arg94, %122, %arg223, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %124 = stablehlo.custom_call @__cudnn$convForward(%123, %arg96) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %125 = stablehlo.get_tuple_element %124[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %126 = "mhlo.fusion"(%arg98, %arg97, %125, %arg225, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %127 = stablehlo.custom_call @__cudnn$convForward(%126, %arg99) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>
    %128 = stablehlo.get_tuple_element %127[0] : (tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>) -> tensor<64x1024x14x14xf32>
    %129 = "mhlo.fusion"(%119, %arg101, %arg100, %120, %128, %arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<64x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x1024x14x14xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x1024x14x14xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x1024x14x14xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x1024x14x14xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x1024x14x14xf32>
      mhlo.return %204 : tensor<64x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64x1024x14x14xf32>, tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %130 = "mhlo.fusion"(%arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<1024xf32>
      %196 = stablehlo.rsqrt %195 : tensor<1024xf32>
      mhlo.return %196 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %131 = stablehlo.custom_call @__cudnn$convForward(%129, %arg102) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %132 = stablehlo.get_tuple_element %131[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %133 = "mhlo.fusion"(%arg104, %arg103, %132, %arg229, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %134 = stablehlo.custom_call @__cudnn$convForward(%133, %arg105) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %135 = stablehlo.get_tuple_element %134[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %136 = "mhlo.fusion"(%arg107, %arg106, %135, %arg231, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %137 = stablehlo.custom_call @__cudnn$convForward(%136, %arg108) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>
    %138 = stablehlo.get_tuple_element %137[0] : (tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>) -> tensor<64x1024x14x14xf32>
    %139 = "mhlo.fusion"(%129, %arg110, %arg109, %130, %138, %arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<64x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x1024x14x14xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x1024x14x14xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x1024x14x14xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x1024x14x14xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x1024x14x14xf32>
      mhlo.return %204 : tensor<64x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64x1024x14x14xf32>, tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %140 = "mhlo.fusion"(%arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<1024xf32>
      %196 = stablehlo.rsqrt %195 : tensor<1024xf32>
      mhlo.return %196 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %141 = stablehlo.custom_call @__cudnn$convForward(%139, %arg111) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %142 = stablehlo.get_tuple_element %141[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %143 = "mhlo.fusion"(%arg113, %arg112, %142, %arg235, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %144 = stablehlo.custom_call @__cudnn$convForward(%143, %arg114) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %145 = stablehlo.get_tuple_element %144[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %146 = "mhlo.fusion"(%arg116, %arg115, %145, %arg237, %arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %147 = stablehlo.custom_call @__cudnn$convForward(%146, %arg117) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>
    %148 = stablehlo.get_tuple_element %147[0] : (tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>) -> tensor<64x1024x14x14xf32>
    %149 = "mhlo.fusion"(%139, %arg119, %arg118, %140, %148, %arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<64x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x1024x14x14xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x1024x14x14xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x1024x14x14xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x1024x14x14xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x1024x14x14xf32>
      mhlo.return %204 : tensor<64x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64x1024x14x14xf32>, tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %150 = "mhlo.fusion"(%arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<1024xf32>
      %196 = stablehlo.rsqrt %195 : tensor<1024xf32>
      mhlo.return %196 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %151 = stablehlo.custom_call @__cudnn$convForward(%149, %arg120) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %152 = stablehlo.get_tuple_element %151[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %153 = "mhlo.fusion"(%arg122, %arg121, %152, %arg241, %arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %154 = stablehlo.custom_call @__cudnn$convForward(%153, %arg123) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>
    %155 = stablehlo.get_tuple_element %154[0] : (tuple<tensor<64x256x14x14xf32>, tensor<0xui8>>) -> tensor<64x256x14x14xf32>
    %156 = "mhlo.fusion"(%arg125, %arg124, %155, %arg243, %arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<64x256x14x14xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x256x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<256xf32>
      %198 = stablehlo.rsqrt %197 : tensor<256xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x256x14x14xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x256x14x14xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x256x14x14xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x14x14xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x256x14x14xf32>
      mhlo.return %206 : tensor<64x256x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %157 = stablehlo.custom_call @__cudnn$convForward(%156, %arg126) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>
    %158 = stablehlo.get_tuple_element %157[0] : (tuple<tensor<64x1024x14x14xf32>, tensor<0xui8>>) -> tensor<64x1024x14x14xf32>
    %159 = "mhlo.fusion"(%149, %arg128, %arg127, %150, %158, %arg245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x1024x14x14xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<64x1024x14x14xf32>, %arg273: tensor<1024xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x1024x14x14xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x1024x14x14xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x1024x14x14xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x1024x14x14xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x1024x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x14x14xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x1024x14x14xf32>
      mhlo.return %204 : tensor<64x1024x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64x1024x14x14xf32>, tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %160 = stablehlo.custom_call @__cudnn$convForward(%159, %arg132) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tuple<tensor<64x512x14x14xf32>, tensor<0xui8>>
    %161 = stablehlo.get_tuple_element %160[0] : (tuple<tensor<64x512x14x14xf32>, tensor<0xui8>>) -> tensor<64x512x14x14xf32>
    %162 = "mhlo.fusion"(%arg134, %arg133, %1, %161, %arg249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<64x512x14x14xf32>, %arg272: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x512x14x14xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x512x14x14xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x512x14x14xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x512x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x14x14xf32>
      %203 = stablehlo.maximum %201, %202 : tensor<64x512x14x14xf32>
      mhlo.return %203 : tensor<64x512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x14x14xf32>, tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %163 = stablehlo.custom_call @__cudnn$convForward(%162, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>
    %164 = stablehlo.get_tuple_element %163[0] : (tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>) -> tensor<64x512x7x7xf32>
    %165 = "mhlo.fusion"(%arg137, %arg136, %164, %arg251, %arg252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<64x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<512xf32>
      %198 = stablehlo.rsqrt %197 : tensor<512xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x512x7x7xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x512x7x7xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x512x7x7xf32>
      mhlo.return %206 : tensor<64x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<64x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %166 = stablehlo.custom_call @__cudnn$convForward(%165, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<64x2048x7x7xf32>, tensor<0xui8>>
    %167 = stablehlo.get_tuple_element %166[0] : (tuple<tensor<64x2048x7x7xf32>, tensor<0xui8>>) -> tensor<64x2048x7x7xf32>
    %168 = "mhlo.fusion"(%arg248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<2048xf32>
      %196 = stablehlo.rsqrt %195 : tensor<2048xf32>
      mhlo.return %196 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %169 = stablehlo.custom_call @__cudnn$convForward(%159, %arg129) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tuple<tensor<64x2048x7x7xf32>, tensor<0xui8>>
    %170 = stablehlo.get_tuple_element %169[0] : (tuple<tensor<64x2048x7x7xf32>, tensor<0xui8>>) -> tensor<64x2048x7x7xf32>
    %171 = "mhlo.fusion"(%arg140, %arg139, %0, %167, %arg253, %arg131, %arg130, %168, %170, %arg247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<64x2048x7x7xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>, %arg274: tensor<2048xf32>, %arg275: tensor<2048xf32>, %arg276: tensor<64x2048x7x7xf32>, %arg277: tensor<2048xf32>):
      %194 = stablehlo.broadcast_in_dim %arg272, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %195 = stablehlo.subtract %arg271, %194 : tensor<64x2048x7x7xf32>
      %196 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x2048x7x7xf32>
      %198 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x2048x7x7xf32>
      %200 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x2048x7x7xf32>
      %202 = stablehlo.broadcast_in_dim %arg277, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %203 = stablehlo.subtract %arg276, %202 : tensor<64x2048x7x7xf32>
      %204 = stablehlo.broadcast_in_dim %arg275, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %205 = stablehlo.multiply %203, %204 : tensor<64x2048x7x7xf32>
      %206 = stablehlo.broadcast_in_dim %arg274, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %207 = stablehlo.multiply %205, %206 : tensor<64x2048x7x7xf32>
      %208 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %209 = stablehlo.add %207, %208 : tensor<64x2048x7x7xf32>
      %210 = stablehlo.add %201, %209 : tensor<64x2048x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %211 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x2048x7x7xf32>
      %212 = stablehlo.maximum %210, %211 : tensor<64x2048x7x7xf32>
      mhlo.return %212 : tensor<64x2048x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048x7x7xf32>, tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %172 = "mhlo.fusion"(%arg260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<2048xf32>
      %196 = stablehlo.rsqrt %195 : tensor<2048xf32>
      mhlo.return %196 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %173 = stablehlo.custom_call @__cudnn$convForward(%171, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>
    %174 = stablehlo.get_tuple_element %173[0] : (tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>) -> tensor<64x512x7x7xf32>
    %175 = "mhlo.fusion"(%arg143, %arg142, %174, %arg255, %arg256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<64x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<512xf32>
      %198 = stablehlo.rsqrt %197 : tensor<512xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x512x7x7xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x512x7x7xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x512x7x7xf32>
      mhlo.return %206 : tensor<64x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<64x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %176 = stablehlo.custom_call @__cudnn$convForward(%175, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>
    %177 = stablehlo.get_tuple_element %176[0] : (tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>) -> tensor<64x512x7x7xf32>
    %178 = "mhlo.fusion"(%arg146, %arg145, %177, %arg257, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<64x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<512xf32>
      %198 = stablehlo.rsqrt %197 : tensor<512xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x512x7x7xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x512x7x7xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x512x7x7xf32>
      mhlo.return %206 : tensor<64x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<64x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %179 = stablehlo.custom_call @__cudnn$convForward(%178, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<64x2048x7x7xf32>, tensor<0xui8>>
    %180 = stablehlo.get_tuple_element %179[0] : (tuple<tensor<64x2048x7x7xf32>, tensor<0xui8>>) -> tensor<64x2048x7x7xf32>
    %181 = "mhlo.fusion"(%171, %arg149, %arg148, %172, %180, %arg259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x2048x7x7xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<2048xf32>, %arg272: tensor<64x2048x7x7xf32>, %arg273: tensor<2048xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x2048x7x7xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x2048x7x7xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x2048x7x7xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x2048x7x7xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x2048x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x2048x7x7xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x2048x7x7xf32>
      mhlo.return %204 : tensor<64x2048x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048x7x7xf32>, tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %182 = "mhlo.fusion"(%arg266) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %195 = stablehlo.add %arg268, %194 : tensor<2048xf32>
      %196 = stablehlo.rsqrt %195 : tensor<2048xf32>
      mhlo.return %196 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %183 = stablehlo.custom_call @__cudnn$convForward(%181, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>
    %184 = stablehlo.get_tuple_element %183[0] : (tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>) -> tensor<64x512x7x7xf32>
    %185 = "mhlo.fusion"(%arg152, %arg151, %184, %arg261, %arg262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<64x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<512xf32>
      %198 = stablehlo.rsqrt %197 : tensor<512xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x512x7x7xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x512x7x7xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x512x7x7xf32>
      mhlo.return %206 : tensor<64x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<64x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %186 = stablehlo.custom_call @__cudnn$convForward(%185, %arg153) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>
    %187 = stablehlo.get_tuple_element %186[0] : (tuple<tensor<64x512x7x7xf32>, tensor<0xui8>>) -> tensor<64x512x7x7xf32>
    %188 = "mhlo.fusion"(%arg155, %arg154, %187, %arg263, %arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<64x512x7x7xf32>, %arg271: tensor<512xf32>, %arg272: tensor<512xf32>):
      %194 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %195 = stablehlo.subtract %arg270, %194 : tensor<64x512x7x7xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %197 = stablehlo.add %arg272, %196 : tensor<512xf32>
      %198 = stablehlo.rsqrt %197 : tensor<512xf32>
      %199 = stablehlo.broadcast_in_dim %198, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %200 = stablehlo.multiply %195, %199 : tensor<64x512x7x7xf32>
      %201 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %202 = stablehlo.multiply %200, %201 : tensor<64x512x7x7xf32>
      %203 = stablehlo.broadcast_in_dim %arg268, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
      %204 = stablehlo.add %202, %203 : tensor<64x512x7x7xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x7x7xf32>
      %206 = stablehlo.maximum %204, %205 : tensor<64x512x7x7xf32>
      mhlo.return %206 : tensor<64x512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<64x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %189 = stablehlo.custom_call @__cudnn$convForward(%188, %arg156) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,2048,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tuple<tensor<64x2048x7x7xf32>, tensor<0xui8>>
    %190 = stablehlo.get_tuple_element %189[0] : (tuple<tensor<64x2048x7x7xf32>, tensor<0xui8>>) -> tensor<64x2048x7x7xf32>
    %191 = "mhlo.fusion"(%181, %arg158, %arg157, %182, %190, %arg265) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg268: tensor<64x2048x7x7xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<2048xf32>, %arg272: tensor<64x2048x7x7xf32>, %arg273: tensor<2048xf32>):
      %194 = stablehlo.broadcast_in_dim %arg273, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %195 = stablehlo.subtract %arg272, %194 : tensor<64x2048x7x7xf32>
      %196 = stablehlo.broadcast_in_dim %arg271, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %197 = stablehlo.multiply %195, %196 : tensor<64x2048x7x7xf32>
      %198 = stablehlo.broadcast_in_dim %arg270, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %199 = stablehlo.multiply %197, %198 : tensor<64x2048x7x7xf32>
      %200 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
      %201 = stablehlo.add %199, %200 : tensor<64x2048x7x7xf32>
      %202 = stablehlo.add %201, %arg268 : tensor<64x2048x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x2048x7x7xf32>
      %204 = stablehlo.maximum %202, %203 : tensor<64x2048x7x7xf32>
      %205 = mhlo.bitcast %204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x49xf32>
      %206 = stablehlo.reduce(%205 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x2048x49xf32>, tensor<f32>) -> tensor<64x2048xf32>
      mhlo.return %206 : tensor<64x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64x2048x7x7xf32>, tensor<2048xf32>) -> tensor<64x2048xf32>
    %192 = "mhlo.fusion"(%191, %arg159) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg268: tensor<64x2048xf32>, %arg269: tensor<1000x2048xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x2048xf32>
      %195 = stablehlo.multiply %arg268, %194 : tensor<64x2048xf32>
      %196 = stablehlo.dot_general %195, %arg269, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<1000x2048xf32>) -> tensor<64x1000xf32>
      mhlo.return %196 : tensor<64x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2048xf32>, tensor<1000x2048xf32>) -> tensor<64x1000xf32>
    %193 = "mhlo.fusion"(%192, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x1000xf32>, %arg269: tensor<1000xf32>):
      %194 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
      %195 = stablehlo.add %194, %arg268 : tensor<64x1000xf32>
      mhlo.return %195 : tensor<64x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1000xf32>, tensor<1000xf32>) -> tensor<64x1000xf32>
    return %193 : tensor<64x1000xf32>
  }
}
