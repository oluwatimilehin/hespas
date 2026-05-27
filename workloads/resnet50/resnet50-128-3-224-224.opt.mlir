module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_1.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<128x7x7x2048xf32>, %arg1: tensor<f32>) -> tensor<128x2048xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2, 1] : (tensor<128x7x7x2048xf32>, tensor<f32>) -> tensor<128x2048xf32>
    return %0 : tensor<128x2048xf32>
  }
  func.func private @fused_computation.1(%arg0: tensor<128x1000xf32>, %arg1: tensor<1000xf32>) -> tensor<128x1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1000xf32>) -> tensor<128x1000xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<128x1000xf32>
    return %1 : tensor<128x1000xf32>
  }
  func.func private @fused_computation.2(%arg0: tensor<128x2048xf32>) -> tensor<128x2048xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x2048xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x2048xf32>
    return %1 : tensor<128x2048xf32>
  }
  func.func private @fused_computation.3(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<128x7x7x2048xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<2048xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<128x7x7x2048xf32>, %arg9: tensor<2048xf32>, %arg10: tensor<2048xf32>, %arg11: tensor<2048xf32>, %arg12: tensor<2048xf32>, %arg13: tensor<128x7x7x2048xf32>, %arg14: tensor<2048xf32>, %arg15: tensor<2048xf32>, %arg16: tensor<2048xf32>, %arg17: tensor<2048xf32>, %arg18: tensor<128x7x7x2048xf32>, %arg19: tensor<2048xf32>) -> tensor<128x7x7x2048xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x7x7x2048xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x7x7x2048xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x7x7x2048xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x7x7x2048xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x7x7x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x7x7x2048xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x7x7x2048xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x7x7x2048xf32>
    %16 = stablehlo.broadcast_in_dim %arg19, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %17 = stablehlo.subtract %arg18, %16 : tensor<128x7x7x2048xf32>
    %18 = stablehlo.broadcast_in_dim %arg17, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x7x7x2048xf32>
    %20 = stablehlo.broadcast_in_dim %arg16, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x7x7x2048xf32>
    %22 = stablehlo.broadcast_in_dim %arg15, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x7x7x2048xf32>
    %24 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %25 = stablehlo.subtract %arg13, %24 : tensor<128x7x7x2048xf32>
    %26 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x7x7x2048xf32>
    %28 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x7x7x2048xf32>
    %30 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x7x7x2048xf32>
    %32 = stablehlo.add %23, %31 : tensor<128x7x7x2048xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x7x7x2048xf32>
    %34 = stablehlo.maximum %32, %33 : tensor<128x7x7x2048xf32>
    %35 = stablehlo.add %15, %34 : tensor<128x7x7x2048xf32>
    %36 = stablehlo.maximum %35, %33 : tensor<128x7x7x2048xf32>
    %37 = stablehlo.add %7, %36 : tensor<128x7x7x2048xf32>
    %38 = stablehlo.maximum %37, %33 : tensor<128x7x7x2048xf32>
    return %38 : tensor<128x7x7x2048xf32>
  }
  func.func private @fused_computation.4(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
  func.func private @fused_computation.5(%arg0: tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,2048]{1,0,2,3}"} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x2048xf32>
    return %1 : tensor<1x1x512x2048xf32>
  }
  func.func private @fused_computation.6(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x7x7x512xf32>, %arg4: tensor<512xf32>) -> tensor<128x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x7x7x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x7x7x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x7x7x512xf32>
    return %9 : tensor<128x7x7x512xf32>
  }
  func.func private @fused_computation.7(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.8(%arg0: tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
    %1 = mhlo.copy %0 : tensor<3x3x512x512xf32>
    return %1 : tensor<3x3x512x512xf32>
  }
  func.func private @fused_computation.9(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x7x7x512xf32>, %arg4: tensor<512xf32>) -> tensor<128x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x7x7x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x7x7x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x7x7x512xf32>
    return %9 : tensor<128x7x7x512xf32>
  }
  func.func private @fused_computation.10(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.11(%arg0: tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,2048,512]{1,0,2,3}"} : (tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x2048x512xf32>
    return %1 : tensor<1x1x2048x512xf32>
  }
  func.func private @fused_computation.12(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<128x7x7x2048xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<2048xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<128x7x7x2048xf32>, %arg9: tensor<2048xf32>, %arg10: tensor<2048xf32>, %arg11: tensor<2048xf32>, %arg12: tensor<2048xf32>, %arg13: tensor<128x7x7x2048xf32>, %arg14: tensor<2048xf32>) -> tensor<128x7x7x2048xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x7x7x2048xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x7x7x2048xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x7x7x2048xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x7x7x2048xf32>
    %8 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %9 = stablehlo.subtract %arg13, %8 : tensor<128x7x7x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x7x7x2048xf32>
    %12 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x7x7x2048xf32>
    %14 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x7x7x2048xf32>
    %16 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %17 = stablehlo.subtract %arg8, %16 : tensor<128x7x7x2048xf32>
    %18 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x7x7x2048xf32>
    %20 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x7x7x2048xf32>
    %22 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x7x7x2048xf32>
    %24 = stablehlo.add %15, %23 : tensor<128x7x7x2048xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x7x7x2048xf32>
    %26 = stablehlo.maximum %24, %25 : tensor<128x7x7x2048xf32>
    %27 = stablehlo.add %7, %26 : tensor<128x7x7x2048xf32>
    %28 = stablehlo.maximum %27, %25 : tensor<128x7x7x2048xf32>
    return %28 : tensor<128x7x7x2048xf32>
  }
  func.func private @fused_computation.13(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
  func.func private @fused_computation.14(%arg0: tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,2048]{1,0,2,3}"} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x2048xf32>
    return %1 : tensor<1x1x512x2048xf32>
  }
  func.func private @fused_computation.15(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x7x7x512xf32>, %arg4: tensor<512xf32>) -> tensor<128x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x7x7x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x7x7x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x7x7x512xf32>
    return %9 : tensor<128x7x7x512xf32>
  }
  func.func private @fused_computation.16(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.17(%arg0: tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
    %1 = mhlo.copy %0 : tensor<3x3x512x512xf32>
    return %1 : tensor<3x3x512x512xf32>
  }
  func.func private @fused_computation.18(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x7x7x512xf32>, %arg4: tensor<512xf32>) -> tensor<128x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x7x7x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x7x7x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x7x7x512xf32>
    return %9 : tensor<128x7x7x512xf32>
  }
  func.func private @fused_computation.19(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.20(%arg0: tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,2048,512]{1,0,2,3}"} : (tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x2048x512xf32>
    return %1 : tensor<1x1x2048x512xf32>
  }
  func.func private @fused_computation.21(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<128x7x7x2048xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<2048xf32>, %arg7: tensor<2048xf32>, %arg8: tensor<128x7x7x2048xf32>, %arg9: tensor<2048xf32>) -> tensor<128x7x7x2048xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<128x7x7x2048xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x7x7x2048xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x7x7x2048xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x7x7x2048xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<128x7x7x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x7x7x2048xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x7x7x2048xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x7x7x2048xf32>
    %16 = stablehlo.add %7, %15 : tensor<128x7x7x2048xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x7x7x2048xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<128x7x7x2048xf32>
    return %18 : tensor<128x7x7x2048xf32>
  }
  func.func private @fused_computation.22(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
  func.func private @fused_computation.23(%arg0: tensor<2048x1024x1x1xf32>) -> tensor<1x1x1024x2048xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,2048]{1,0,2,3}"} : (tensor<2048x1024x1x1xf32>) -> tensor<1x1x1024x2048xf32>
    %1 = mhlo.copy %0 : tensor<1x1x1024x2048xf32>
    return %1 : tensor<1x1x1024x2048xf32>
  }
  func.func private @fused_computation.24(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    return %2 : tensor<2048xf32>
  }
  func.func private @fused_computation.25(%arg0: tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,2048]{1,0,2,3}"} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x2048xf32>
    return %1 : tensor<1x1x512x2048xf32>
  }
  func.func private @fused_computation.26(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x7x7x512xf32>, %arg4: tensor<512xf32>) -> tensor<128x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x7x7x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x7x7x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x7x7x512xf32>
    return %9 : tensor<128x7x7x512xf32>
  }
  func.func private @fused_computation.27(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.28(%arg0: tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
    %1 = mhlo.copy %0 : tensor<3x3x512x512xf32>
    return %1 : tensor<3x3x512x512xf32>
  }
  func.func private @fused_computation.29(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x14x14x512xf32>, %arg4: tensor<512xf32>) -> tensor<128x14x14x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x14x14x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x14x14x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x14x14x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x14x14x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x512xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x512xf32>
    return %9 : tensor<128x14x14x512xf32>
  }
  func.func private @fused_computation.30(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.31(%arg0: tensor<512x1024x1x1xf32>) -> tensor<1x1x1024x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,512]{1,0,2,3}"} : (tensor<512x1024x1x1xf32>) -> tensor<1x1x1024x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x1024x512xf32>
    return %1 : tensor<1x1x1024x512xf32>
  }
  func.func private @fused_computation.32(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<128x14x14x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<128x14x14x1024xf32>, %arg9: tensor<1024xf32>, %arg10: tensor<1024xf32>, %arg11: tensor<1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<128x14x14x1024xf32>, %arg14: tensor<1024xf32>, %arg15: tensor<1024xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<1024xf32>, %arg18: tensor<128x14x14x1024xf32>, %arg19: tensor<1024xf32>, %arg20: tensor<1024xf32>, %arg21: tensor<1024xf32>, %arg22: tensor<1024xf32>, %arg23: tensor<128x14x14x1024xf32>, %arg24: tensor<1024xf32>, %arg25: tensor<1024xf32>, %arg26: tensor<1024xf32>, %arg27: tensor<1024xf32>, %arg28: tensor<128x14x14x1024xf32>, %arg29: tensor<1024xf32>, %arg30: tensor<1024xf32>, %arg31: tensor<1024xf32>, %arg32: tensor<1024xf32>, %arg33: tensor<128x14x14x1024xf32>, %arg34: tensor<1024xf32>) -> tensor<128x14x14x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x1024xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x1024xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x1024xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x1024xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x14x14x1024xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x14x14x1024xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x14x14x1024xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x14x14x1024xf32>
    %16 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %17 = stablehlo.subtract %arg13, %16 : tensor<128x14x14x1024xf32>
    %18 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x14x14x1024xf32>
    %20 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x14x14x1024xf32>
    %22 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x14x14x1024xf32>
    %24 = stablehlo.broadcast_in_dim %arg19, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %25 = stablehlo.subtract %arg18, %24 : tensor<128x14x14x1024xf32>
    %26 = stablehlo.broadcast_in_dim %arg17, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x14x14x1024xf32>
    %28 = stablehlo.broadcast_in_dim %arg16, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x14x14x1024xf32>
    %30 = stablehlo.broadcast_in_dim %arg15, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x14x14x1024xf32>
    %32 = stablehlo.broadcast_in_dim %arg24, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %33 = stablehlo.subtract %arg23, %32 : tensor<128x14x14x1024xf32>
    %34 = stablehlo.broadcast_in_dim %arg22, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %35 = stablehlo.multiply %33, %34 : tensor<128x14x14x1024xf32>
    %36 = stablehlo.broadcast_in_dim %arg21, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<128x14x14x1024xf32>
    %38 = stablehlo.broadcast_in_dim %arg20, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %39 = stablehlo.add %37, %38 : tensor<128x14x14x1024xf32>
    %40 = stablehlo.broadcast_in_dim %arg34, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %41 = stablehlo.subtract %arg33, %40 : tensor<128x14x14x1024xf32>
    %42 = stablehlo.broadcast_in_dim %arg32, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<128x14x14x1024xf32>
    %44 = stablehlo.broadcast_in_dim %arg31, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<128x14x14x1024xf32>
    %46 = stablehlo.broadcast_in_dim %arg30, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %47 = stablehlo.add %45, %46 : tensor<128x14x14x1024xf32>
    %48 = stablehlo.broadcast_in_dim %arg29, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %49 = stablehlo.subtract %arg28, %48 : tensor<128x14x14x1024xf32>
    %50 = stablehlo.broadcast_in_dim %arg27, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %51 = stablehlo.multiply %49, %50 : tensor<128x14x14x1024xf32>
    %52 = stablehlo.broadcast_in_dim %arg26, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %53 = stablehlo.multiply %51, %52 : tensor<128x14x14x1024xf32>
    %54 = stablehlo.broadcast_in_dim %arg25, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %55 = stablehlo.add %53, %54 : tensor<128x14x14x1024xf32>
    %56 = stablehlo.add %47, %55 : tensor<128x14x14x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
    %58 = stablehlo.maximum %56, %57 : tensor<128x14x14x1024xf32>
    %59 = stablehlo.add %39, %58 : tensor<128x14x14x1024xf32>
    %60 = stablehlo.maximum %59, %57 : tensor<128x14x14x1024xf32>
    %61 = stablehlo.add %31, %60 : tensor<128x14x14x1024xf32>
    %62 = stablehlo.maximum %61, %57 : tensor<128x14x14x1024xf32>
    %63 = stablehlo.add %23, %62 : tensor<128x14x14x1024xf32>
    %64 = stablehlo.maximum %63, %57 : tensor<128x14x14x1024xf32>
    %65 = stablehlo.add %15, %64 : tensor<128x14x14x1024xf32>
    %66 = stablehlo.maximum %65, %57 : tensor<128x14x14x1024xf32>
    %67 = stablehlo.add %7, %66 : tensor<128x14x14x1024xf32>
    %68 = stablehlo.maximum %67, %57 : tensor<128x14x14x1024xf32>
    return %68 : tensor<128x14x14x1024xf32>
  }
  func.func private @fused_computation.33(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_computation.34(%arg0: tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x1024xf32>
    return %1 : tensor<1x1x256x1024xf32>
  }
  func.func private @fused_computation.35(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.36(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.37(%arg0: tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x256x256xf32>
    return %1 : tensor<3x3x256x256xf32>
  }
  func.func private @fused_computation.38(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.39(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.40(%arg0: tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x1024x256xf32>
    return %1 : tensor<1x1x1024x256xf32>
  }
  func.func private @fused_computation.41(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<128x14x14x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<128x14x14x1024xf32>, %arg9: tensor<1024xf32>, %arg10: tensor<1024xf32>, %arg11: tensor<1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<128x14x14x1024xf32>, %arg14: tensor<1024xf32>, %arg15: tensor<1024xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<1024xf32>, %arg18: tensor<128x14x14x1024xf32>, %arg19: tensor<1024xf32>, %arg20: tensor<1024xf32>, %arg21: tensor<1024xf32>, %arg22: tensor<1024xf32>, %arg23: tensor<128x14x14x1024xf32>, %arg24: tensor<1024xf32>, %arg25: tensor<1024xf32>, %arg26: tensor<1024xf32>, %arg27: tensor<1024xf32>, %arg28: tensor<128x14x14x1024xf32>, %arg29: tensor<1024xf32>) -> tensor<128x14x14x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x1024xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x1024xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x1024xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x1024xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x14x14x1024xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x14x14x1024xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x14x14x1024xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x14x14x1024xf32>
    %16 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %17 = stablehlo.subtract %arg13, %16 : tensor<128x14x14x1024xf32>
    %18 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x14x14x1024xf32>
    %20 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x14x14x1024xf32>
    %22 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x14x14x1024xf32>
    %24 = stablehlo.broadcast_in_dim %arg19, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %25 = stablehlo.subtract %arg18, %24 : tensor<128x14x14x1024xf32>
    %26 = stablehlo.broadcast_in_dim %arg17, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x14x14x1024xf32>
    %28 = stablehlo.broadcast_in_dim %arg16, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x14x14x1024xf32>
    %30 = stablehlo.broadcast_in_dim %arg15, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x14x14x1024xf32>
    %32 = stablehlo.broadcast_in_dim %arg29, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %33 = stablehlo.subtract %arg28, %32 : tensor<128x14x14x1024xf32>
    %34 = stablehlo.broadcast_in_dim %arg27, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %35 = stablehlo.multiply %33, %34 : tensor<128x14x14x1024xf32>
    %36 = stablehlo.broadcast_in_dim %arg26, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<128x14x14x1024xf32>
    %38 = stablehlo.broadcast_in_dim %arg25, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %39 = stablehlo.add %37, %38 : tensor<128x14x14x1024xf32>
    %40 = stablehlo.broadcast_in_dim %arg24, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %41 = stablehlo.subtract %arg23, %40 : tensor<128x14x14x1024xf32>
    %42 = stablehlo.broadcast_in_dim %arg22, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<128x14x14x1024xf32>
    %44 = stablehlo.broadcast_in_dim %arg21, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<128x14x14x1024xf32>
    %46 = stablehlo.broadcast_in_dim %arg20, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %47 = stablehlo.add %45, %46 : tensor<128x14x14x1024xf32>
    %48 = stablehlo.add %39, %47 : tensor<128x14x14x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
    %50 = stablehlo.maximum %48, %49 : tensor<128x14x14x1024xf32>
    %51 = stablehlo.add %31, %50 : tensor<128x14x14x1024xf32>
    %52 = stablehlo.maximum %51, %49 : tensor<128x14x14x1024xf32>
    %53 = stablehlo.add %23, %52 : tensor<128x14x14x1024xf32>
    %54 = stablehlo.maximum %53, %49 : tensor<128x14x14x1024xf32>
    %55 = stablehlo.add %15, %54 : tensor<128x14x14x1024xf32>
    %56 = stablehlo.maximum %55, %49 : tensor<128x14x14x1024xf32>
    %57 = stablehlo.add %7, %56 : tensor<128x14x14x1024xf32>
    %58 = stablehlo.maximum %57, %49 : tensor<128x14x14x1024xf32>
    return %58 : tensor<128x14x14x1024xf32>
  }
  func.func private @fused_computation.42(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_computation.43(%arg0: tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x1024xf32>
    return %1 : tensor<1x1x256x1024xf32>
  }
  func.func private @fused_computation.44(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.45(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.46(%arg0: tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x256x256xf32>
    return %1 : tensor<3x3x256x256xf32>
  }
  func.func private @fused_computation.47(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.48(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.49(%arg0: tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x1024x256xf32>
    return %1 : tensor<1x1x1024x256xf32>
  }
  func.func private @fused_computation.50(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<128x14x14x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<128x14x14x1024xf32>, %arg9: tensor<1024xf32>, %arg10: tensor<1024xf32>, %arg11: tensor<1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<128x14x14x1024xf32>, %arg14: tensor<1024xf32>, %arg15: tensor<1024xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<1024xf32>, %arg18: tensor<128x14x14x1024xf32>, %arg19: tensor<1024xf32>, %arg20: tensor<1024xf32>, %arg21: tensor<1024xf32>, %arg22: tensor<1024xf32>, %arg23: tensor<128x14x14x1024xf32>, %arg24: tensor<1024xf32>) -> tensor<128x14x14x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x1024xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x1024xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x1024xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x1024xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x14x14x1024xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x14x14x1024xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x14x14x1024xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x14x14x1024xf32>
    %16 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %17 = stablehlo.subtract %arg13, %16 : tensor<128x14x14x1024xf32>
    %18 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x14x14x1024xf32>
    %20 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x14x14x1024xf32>
    %22 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x14x14x1024xf32>
    %24 = stablehlo.broadcast_in_dim %arg24, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %25 = stablehlo.subtract %arg23, %24 : tensor<128x14x14x1024xf32>
    %26 = stablehlo.broadcast_in_dim %arg22, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x14x14x1024xf32>
    %28 = stablehlo.broadcast_in_dim %arg21, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x14x14x1024xf32>
    %30 = stablehlo.broadcast_in_dim %arg20, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x14x14x1024xf32>
    %32 = stablehlo.broadcast_in_dim %arg19, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %33 = stablehlo.subtract %arg18, %32 : tensor<128x14x14x1024xf32>
    %34 = stablehlo.broadcast_in_dim %arg17, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %35 = stablehlo.multiply %33, %34 : tensor<128x14x14x1024xf32>
    %36 = stablehlo.broadcast_in_dim %arg16, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<128x14x14x1024xf32>
    %38 = stablehlo.broadcast_in_dim %arg15, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %39 = stablehlo.add %37, %38 : tensor<128x14x14x1024xf32>
    %40 = stablehlo.add %31, %39 : tensor<128x14x14x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
    %42 = stablehlo.maximum %40, %41 : tensor<128x14x14x1024xf32>
    %43 = stablehlo.add %23, %42 : tensor<128x14x14x1024xf32>
    %44 = stablehlo.maximum %43, %41 : tensor<128x14x14x1024xf32>
    %45 = stablehlo.add %15, %44 : tensor<128x14x14x1024xf32>
    %46 = stablehlo.maximum %45, %41 : tensor<128x14x14x1024xf32>
    %47 = stablehlo.add %7, %46 : tensor<128x14x14x1024xf32>
    %48 = stablehlo.maximum %47, %41 : tensor<128x14x14x1024xf32>
    return %48 : tensor<128x14x14x1024xf32>
  }
  func.func private @fused_computation.51(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_computation.52(%arg0: tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x1024xf32>
    return %1 : tensor<1x1x256x1024xf32>
  }
  func.func private @fused_computation.53(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.54(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.55(%arg0: tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x256x256xf32>
    return %1 : tensor<3x3x256x256xf32>
  }
  func.func private @fused_computation.56(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.57(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.58(%arg0: tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x1024x256xf32>
    return %1 : tensor<1x1x1024x256xf32>
  }
  func.func private @fused_computation.59(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<128x14x14x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<128x14x14x1024xf32>, %arg9: tensor<1024xf32>, %arg10: tensor<1024xf32>, %arg11: tensor<1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<128x14x14x1024xf32>, %arg14: tensor<1024xf32>, %arg15: tensor<1024xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<1024xf32>, %arg18: tensor<128x14x14x1024xf32>, %arg19: tensor<1024xf32>) -> tensor<128x14x14x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x1024xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x1024xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x1024xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x1024xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x14x14x1024xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x14x14x1024xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x14x14x1024xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x14x14x1024xf32>
    %16 = stablehlo.broadcast_in_dim %arg19, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %17 = stablehlo.subtract %arg18, %16 : tensor<128x14x14x1024xf32>
    %18 = stablehlo.broadcast_in_dim %arg17, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x14x14x1024xf32>
    %20 = stablehlo.broadcast_in_dim %arg16, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x14x14x1024xf32>
    %22 = stablehlo.broadcast_in_dim %arg15, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x14x14x1024xf32>
    %24 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %25 = stablehlo.subtract %arg13, %24 : tensor<128x14x14x1024xf32>
    %26 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x14x14x1024xf32>
    %28 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x14x14x1024xf32>
    %30 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x14x14x1024xf32>
    %32 = stablehlo.add %23, %31 : tensor<128x14x14x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
    %34 = stablehlo.maximum %32, %33 : tensor<128x14x14x1024xf32>
    %35 = stablehlo.add %15, %34 : tensor<128x14x14x1024xf32>
    %36 = stablehlo.maximum %35, %33 : tensor<128x14x14x1024xf32>
    %37 = stablehlo.add %7, %36 : tensor<128x14x14x1024xf32>
    %38 = stablehlo.maximum %37, %33 : tensor<128x14x14x1024xf32>
    return %38 : tensor<128x14x14x1024xf32>
  }
  func.func private @fused_computation.60(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_computation.61(%arg0: tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x1024xf32>
    return %1 : tensor<1x1x256x1024xf32>
  }
  func.func private @fused_computation.62(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.63(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.64(%arg0: tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x256x256xf32>
    return %1 : tensor<3x3x256x256xf32>
  }
  func.func private @fused_computation.65(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.66(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.67(%arg0: tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x1024x256xf32>
    return %1 : tensor<1x1x1024x256xf32>
  }
  func.func private @fused_computation.68(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<128x14x14x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<128x14x14x1024xf32>, %arg9: tensor<1024xf32>, %arg10: tensor<1024xf32>, %arg11: tensor<1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<128x14x14x1024xf32>, %arg14: tensor<1024xf32>) -> tensor<128x14x14x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x1024xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x1024xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x1024xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x1024xf32>
    %8 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %9 = stablehlo.subtract %arg13, %8 : tensor<128x14x14x1024xf32>
    %10 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x14x14x1024xf32>
    %12 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x14x14x1024xf32>
    %14 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x14x14x1024xf32>
    %16 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %17 = stablehlo.subtract %arg8, %16 : tensor<128x14x14x1024xf32>
    %18 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x14x14x1024xf32>
    %20 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x14x14x1024xf32>
    %22 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x14x14x1024xf32>
    %24 = stablehlo.add %15, %23 : tensor<128x14x14x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
    %26 = stablehlo.maximum %24, %25 : tensor<128x14x14x1024xf32>
    %27 = stablehlo.add %7, %26 : tensor<128x14x14x1024xf32>
    %28 = stablehlo.maximum %27, %25 : tensor<128x14x14x1024xf32>
    return %28 : tensor<128x14x14x1024xf32>
  }
  func.func private @fused_computation.69(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_computation.70(%arg0: tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x1024xf32>
    return %1 : tensor<1x1x256x1024xf32>
  }
  func.func private @fused_computation.71(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.72(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x256x256xf32>
    return %1 : tensor<3x3x256x256xf32>
  }
  func.func private @fused_computation.74(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.76(%arg0: tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x1024x256xf32>
    return %1 : tensor<1x1x1024x256xf32>
  }
  func.func private @fused_computation.77(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<128x14x14x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<128x14x14x1024xf32>, %arg9: tensor<1024xf32>) -> tensor<128x14x14x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<128x14x14x1024xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x1024xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x1024xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x1024xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<128x14x14x1024xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x14x14x1024xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x14x14x1024xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x14x14x1024xf32>
    %16 = stablehlo.add %7, %15 : tensor<128x14x14x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<128x14x14x1024xf32>
    return %18 : tensor<128x14x14x1024xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_computation.79(%arg0: tensor<1024x512x1x1xf32>) -> tensor<1x1x512x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,1024]{1,0,2,3}"} : (tensor<1024x512x1x1xf32>) -> tensor<1x1x512x1024xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x1024xf32>
    return %1 : tensor<1x1x512x1024xf32>
  }
  func.func private @fused_computation.80(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x1024xf32>
    return %1 : tensor<1x1x256x1024xf32>
  }
  func.func private @fused_computation.82(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x14x14x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x14x14x256xf32>
    return %9 : tensor<128x14x14x256xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x256x256xf32>
    return %1 : tensor<3x3x256x256xf32>
  }
  func.func private @fused_computation.85(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x28x28x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x28x28x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x28x28x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x28x28x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x28x28x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x28x28x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x28x28x256xf32>
    return %9 : tensor<128x28x28x256xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.87(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.88(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x28x28x512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512xf32>, %arg8: tensor<128x28x28x512xf32>, %arg9: tensor<512xf32>, %arg10: tensor<512xf32>, %arg11: tensor<512xf32>, %arg12: tensor<512xf32>, %arg13: tensor<128x28x28x512xf32>, %arg14: tensor<512xf32>, %arg15: tensor<512xf32>, %arg16: tensor<512xf32>, %arg17: tensor<512xf32>, %arg18: tensor<128x28x28x512xf32>, %arg19: tensor<512xf32>, %arg20: tensor<512xf32>, %arg21: tensor<512xf32>, %arg22: tensor<512xf32>, %arg23: tensor<128x28x28x512xf32>, %arg24: tensor<512xf32>) -> tensor<128x28x28x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x28x28x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x28x28x512xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x28x28x512xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x28x28x512xf32>
    %16 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %17 = stablehlo.subtract %arg13, %16 : tensor<128x28x28x512xf32>
    %18 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x28x28x512xf32>
    %20 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x28x28x512xf32>
    %22 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x28x28x512xf32>
    %24 = stablehlo.broadcast_in_dim %arg24, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %25 = stablehlo.subtract %arg23, %24 : tensor<128x28x28x512xf32>
    %26 = stablehlo.broadcast_in_dim %arg22, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x28x28x512xf32>
    %28 = stablehlo.broadcast_in_dim %arg21, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x28x28x512xf32>
    %30 = stablehlo.broadcast_in_dim %arg20, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x28x28x512xf32>
    %32 = stablehlo.broadcast_in_dim %arg19, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %33 = stablehlo.subtract %arg18, %32 : tensor<128x28x28x512xf32>
    %34 = stablehlo.broadcast_in_dim %arg17, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %35 = stablehlo.multiply %33, %34 : tensor<128x28x28x512xf32>
    %36 = stablehlo.broadcast_in_dim %arg16, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<128x28x28x512xf32>
    %38 = stablehlo.broadcast_in_dim %arg15, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %39 = stablehlo.add %37, %38 : tensor<128x28x28x512xf32>
    %40 = stablehlo.add %31, %39 : tensor<128x28x28x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x512xf32>
    %42 = stablehlo.maximum %40, %41 : tensor<128x28x28x512xf32>
    %43 = stablehlo.add %23, %42 : tensor<128x28x28x512xf32>
    %44 = stablehlo.maximum %43, %41 : tensor<128x28x28x512xf32>
    %45 = stablehlo.add %15, %44 : tensor<128x28x28x512xf32>
    %46 = stablehlo.maximum %45, %41 : tensor<128x28x28x512xf32>
    %47 = stablehlo.add %7, %46 : tensor<128x28x28x512xf32>
    %48 = stablehlo.maximum %47, %41 : tensor<128x28x28x512xf32>
    return %48 : tensor<128x28x28x512xf32>
  }
  func.func private @fused_computation.89(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x512xf32>
    return %1 : tensor<1x1x128x512xf32>
  }
  func.func private @fused_computation.91(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<128x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x28x28x128xf32>
    return %9 : tensor<128x28x28x128xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.93(%arg0: tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x128x128xf32>
    return %1 : tensor<3x3x128x128xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<128x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x28x28x128xf32>
    return %9 : tensor<128x28x28x128xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,128]{1,0,2,3}"} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x128xf32>
    return %1 : tensor<1x1x512x128xf32>
  }
  func.func private @fused_computation.97(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x28x28x512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512xf32>, %arg8: tensor<128x28x28x512xf32>, %arg9: tensor<512xf32>, %arg10: tensor<512xf32>, %arg11: tensor<512xf32>, %arg12: tensor<512xf32>, %arg13: tensor<128x28x28x512xf32>, %arg14: tensor<512xf32>, %arg15: tensor<512xf32>, %arg16: tensor<512xf32>, %arg17: tensor<512xf32>, %arg18: tensor<128x28x28x512xf32>, %arg19: tensor<512xf32>) -> tensor<128x28x28x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x28x28x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x28x28x512xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x28x28x512xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x28x28x512xf32>
    %16 = stablehlo.broadcast_in_dim %arg19, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %17 = stablehlo.subtract %arg18, %16 : tensor<128x28x28x512xf32>
    %18 = stablehlo.broadcast_in_dim %arg17, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x28x28x512xf32>
    %20 = stablehlo.broadcast_in_dim %arg16, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x28x28x512xf32>
    %22 = stablehlo.broadcast_in_dim %arg15, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x28x28x512xf32>
    %24 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %25 = stablehlo.subtract %arg13, %24 : tensor<128x28x28x512xf32>
    %26 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x28x28x512xf32>
    %28 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x28x28x512xf32>
    %30 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x28x28x512xf32>
    %32 = stablehlo.add %23, %31 : tensor<128x28x28x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x512xf32>
    %34 = stablehlo.maximum %32, %33 : tensor<128x28x28x512xf32>
    %35 = stablehlo.add %15, %34 : tensor<128x28x28x512xf32>
    %36 = stablehlo.maximum %35, %33 : tensor<128x28x28x512xf32>
    %37 = stablehlo.add %7, %36 : tensor<128x28x28x512xf32>
    %38 = stablehlo.maximum %37, %33 : tensor<128x28x28x512xf32>
    return %38 : tensor<128x28x28x512xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x512xf32>
    return %1 : tensor<1x1x128x512xf32>
  }
  func.func private @fused_computation.100(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<128x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x28x28x128xf32>
    return %9 : tensor<128x28x28x128xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x128x128xf32>
    return %1 : tensor<3x3x128x128xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<128x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x28x28x128xf32>
    return %9 : tensor<128x28x28x128xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,128]{1,0,2,3}"} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x128xf32>
    return %1 : tensor<1x1x512x128xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x28x28x512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512xf32>, %arg8: tensor<128x28x28x512xf32>, %arg9: tensor<512xf32>, %arg10: tensor<512xf32>, %arg11: tensor<512xf32>, %arg12: tensor<512xf32>, %arg13: tensor<128x28x28x512xf32>, %arg14: tensor<512xf32>) -> tensor<128x28x28x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %9 = stablehlo.subtract %arg13, %8 : tensor<128x28x28x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x28x28x512xf32>
    %12 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x28x28x512xf32>
    %14 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x28x28x512xf32>
    %16 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %17 = stablehlo.subtract %arg8, %16 : tensor<128x28x28x512xf32>
    %18 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x28x28x512xf32>
    %20 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x28x28x512xf32>
    %22 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x28x28x512xf32>
    %24 = stablehlo.add %15, %23 : tensor<128x28x28x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x512xf32>
    %26 = stablehlo.maximum %24, %25 : tensor<128x28x28x512xf32>
    %27 = stablehlo.add %7, %26 : tensor<128x28x28x512xf32>
    %28 = stablehlo.maximum %27, %25 : tensor<128x28x28x512xf32>
    return %28 : tensor<128x28x28x512xf32>
  }
  func.func private @fused_computation.107(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x512xf32>
    return %1 : tensor<1x1x128x512xf32>
  }
  func.func private @fused_computation.109(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<128x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x28x28x128xf32>
    return %9 : tensor<128x28x28x128xf32>
  }
  func.func private @fused_computation.110(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x128x128xf32>
    return %1 : tensor<3x3x128x128xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<128x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x28x28x128xf32>
    return %9 : tensor<128x28x28x128xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,128]{1,0,2,3}"} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x128xf32>
    return %1 : tensor<1x1x512x128xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<128x28x28x512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512xf32>, %arg8: tensor<128x28x28x512xf32>, %arg9: tensor<512xf32>) -> tensor<128x28x28x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<128x28x28x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<128x28x28x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x28x28x512xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x28x28x512xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x28x28x512xf32>
    %16 = stablehlo.add %7, %15 : tensor<128x28x28x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x512xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<128x28x28x512xf32>
    return %18 : tensor<128x28x28x512xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.117(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.119(%arg0: tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x512xf32>
    return %1 : tensor<1x1x128x512xf32>
  }
  func.func private @fused_computation.120(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<128x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x28x28x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x28x28x128xf32>
    return %9 : tensor<128x28x28x128xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x128x128xf32>
    return %1 : tensor<3x3x128x128xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128x56x56x128xf32>, %arg4: tensor<128xf32>) -> tensor<128x56x56x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<128x56x56x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<128x56x56x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<128x56x56x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<128x56x56x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x56x56x128xf32>
    return %9 : tensor<128x56x56x128xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x56x56x256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<128x56x56x256xf32>, %arg9: tensor<256xf32>, %arg10: tensor<256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<256xf32>, %arg13: tensor<128x56x56x256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<256xf32>, %arg17: tensor<256xf32>, %arg18: tensor<128x56x56x256xf32>, %arg19: tensor<256xf32>) -> tensor<128x56x56x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %9 = stablehlo.subtract %arg8, %8 : tensor<128x56x56x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x56x56x256xf32>
    %12 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x56x56x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x56x56x256xf32>
    %16 = stablehlo.broadcast_in_dim %arg19, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %17 = stablehlo.subtract %arg18, %16 : tensor<128x56x56x256xf32>
    %18 = stablehlo.broadcast_in_dim %arg17, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x56x56x256xf32>
    %20 = stablehlo.broadcast_in_dim %arg16, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x56x56x256xf32>
    %22 = stablehlo.broadcast_in_dim %arg15, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x56x56x256xf32>
    %24 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %25 = stablehlo.subtract %arg13, %24 : tensor<128x56x56x256xf32>
    %26 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x56x56x256xf32>
    %28 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x56x56x256xf32>
    %30 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x56x56x256xf32>
    %32 = stablehlo.add %23, %31 : tensor<128x56x56x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x256xf32>
    %34 = stablehlo.maximum %32, %33 : tensor<128x56x56x256xf32>
    %35 = stablehlo.add %15, %34 : tensor<128x56x56x256xf32>
    %36 = stablehlo.maximum %35, %33 : tensor<128x56x56x256xf32>
    %37 = stablehlo.add %7, %36 : tensor<128x56x56x256xf32>
    %38 = stablehlo.maximum %37, %33 : tensor<128x56x56x256xf32>
    return %38 : tensor<128x56x56x256xf32>
  }
  func.func private @fused_computation.127(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x256xf32>
    return %1 : tensor<1x1x64x256xf32>
  }
  func.func private @fused_computation.129(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<128x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x56x56x64xf32>
    return %9 : tensor<128x56x56x64xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
    %1 = mhlo.copy %0 : tensor<3x3x64x64xf32>
    return %1 : tensor<3x3x64x64xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<128x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x56x56x64xf32>
    return %9 : tensor<128x56x56x64xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,64]{1,0,2,3}"} : (tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x64xf32>
    return %1 : tensor<1x1x256x64xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x56x56x256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<128x56x56x256xf32>, %arg9: tensor<256xf32>, %arg10: tensor<256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<256xf32>, %arg13: tensor<128x56x56x256xf32>, %arg14: tensor<256xf32>) -> tensor<128x56x56x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg14, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %9 = stablehlo.subtract %arg13, %8 : tensor<128x56x56x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg12, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x56x56x256xf32>
    %12 = stablehlo.broadcast_in_dim %arg11, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x56x56x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg10, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x56x56x256xf32>
    %16 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %17 = stablehlo.subtract %arg8, %16 : tensor<128x56x56x256xf32>
    %18 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x56x56x256xf32>
    %20 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x56x56x256xf32>
    %22 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x56x56x256xf32>
    %24 = stablehlo.add %15, %23 : tensor<128x56x56x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x256xf32>
    %26 = stablehlo.maximum %24, %25 : tensor<128x56x56x256xf32>
    %27 = stablehlo.add %7, %26 : tensor<128x56x56x256xf32>
    %28 = stablehlo.maximum %27, %25 : tensor<128x56x56x256xf32>
    return %28 : tensor<128x56x56x256xf32>
  }
  func.func private @fused_computation.136(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x256xf32>
    return %1 : tensor<1x1x64x256xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<128x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x56x56x64xf32>
    return %9 : tensor<128x56x56x64xf32>
  }
  func.func private @fused_computation.139(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.140(%arg0: tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
    %1 = mhlo.copy %0 : tensor<3x3x64x64xf32>
    return %1 : tensor<3x3x64x64xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<128x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x56x56x64xf32>
    return %9 : tensor<128x56x56x64xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,64]{1,0,2,3}"} : (tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x64xf32>
    return %1 : tensor<1x1x256x64xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x56x56x256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<128x56x56x256xf32>, %arg9: tensor<256xf32>) -> tensor<128x56x56x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<128x56x56x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<128x56x56x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x56x56x256xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<128x56x56x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x56x56x256xf32>
    %16 = stablehlo.add %7, %15 : tensor<128x56x56x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x256xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<128x56x56x256xf32>
    return %18 : tensor<128x56x56x256xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x256xf32>
    return %1 : tensor<1x1x64x256xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<128x64x56x56xf32>) -> tensor<128x56x56x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,56,56,64]{2,1,3,0}"} : (tensor<128x64x56x56xf32>) -> tensor<128x56x56x64xf32>
    %1 = mhlo.copy %0 : tensor<128x56x56x64xf32>
    return %1 : tensor<128x56x56x64xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.149(%arg0: tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x256xf32>
    return %1 : tensor<1x1x64x256xf32>
  }
  func.func private @fused_computation.150(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<128x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x56x56x64xf32>
    return %9 : tensor<128x56x56x64xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.152(%arg0: tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
    %1 = mhlo.copy %0 : tensor<3x3x64x64xf32>
    return %1 : tensor<3x3x64x64xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<128x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x56x56x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x56x56x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x56x56x64xf32>
    return %9 : tensor<128x56x56x64xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x64xf32>
    return %1 : tensor<1x1x64x64xf32>
  }
  func.func private @fused_computation.157(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<128x112x112x64xf32>, %arg4: tensor<64xf32>) -> tensor<128x64x112x112xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<128x112x112x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x112x112x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<128x112x112x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x112x112x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<128x112x112x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x112x112x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<128x112x112x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x112x112x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x112x112x64xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x112x112x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,64,112,112]{1,3,2,0}"} : (tensor<128x112x112x64xf32>) -> tensor<128x64x112x112xf32>
    %11 = mhlo.copy %10 : tensor<128x64x112x112xf32>
    return %11 : tensor<128x64x112x112xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<64x3x7x7xf32>) -> tensor<7x7x3x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,3,64]{1,0,2,3}"} : (tensor<64x3x7x7xf32>) -> tensor<7x7x3x64xf32>
    %1 = mhlo.copy %0 : tensor<7x7x3x64xf32>
    return %1 : tensor<7x7x3x64xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<128x3x224x224xf32>) -> tensor<128x224x224x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,224,224,3]{2,1,3,0}"} : (tensor<128x3x224x224xf32>) -> tensor<128x224x224x3xf32>
    %1 = mhlo.copy %0 : tensor<128x224x224x3xf32>
    return %1 : tensor<128x224x224x3xf32>
  }
  func.func private @region_0.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<128x64x112x112xf32>, %arg1: tensor<f32>) -> tensor<128x64x56x56xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x56x56xf32>
    return %0 : tensor<128x64x56x56xf32>
  }
  func.func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64x1x1xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<64x64x1x1xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x3x3xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<256x64x1x1xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<64x256x1x1xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64x64x3x3xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<256x64x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<64x256x1x1xf32>, %arg25: tensor<64xf32>, %arg26: tensor<64xf32>, %arg27: tensor<64x64x3x3xf32>, %arg28: tensor<64xf32>, %arg29: tensor<64xf32>, %arg30: tensor<256x64x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<512x256x1x1xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<128x256x1x1xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128x128x3x3xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<512x128x1x1xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<128x512x1x1xf32>, %arg46: tensor<128xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128x128x3x3xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<512x128x1x1xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512xf32>, %arg54: tensor<128x512x1x1xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128x128x3x3xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<512x128x1x1xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512xf32>, %arg63: tensor<128x512x1x1xf32>, %arg64: tensor<128xf32>, %arg65: tensor<128xf32>, %arg66: tensor<128x128x3x3xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<512x128x1x1xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<1024x512x1x1xf32>, %arg73: tensor<1024xf32>, %arg74: tensor<1024xf32>, %arg75: tensor<256x512x1x1xf32>, %arg76: tensor<256xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x256x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<256xf32>, %arg81: tensor<1024x256x1x1xf32>, %arg82: tensor<1024xf32>, %arg83: tensor<1024xf32>, %arg84: tensor<256x1024x1x1xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256x256x3x3xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<1024x256x1x1xf32>, %arg91: tensor<1024xf32>, %arg92: tensor<1024xf32>, %arg93: tensor<256x1024x1x1xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256x256x3x3xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<1024x256x1x1xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<1024xf32>, %arg102: tensor<256x1024x1x1xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256x256x3x3xf32>, %arg106: tensor<256xf32>, %arg107: tensor<256xf32>, %arg108: tensor<1024x256x1x1xf32>, %arg109: tensor<1024xf32>, %arg110: tensor<1024xf32>, %arg111: tensor<256x1024x1x1xf32>, %arg112: tensor<256xf32>, %arg113: tensor<256xf32>, %arg114: tensor<256x256x3x3xf32>, %arg115: tensor<256xf32>, %arg116: tensor<256xf32>, %arg117: tensor<1024x256x1x1xf32>, %arg118: tensor<1024xf32>, %arg119: tensor<1024xf32>, %arg120: tensor<256x1024x1x1xf32>, %arg121: tensor<256xf32>, %arg122: tensor<256xf32>, %arg123: tensor<256x256x3x3xf32>, %arg124: tensor<256xf32>, %arg125: tensor<256xf32>, %arg126: tensor<1024x256x1x1xf32>, %arg127: tensor<1024xf32>, %arg128: tensor<1024xf32>, %arg129: tensor<2048x1024x1x1xf32>, %arg130: tensor<2048xf32>, %arg131: tensor<2048xf32>, %arg132: tensor<512x1024x1x1xf32>, %arg133: tensor<512xf32>, %arg134: tensor<512xf32>, %arg135: tensor<512x512x3x3xf32>, %arg136: tensor<512xf32>, %arg137: tensor<512xf32>, %arg138: tensor<2048x512x1x1xf32>, %arg139: tensor<2048xf32>, %arg140: tensor<2048xf32>, %arg141: tensor<512x2048x1x1xf32>, %arg142: tensor<512xf32>, %arg143: tensor<512xf32>, %arg144: tensor<512x512x3x3xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<2048x512x1x1xf32>, %arg148: tensor<2048xf32>, %arg149: tensor<2048xf32>, %arg150: tensor<512x2048x1x1xf32>, %arg151: tensor<512xf32>, %arg152: tensor<512xf32>, %arg153: tensor<512x512x3x3xf32>, %arg154: tensor<512xf32>, %arg155: tensor<512xf32>, %arg156: tensor<2048x512x1x1xf32>, %arg157: tensor<2048xf32>, %arg158: tensor<2048xf32>, %arg159: tensor<1000x2048xf32>, %arg160: tensor<1000xf32>, %arg161: tensor<64xf32>, %arg162: tensor<64xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<64xf32>, %arg166: tensor<64xf32>, %arg167: tensor<64xf32>, %arg168: tensor<64xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<64xf32>, %arg174: tensor<64xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<64xf32>, %arg178: tensor<64xf32>, %arg179: tensor<64xf32>, %arg180: tensor<64xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512xf32>, %arg184: tensor<512xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<128xf32>, %arg188: tensor<128xf32>, %arg189: tensor<512xf32>, %arg190: tensor<512xf32>, %arg191: tensor<128xf32>, %arg192: tensor<128xf32>, %arg193: tensor<128xf32>, %arg194: tensor<128xf32>, %arg195: tensor<512xf32>, %arg196: tensor<512xf32>, %arg197: tensor<128xf32>, %arg198: tensor<128xf32>, %arg199: tensor<128xf32>, %arg200: tensor<128xf32>, %arg201: tensor<512xf32>, %arg202: tensor<512xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<512xf32>, %arg208: tensor<512xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1024xf32>, %arg211: tensor<256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<1024xf32>, %arg216: tensor<1024xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<1024xf32>, %arg222: tensor<1024xf32>, %arg223: tensor<256xf32>, %arg224: tensor<256xf32>, %arg225: tensor<256xf32>, %arg226: tensor<256xf32>, %arg227: tensor<1024xf32>, %arg228: tensor<1024xf32>, %arg229: tensor<256xf32>, %arg230: tensor<256xf32>, %arg231: tensor<256xf32>, %arg232: tensor<256xf32>, %arg233: tensor<1024xf32>, %arg234: tensor<1024xf32>, %arg235: tensor<256xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<256xf32>, %arg239: tensor<1024xf32>, %arg240: tensor<1024xf32>, %arg241: tensor<256xf32>, %arg242: tensor<256xf32>, %arg243: tensor<256xf32>, %arg244: tensor<256xf32>, %arg245: tensor<1024xf32>, %arg246: tensor<1024xf32>, %arg247: tensor<2048xf32>, %arg248: tensor<2048xf32>, %arg249: tensor<512xf32>, %arg250: tensor<512xf32>, %arg251: tensor<512xf32>, %arg252: tensor<512xf32>, %arg253: tensor<2048xf32>, %arg254: tensor<2048xf32>, %arg255: tensor<512xf32>, %arg256: tensor<512xf32>, %arg257: tensor<512xf32>, %arg258: tensor<512xf32>, %arg259: tensor<2048xf32>, %arg260: tensor<2048xf32>, %arg261: tensor<512xf32>, %arg262: tensor<512xf32>, %arg263: tensor<512xf32>, %arg264: tensor<512xf32>, %arg265: tensor<2048xf32>, %arg266: tensor<2048xf32>, %arg267: tensor<128x3x224x224xf32>) -> tensor<128x1000xf32> {
    %0 = "mhlo.fusion"(%arg266) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<2048xf32>
      %315 = stablehlo.rsqrt %314 : tensor<2048xf32>
      mhlo.return %315 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %1 = "mhlo.fusion"(%arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %2 = "mhlo.fusion"(%arg262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %3 = "mhlo.fusion"(%arg260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<2048xf32>
      %315 = stablehlo.rsqrt %314 : tensor<2048xf32>
      mhlo.return %315 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %4 = "mhlo.fusion"(%arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %5 = "mhlo.fusion"(%arg256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %6 = "mhlo.fusion"(%arg248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<2048xf32>
      %315 = stablehlo.rsqrt %314 : tensor<2048xf32>
      mhlo.return %315 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %7 = "mhlo.fusion"(%arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<1024xf32>
      mhlo.return %315 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %8 = "mhlo.fusion"(%arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %9 = "mhlo.fusion"(%arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %10 = "mhlo.fusion"(%arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<1024xf32>
      mhlo.return %315 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %11 = "mhlo.fusion"(%arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %12 = "mhlo.fusion"(%arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %13 = "mhlo.fusion"(%arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<1024xf32>
      mhlo.return %315 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %14 = "mhlo.fusion"(%arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %15 = "mhlo.fusion"(%arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %16 = "mhlo.fusion"(%arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<1024xf32>
      mhlo.return %315 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %17 = "mhlo.fusion"(%arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %18 = "mhlo.fusion"(%arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %19 = "mhlo.fusion"(%arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<1024xf32>
      mhlo.return %315 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %20 = "mhlo.fusion"(%arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %21 = "mhlo.fusion"(%arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %22 = "mhlo.fusion"(%arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<1024xf32>
      mhlo.return %315 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %23 = "mhlo.fusion"(%arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %24 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %25 = "mhlo.fusion"(%arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %26 = "mhlo.fusion"(%arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %27 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %28 = "mhlo.fusion"(%arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %29 = "mhlo.fusion"(%arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %30 = "mhlo.fusion"(%arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %31 = "mhlo.fusion"(%arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %32 = "mhlo.fusion"(%arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %33 = "mhlo.fusion"(%arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %34 = "mhlo.fusion"(%arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %35 = "mhlo.fusion"(%arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %36 = "mhlo.fusion"(%arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %37 = "mhlo.fusion"(%arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %38 = "mhlo.fusion"(%arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %39 = "mhlo.fusion"(%arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %40 = "mhlo.fusion"(%arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %41 = "mhlo.fusion"(%arg267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x3x224x224xf32>):
      %313 = stablehlo.transpose %arg268, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,224,224,3]{2,1,3,0}"} : (tensor<128x3x224x224xf32>) -> tensor<128x224x224x3xf32>
      %314 = mhlo.copy %313 : tensor<128x224x224x3xf32>
      mhlo.return %314 : tensor<128x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<128x3x224x224xf32>) -> tensor<128x224x224x3xf32>
    %42 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x3x7x7xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,3,64]{1,0,2,3}"} : (tensor<64x3x7x7xf32>) -> tensor<7x7x3x64xf32>
      %314 = mhlo.copy %313 : tensor<7x7x3x64xf32>
      mhlo.return %314 : tensor<7x7x3x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x3x7x7xf32>) -> tensor<7x7x3x64xf32>
    %43 = stablehlo.convolution(%41, %42) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<128x112x112x64xf32>
    %44 = "mhlo.fusion"(%arg2, %arg1, %40, %43, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x112x112x64xf32>, %arg272: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<64xf32>) -> tensor<128x112x112x64xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x112x112x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<64xf32>) -> tensor<128x112x112x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x112x112x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<64xf32>) -> tensor<128x112x112x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x112x112x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<64xf32>) -> tensor<128x112x112x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x112x112x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x112x112x64xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x112x112x64xf32>
      %323 = stablehlo.transpose %322, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,64,112,112]{1,3,2,0}"} : (tensor<128x112x112x64xf32>) -> tensor<128x64x112x112xf32>
      %324 = mhlo.copy %323 : tensor<128x64x112x112xf32>
      mhlo.return %324 : tensor<128x64x112x112xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x112x112x64xf32>, tensor<64xf32>) -> tensor<128x64x112x112xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %45 = "mhlo.fusion"(%44, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x64x112x112xf32>, %arg269: tensor<f32>):
      %313 = "stablehlo.reduce_window"(%arg268, %arg269) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
      ^bb0(%arg270: tensor<f32>, %arg271: tensor<f32>):
        %314 = stablehlo.maximum %arg270, %arg271 : tensor<f32>
        stablehlo.return %314 : tensor<f32>
      }) : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x56x56xf32>
      mhlo.return %313 : tensor<128x64x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x112x112xf32>, tensor<f32>) -> tensor<128x64x56x56xf32>
    %46 = "mhlo.fusion"(%45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x64x56x56xf32>):
      %313 = stablehlo.transpose %arg268, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,56,56,64]{2,1,3,0}"} : (tensor<128x64x56x56xf32>) -> tensor<128x56x56x64xf32>
      %314 = mhlo.copy %313 : tensor<128x56x56x64xf32>
      mhlo.return %314 : tensor<128x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x56x56xf32>) -> tensor<128x56x56x64xf32>
    %47 = mhlo.bitcast %46 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x56x56x64xf32>) -> tensor<401408x64xf32>
    %48 = "mhlo.fusion"(%arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x64x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x64x256xf32>
      mhlo.return %314 : tensor<1x1x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %49 = mhlo.bitcast %48 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %50 = stablehlo.dot %47, %49, precision = [DEFAULT, DEFAULT] : (tensor<401408x64xf32>, tensor<64x256xf32>) -> tensor<401408x256xf32>
    %51 = mhlo.bitcast %50 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x256xf32>) -> tensor<128x56x56x256xf32>
    %52 = "mhlo.fusion"(%arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %53 = "mhlo.fusion"(%arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %54 = "mhlo.fusion"(%arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %55 = "mhlo.fusion"(%arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x64x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
      %314 = mhlo.copy %313 : tensor<1x1x64x64xf32>
      mhlo.return %314 : tensor<1x1x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
    %56 = mhlo.bitcast %55 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x64xf32>) -> tensor<64x64xf32>
    %57 = stablehlo.dot %47, %56, precision = [DEFAULT, DEFAULT] : (tensor<401408x64xf32>, tensor<64x64xf32>) -> tensor<401408x64xf32>
    %58 = mhlo.bitcast %57 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x64xf32>) -> tensor<128x56x56x64xf32>
    %59 = "mhlo.fusion"(%arg8, %arg7, %54, %58, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x56x56x64xf32>, %arg272: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x56x56x64xf32>
      mhlo.return %322 : tensor<128x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x56x56x64xf32>, tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %60 = "mhlo.fusion"(%arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x64x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
      %314 = mhlo.copy %313 : tensor<3x3x64x64xf32>
      mhlo.return %314 : tensor<3x3x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
    %61 = stablehlo.convolution(%59, %60) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<128x56x56x64xf32>
    %62 = "mhlo.fusion"(%arg11, %arg10, %53, %61, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x56x56x64xf32>, %arg272: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x56x56x64xf32>
      mhlo.return %322 : tensor<128x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x56x56x64xf32>, tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x56x56x64xf32>) -> tensor<401408x64xf32>
    %64 = "mhlo.fusion"(%arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x64x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x64x256xf32>
      mhlo.return %314 : tensor<1x1x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %66 = stablehlo.dot %63, %65, precision = [DEFAULT, DEFAULT] : (tensor<401408x64xf32>, tensor<64x256xf32>) -> tensor<401408x256xf32>
    %67 = mhlo.bitcast %66 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x256xf32>) -> tensor<128x56x56x256xf32>
    %68 = "mhlo.fusion"(%arg5, %arg4, %39, %51, %arg163, %arg14, %arg13, %52, %67, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x56x56x256xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<256xf32>, %arg276: tensor<128x56x56x256xf32>, %arg277: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %314 = stablehlo.subtract %arg276, %313 : tensor<128x56x56x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x256xf32>
      %321 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %322 = stablehlo.subtract %arg271, %321 : tensor<128x56x56x256xf32>
      %323 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x56x56x256xf32>
      %325 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x56x56x256xf32>
      %327 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x56x56x256xf32>
      %329 = stablehlo.add %320, %328 : tensor<128x56x56x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x256xf32>
      %331 = stablehlo.maximum %329, %330 : tensor<128x56x56x256xf32>
      mhlo.return %331 : tensor<128x56x56x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %69 = mhlo.bitcast %68 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x56x56x256xf32>) -> tensor<401408x256xf32>
    %70 = "mhlo.fusion"(%arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,64]{1,0,2,3}"} : (tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x64xf32>
      mhlo.return %314 : tensor<1x1x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32>
    %71 = mhlo.bitcast %70 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x64xf32>) -> tensor<256x64xf32>
    %72 = stablehlo.dot %69, %71, precision = [DEFAULT, DEFAULT] : (tensor<401408x256xf32>, tensor<256x64xf32>) -> tensor<401408x64xf32>
    %73 = mhlo.bitcast %72 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x64xf32>) -> tensor<128x56x56x64xf32>
    %74 = "mhlo.fusion"(%arg17, %arg16, %38, %73, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x56x56x64xf32>, %arg272: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x56x56x64xf32>
      mhlo.return %322 : tensor<128x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x56x56x64xf32>, tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %75 = "mhlo.fusion"(%arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x64x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
      %314 = mhlo.copy %313 : tensor<3x3x64x64xf32>
      mhlo.return %314 : tensor<3x3x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
    %76 = stablehlo.convolution(%74, %75) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<128x56x56x64xf32>
    %77 = "mhlo.fusion"(%arg20, %arg19, %37, %76, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x56x56x64xf32>, %arg272: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x56x56x64xf32>
      mhlo.return %322 : tensor<128x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x56x56x64xf32>, tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %78 = mhlo.bitcast %77 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x56x56x64xf32>) -> tensor<401408x64xf32>
    %79 = "mhlo.fusion"(%arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x64x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x64x256xf32>
      mhlo.return %314 : tensor<1x1x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %81 = stablehlo.dot %78, %80, precision = [DEFAULT, DEFAULT] : (tensor<401408x64xf32>, tensor<64x256xf32>) -> tensor<401408x256xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x256xf32>) -> tensor<128x56x56x256xf32>
    %83 = "mhlo.fusion"(%arg23, %arg22, %36, %82, %arg175, %arg5, %arg4, %39, %51, %arg163, %arg14, %arg13, %52, %67, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x56x56x256xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<256xf32>, %arg276: tensor<128x56x56x256xf32>, %arg277: tensor<256xf32>, %arg278: tensor<256xf32>, %arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<128x56x56x256xf32>, %arg282: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x256xf32>
      %321 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %322 = stablehlo.subtract %arg281, %321 : tensor<128x56x56x256xf32>
      %323 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x56x56x256xf32>
      %325 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x56x56x256xf32>
      %327 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x56x56x256xf32>
      %329 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %330 = stablehlo.subtract %arg276, %329 : tensor<128x56x56x256xf32>
      %331 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x56x56x256xf32>
      %333 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x56x56x256xf32>
      %335 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x56x56x256xf32>
      %337 = stablehlo.add %328, %336 : tensor<128x56x56x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x256xf32>
      %339 = stablehlo.maximum %337, %338 : tensor<128x56x56x256xf32>
      %340 = stablehlo.add %320, %339 : tensor<128x56x56x256xf32>
      %341 = stablehlo.maximum %340, %338 : tensor<128x56x56x256xf32>
      mhlo.return %341 : tensor<128x56x56x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %84 = mhlo.bitcast %83 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x56x56x256xf32>) -> tensor<401408x256xf32>
    %85 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,64]{1,0,2,3}"} : (tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x64xf32>
      mhlo.return %314 : tensor<1x1x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x1x1xf32>) -> tensor<1x1x256x64xf32>
    %86 = mhlo.bitcast %85 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x64xf32>) -> tensor<256x64xf32>
    %87 = stablehlo.dot %84, %86, precision = [DEFAULT, DEFAULT] : (tensor<401408x256xf32>, tensor<256x64xf32>) -> tensor<401408x64xf32>
    %88 = mhlo.bitcast %87 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x64xf32>) -> tensor<128x56x56x64xf32>
    %89 = "mhlo.fusion"(%arg26, %arg25, %35, %88, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x56x56x64xf32>, %arg272: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x56x56x64xf32>
      mhlo.return %322 : tensor<128x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x56x56x64xf32>, tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %90 = "mhlo.fusion"(%arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64x64x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,64]{1,0,2,3}"} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
      %314 = mhlo.copy %313 : tensor<3x3x64x64xf32>
      mhlo.return %314 : tensor<3x3x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x3x3xf32>) -> tensor<3x3x64x64xf32>
    %91 = stablehlo.convolution(%89, %90) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<128x56x56x64xf32>
    %92 = "mhlo.fusion"(%arg29, %arg28, %34, %91, %arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<64xf32>, %arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<128x56x56x64xf32>, %arg272: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<64xf32>) -> tensor<128x56x56x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x64xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x56x56x64xf32>
      mhlo.return %322 : tensor<128x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x56x56x64xf32>, tensor<64xf32>) -> tensor<128x56x56x64xf32>
    %93 = mhlo.bitcast %92 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x56x56x64xf32>) -> tensor<401408x64xf32>
    %94 = "mhlo.fusion"(%arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x64x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,256]{1,0,2,3}"} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x64x256xf32>
      mhlo.return %314 : tensor<1x1x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64x1x1xf32>) -> tensor<1x1x64x256xf32>
    %95 = mhlo.bitcast %94 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %96 = stablehlo.dot %93, %95, precision = [DEFAULT, DEFAULT] : (tensor<401408x64xf32>, tensor<64x256xf32>) -> tensor<401408x256xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x256xf32>) -> tensor<128x56x56x256xf32>
    %98 = "mhlo.fusion"(%arg32, %arg31, %33, %97, %arg181, %arg23, %arg22, %36, %82, %arg175, %arg5, %arg4, %39, %51, %arg163, %arg14, %arg13, %52, %67, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x56x56x256xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<256xf32>, %arg276: tensor<128x56x56x256xf32>, %arg277: tensor<256xf32>, %arg278: tensor<256xf32>, %arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<128x56x56x256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>, %arg285: tensor<256xf32>, %arg286: tensor<128x56x56x256xf32>, %arg287: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x256xf32>
      %321 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %322 = stablehlo.subtract %arg276, %321 : tensor<128x56x56x256xf32>
      %323 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x56x56x256xf32>
      %325 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x56x56x256xf32>
      %327 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x56x56x256xf32>
      %329 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %330 = stablehlo.subtract %arg286, %329 : tensor<128x56x56x256xf32>
      %331 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x56x56x256xf32>
      %333 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x56x56x256xf32>
      %335 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x56x56x256xf32>
      %337 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %338 = stablehlo.subtract %arg281, %337 : tensor<128x56x56x256xf32>
      %339 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<128x56x56x256xf32>
      %341 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<128x56x56x256xf32>
      %343 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<256xf32>) -> tensor<128x56x56x256xf32>
      %344 = stablehlo.add %342, %343 : tensor<128x56x56x256xf32>
      %345 = stablehlo.add %336, %344 : tensor<128x56x56x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %346 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x256xf32>
      %347 = stablehlo.maximum %345, %346 : tensor<128x56x56x256xf32>
      %348 = stablehlo.add %328, %347 : tensor<128x56x56x256xf32>
      %349 = stablehlo.maximum %348, %346 : tensor<128x56x56x256xf32>
      %350 = stablehlo.add %320, %349 : tensor<128x56x56x256xf32>
      %351 = stablehlo.maximum %350, %346 : tensor<128x56x56x256xf32>
      mhlo.return %351 : tensor<128x56x56x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x56x56x256xf32>, tensor<256xf32>) -> tensor<128x56x56x256xf32>
    %99 = "mhlo.fusion"(%arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x512xf32>
      mhlo.return %314 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %100 = stablehlo.convolution(%98, %99) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x56x56x256xf32>, tensor<1x1x256x512xf32>) -> tensor<128x28x28x512xf32>
    %101 = "mhlo.fusion"(%arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %102 = "mhlo.fusion"(%arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %103 = "mhlo.fusion"(%arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %104 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x56x56x256xf32>) -> tensor<401408x256xf32>
    %105 = "mhlo.fusion"(%arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x128xf32>
      mhlo.return %314 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %106 = mhlo.bitcast %105 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %107 = stablehlo.dot %104, %106, precision = [DEFAULT, DEFAULT] : (tensor<401408x256xf32>, tensor<256x128xf32>) -> tensor<401408x128xf32>
    %108 = mhlo.bitcast %107 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x128xf32>) -> tensor<128x56x56x128xf32>
    %109 = "mhlo.fusion"(%arg38, %arg37, %103, %108, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x56x56x128xf32>, %arg272: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<128x56x56x128xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x56x56x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<128x56x56x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x56x56x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<128x56x56x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x56x56x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<128x56x56x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x56x56x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x56x56x128xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x56x56x128xf32>
      mhlo.return %322 : tensor<128x56x56x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x56x56x128xf32>, tensor<128xf32>) -> tensor<128x56x56x128xf32>
    %110 = "mhlo.fusion"(%arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x128x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
      %314 = mhlo.copy %313 : tensor<3x3x128x128xf32>
      mhlo.return %314 : tensor<3x3x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %111 = stablehlo.convolution(%109, %110) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x56x56x128xf32>, tensor<3x3x128x128xf32>) -> tensor<128x28x28x128xf32>
    %112 = "mhlo.fusion"(%arg41, %arg40, %102, %111, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x28x28x128xf32>, %arg272: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x28x28x128xf32>
      mhlo.return %322 : tensor<128x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x28x28x128xf32>, tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %113 = mhlo.bitcast %112 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x28x28x128xf32>) -> tensor<100352x128xf32>
    %114 = "mhlo.fusion"(%arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x128x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
      %314 = mhlo.copy %313 : tensor<1x1x128x512xf32>
      mhlo.return %314 : tensor<1x1x128x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %116 = stablehlo.dot %113, %115, precision = [DEFAULT, DEFAULT] : (tensor<100352x128xf32>, tensor<128x512xf32>) -> tensor<100352x512xf32>
    %117 = mhlo.bitcast %116 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x512xf32>) -> tensor<128x28x28x512xf32>
    %118 = "mhlo.fusion"(%arg35, %arg34, %32, %100, %arg183, %arg44, %arg43, %101, %117, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x28x28x512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>, %arg274: tensor<512xf32>, %arg275: tensor<512xf32>, %arg276: tensor<128x28x28x512xf32>, %arg277: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %314 = stablehlo.subtract %arg276, %313 : tensor<128x28x28x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x512xf32>
      %321 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %322 = stablehlo.subtract %arg271, %321 : tensor<128x28x28x512xf32>
      %323 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x28x28x512xf32>
      %325 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x28x28x512xf32>
      %327 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x28x28x512xf32>
      %329 = stablehlo.add %320, %328 : tensor<128x28x28x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x512xf32>
      %331 = stablehlo.maximum %329, %330 : tensor<128x28x28x512xf32>
      mhlo.return %331 : tensor<128x28x28x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %119 = mhlo.bitcast %118 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x28x28x512xf32>) -> tensor<100352x512xf32>
    %120 = "mhlo.fusion"(%arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x512x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,128]{1,0,2,3}"} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
      %314 = mhlo.copy %313 : tensor<1x1x512x128xf32>
      mhlo.return %314 : tensor<1x1x512x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
    %121 = mhlo.bitcast %120 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %122 = stablehlo.dot %119, %121, precision = [DEFAULT, DEFAULT] : (tensor<100352x512xf32>, tensor<512x128xf32>) -> tensor<100352x128xf32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x128xf32>) -> tensor<128x28x28x128xf32>
    %124 = "mhlo.fusion"(%arg47, %arg46, %31, %123, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x28x28x128xf32>, %arg272: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x28x28x128xf32>
      mhlo.return %322 : tensor<128x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x28x28x128xf32>, tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %125 = "mhlo.fusion"(%arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x128x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
      %314 = mhlo.copy %313 : tensor<3x3x128x128xf32>
      mhlo.return %314 : tensor<3x3x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %126 = stablehlo.convolution(%124, %125) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<128x28x28x128xf32>
    %127 = "mhlo.fusion"(%arg50, %arg49, %30, %126, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x28x28x128xf32>, %arg272: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x28x28x128xf32>
      mhlo.return %322 : tensor<128x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x28x28x128xf32>, tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %128 = mhlo.bitcast %127 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x28x28x128xf32>) -> tensor<100352x128xf32>
    %129 = "mhlo.fusion"(%arg51) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x128x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
      %314 = mhlo.copy %313 : tensor<1x1x128x512xf32>
      mhlo.return %314 : tensor<1x1x128x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %131 = stablehlo.dot %128, %130, precision = [DEFAULT, DEFAULT] : (tensor<100352x128xf32>, tensor<128x512xf32>) -> tensor<100352x512xf32>
    %132 = mhlo.bitcast %131 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x512xf32>) -> tensor<128x28x28x512xf32>
    %133 = "mhlo.fusion"(%arg53, %arg52, %29, %132, %arg195, %arg35, %arg34, %32, %100, %arg183, %arg44, %arg43, %101, %117, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x28x28x512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>, %arg274: tensor<512xf32>, %arg275: tensor<512xf32>, %arg276: tensor<128x28x28x512xf32>, %arg277: tensor<512xf32>, %arg278: tensor<512xf32>, %arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<128x28x28x512xf32>, %arg282: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x512xf32>
      %321 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %322 = stablehlo.subtract %arg281, %321 : tensor<128x28x28x512xf32>
      %323 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x28x28x512xf32>
      %325 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x28x28x512xf32>
      %327 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x28x28x512xf32>
      %329 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %330 = stablehlo.subtract %arg276, %329 : tensor<128x28x28x512xf32>
      %331 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x28x28x512xf32>
      %333 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x28x28x512xf32>
      %335 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x28x28x512xf32>
      %337 = stablehlo.add %328, %336 : tensor<128x28x28x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x512xf32>
      %339 = stablehlo.maximum %337, %338 : tensor<128x28x28x512xf32>
      %340 = stablehlo.add %320, %339 : tensor<128x28x28x512xf32>
      %341 = stablehlo.maximum %340, %338 : tensor<128x28x28x512xf32>
      mhlo.return %341 : tensor<128x28x28x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %134 = mhlo.bitcast %133 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x28x28x512xf32>) -> tensor<100352x512xf32>
    %135 = "mhlo.fusion"(%arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x512x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,128]{1,0,2,3}"} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
      %314 = mhlo.copy %313 : tensor<1x1x512x128xf32>
      mhlo.return %314 : tensor<1x1x512x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %137 = stablehlo.dot %134, %136, precision = [DEFAULT, DEFAULT] : (tensor<100352x512xf32>, tensor<512x128xf32>) -> tensor<100352x128xf32>
    %138 = mhlo.bitcast %137 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x128xf32>) -> tensor<128x28x28x128xf32>
    %139 = "mhlo.fusion"(%arg56, %arg55, %28, %138, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x28x28x128xf32>, %arg272: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x28x28x128xf32>
      mhlo.return %322 : tensor<128x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x28x28x128xf32>, tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %140 = "mhlo.fusion"(%arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x128x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
      %314 = mhlo.copy %313 : tensor<3x3x128x128xf32>
      mhlo.return %314 : tensor<3x3x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %141 = stablehlo.convolution(%139, %140) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<128x28x28x128xf32>
    %142 = "mhlo.fusion"(%arg59, %arg58, %27, %141, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x28x28x128xf32>, %arg272: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x28x28x128xf32>
      mhlo.return %322 : tensor<128x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x28x28x128xf32>, tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %143 = mhlo.bitcast %142 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x28x28x128xf32>) -> tensor<100352x128xf32>
    %144 = "mhlo.fusion"(%arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x128x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
      %314 = mhlo.copy %313 : tensor<1x1x128x512xf32>
      mhlo.return %314 : tensor<1x1x128x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %145 = mhlo.bitcast %144 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %146 = stablehlo.dot %143, %145, precision = [DEFAULT, DEFAULT] : (tensor<100352x128xf32>, tensor<128x512xf32>) -> tensor<100352x512xf32>
    %147 = mhlo.bitcast %146 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x512xf32>) -> tensor<128x28x28x512xf32>
    %148 = "mhlo.fusion"(%arg62, %arg61, %26, %147, %arg201, %arg53, %arg52, %29, %132, %arg195, %arg35, %arg34, %32, %100, %arg183, %arg44, %arg43, %101, %117, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x28x28x512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>, %arg274: tensor<512xf32>, %arg275: tensor<512xf32>, %arg276: tensor<128x28x28x512xf32>, %arg277: tensor<512xf32>, %arg278: tensor<512xf32>, %arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<128x28x28x512xf32>, %arg282: tensor<512xf32>, %arg283: tensor<512xf32>, %arg284: tensor<512xf32>, %arg285: tensor<512xf32>, %arg286: tensor<128x28x28x512xf32>, %arg287: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x512xf32>
      %321 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %322 = stablehlo.subtract %arg276, %321 : tensor<128x28x28x512xf32>
      %323 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x28x28x512xf32>
      %325 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x28x28x512xf32>
      %327 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x28x28x512xf32>
      %329 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %330 = stablehlo.subtract %arg286, %329 : tensor<128x28x28x512xf32>
      %331 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x28x28x512xf32>
      %333 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x28x28x512xf32>
      %335 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x28x28x512xf32>
      %337 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %338 = stablehlo.subtract %arg281, %337 : tensor<128x28x28x512xf32>
      %339 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<128x28x28x512xf32>
      %341 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<128x28x28x512xf32>
      %343 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %344 = stablehlo.add %342, %343 : tensor<128x28x28x512xf32>
      %345 = stablehlo.add %336, %344 : tensor<128x28x28x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %346 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x512xf32>
      %347 = stablehlo.maximum %345, %346 : tensor<128x28x28x512xf32>
      %348 = stablehlo.add %328, %347 : tensor<128x28x28x512xf32>
      %349 = stablehlo.maximum %348, %346 : tensor<128x28x28x512xf32>
      %350 = stablehlo.add %320, %349 : tensor<128x28x28x512xf32>
      %351 = stablehlo.maximum %350, %346 : tensor<128x28x28x512xf32>
      mhlo.return %351 : tensor<128x28x28x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %149 = mhlo.bitcast %148 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x28x28x512xf32>) -> tensor<100352x512xf32>
    %150 = "mhlo.fusion"(%arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x512x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,128]{1,0,2,3}"} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
      %314 = mhlo.copy %313 : tensor<1x1x512x128xf32>
      mhlo.return %314 : tensor<1x1x512x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x512x1x1xf32>) -> tensor<1x1x512x128xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %152 = stablehlo.dot %149, %151, precision = [DEFAULT, DEFAULT] : (tensor<100352x512xf32>, tensor<512x128xf32>) -> tensor<100352x128xf32>
    %153 = mhlo.bitcast %152 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x128xf32>) -> tensor<128x28x28x128xf32>
    %154 = "mhlo.fusion"(%arg65, %arg64, %25, %153, %arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x28x28x128xf32>, %arg272: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x28x28x128xf32>
      mhlo.return %322 : tensor<128x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x28x28x128xf32>, tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %155 = "mhlo.fusion"(%arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x128x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,128,128]{1,0,2,3}"} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
      %314 = mhlo.copy %313 : tensor<3x3x128x128xf32>
      mhlo.return %314 : tensor<3x3x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x3x3xf32>) -> tensor<3x3x128x128xf32>
    %156 = stablehlo.convolution(%154, %155) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<128x28x28x128xf32>
    %157 = "mhlo.fusion"(%arg68, %arg67, %24, %156, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128x28x28x128xf32>, %arg272: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<128x28x28x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x128xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x28x28x128xf32>
      mhlo.return %322 : tensor<128x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x28x28x128xf32>, tensor<128xf32>) -> tensor<128x28x28x128xf32>
    %158 = mhlo.bitcast %157 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x28x28x128xf32>) -> tensor<100352x128xf32>
    %159 = "mhlo.fusion"(%arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x128x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,512]{1,0,2,3}"} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
      %314 = mhlo.copy %313 : tensor<1x1x128x512xf32>
      mhlo.return %314 : tensor<1x1x128x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x1x1xf32>) -> tensor<1x1x128x512xf32>
    %160 = mhlo.bitcast %159 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %161 = stablehlo.dot %158, %160, precision = [DEFAULT, DEFAULT] : (tensor<100352x128xf32>, tensor<128x512xf32>) -> tensor<100352x512xf32>
    %162 = mhlo.bitcast %161 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x512xf32>) -> tensor<128x28x28x512xf32>
    %163 = "mhlo.fusion"(%arg71, %arg70, %23, %162, %arg207, %arg62, %arg61, %26, %147, %arg201, %arg53, %arg52, %29, %132, %arg195, %arg35, %arg34, %32, %100, %arg183, %arg44, %arg43, %101, %117, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x28x28x512xf32>, %arg272: tensor<512xf32>, %arg273: tensor<512xf32>, %arg274: tensor<512xf32>, %arg275: tensor<512xf32>, %arg276: tensor<128x28x28x512xf32>, %arg277: tensor<512xf32>, %arg278: tensor<512xf32>, %arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<128x28x28x512xf32>, %arg282: tensor<512xf32>, %arg283: tensor<512xf32>, %arg284: tensor<512xf32>, %arg285: tensor<512xf32>, %arg286: tensor<128x28x28x512xf32>, %arg287: tensor<512xf32>, %arg288: tensor<512xf32>, %arg289: tensor<512xf32>, %arg290: tensor<512xf32>, %arg291: tensor<128x28x28x512xf32>, %arg292: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x512xf32>
      %321 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %322 = stablehlo.subtract %arg276, %321 : tensor<128x28x28x512xf32>
      %323 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x28x28x512xf32>
      %325 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x28x28x512xf32>
      %327 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x28x28x512xf32>
      %329 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %330 = stablehlo.subtract %arg281, %329 : tensor<128x28x28x512xf32>
      %331 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x28x28x512xf32>
      %333 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x28x28x512xf32>
      %335 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x28x28x512xf32>
      %337 = stablehlo.broadcast_in_dim %arg292, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %338 = stablehlo.subtract %arg291, %337 : tensor<128x28x28x512xf32>
      %339 = stablehlo.broadcast_in_dim %arg290, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<128x28x28x512xf32>
      %341 = stablehlo.broadcast_in_dim %arg289, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<128x28x28x512xf32>
      %343 = stablehlo.broadcast_in_dim %arg288, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %344 = stablehlo.add %342, %343 : tensor<128x28x28x512xf32>
      %345 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %346 = stablehlo.subtract %arg286, %345 : tensor<128x28x28x512xf32>
      %347 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<128x28x28x512xf32>
      %349 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<128x28x28x512xf32>
      %351 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<512xf32>) -> tensor<128x28x28x512xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x28x28x512xf32>
      %353 = stablehlo.add %344, %352 : tensor<128x28x28x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x512xf32>
      %355 = stablehlo.maximum %353, %354 : tensor<128x28x28x512xf32>
      %356 = stablehlo.add %336, %355 : tensor<128x28x28x512xf32>
      %357 = stablehlo.maximum %356, %354 : tensor<128x28x28x512xf32>
      %358 = stablehlo.add %328, %357 : tensor<128x28x28x512xf32>
      %359 = stablehlo.maximum %358, %354 : tensor<128x28x28x512xf32>
      %360 = stablehlo.add %320, %359 : tensor<128x28x28x512xf32>
      %361 = stablehlo.maximum %360, %354 : tensor<128x28x28x512xf32>
      mhlo.return %361 : tensor<128x28x28x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x28x28x512xf32>, tensor<512xf32>) -> tensor<128x28x28x512xf32>
    %164 = "mhlo.fusion"(%arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024x512x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,1024]{1,0,2,3}"} : (tensor<1024x512x1x1xf32>) -> tensor<1x1x512x1024xf32>
      %314 = mhlo.copy %313 : tensor<1x1x512x1024xf32>
      mhlo.return %314 : tensor<1x1x512x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x512x1x1xf32>) -> tensor<1x1x512x1024xf32>
    %165 = stablehlo.convolution(%163, %164) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x28x28x512xf32>, tensor<1x1x512x1024xf32>) -> tensor<128x14x14x1024xf32>
    %166 = "mhlo.fusion"(%arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<1024xf32>
      mhlo.return %315 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %167 = "mhlo.fusion"(%arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %168 = "mhlo.fusion"(%arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %169 = mhlo.bitcast %163 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x28x28x512xf32>) -> tensor<100352x512xf32>
    %170 = "mhlo.fusion"(%arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x512x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x512x256xf32>
      mhlo.return %314 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %171 = mhlo.bitcast %170 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %172 = stablehlo.dot %169, %171, precision = [DEFAULT, DEFAULT] : (tensor<100352x512xf32>, tensor<512x256xf32>) -> tensor<100352x256xf32>
    %173 = mhlo.bitcast %172 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x256xf32>) -> tensor<128x28x28x256xf32>
    %174 = "mhlo.fusion"(%arg77, %arg76, %168, %173, %arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x28x28x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x28x28x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x28x28x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x28x28x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x28x28x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x28x28x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x28x28x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x28x28x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x28x28x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x28x28x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x28x28x256xf32>
      mhlo.return %322 : tensor<128x28x28x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x28x28x256xf32>, tensor<256xf32>) -> tensor<128x28x28x256xf32>
    %175 = "mhlo.fusion"(%arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x256x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
      %314 = mhlo.copy %313 : tensor<3x3x256x256xf32>
      mhlo.return %314 : tensor<3x3x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %176 = stablehlo.convolution(%174, %175) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x28x28x256xf32>, tensor<3x3x256x256xf32>) -> tensor<128x14x14x256xf32>
    %177 = "mhlo.fusion"(%arg80, %arg79, %167, %176, %arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %178 = mhlo.bitcast %177 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x256xf32>) -> tensor<25088x256xf32>
    %179 = "mhlo.fusion"(%arg81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x1024xf32>
      mhlo.return %314 : tensor<1x1x256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %180 = mhlo.bitcast %179 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %181 = stablehlo.dot %178, %180, precision = [DEFAULT, DEFAULT] : (tensor<25088x256xf32>, tensor<256x1024xf32>) -> tensor<25088x1024xf32>
    %182 = mhlo.bitcast %181 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x1024xf32>) -> tensor<128x14x14x1024xf32>
    %183 = "mhlo.fusion"(%arg74, %arg73, %22, %165, %arg209, %arg83, %arg82, %166, %182, %arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<128x14x14x1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<128x14x14x1024xf32>, %arg277: tensor<1024xf32>):
      %313 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %314 = stablehlo.subtract %arg276, %313 : tensor<128x14x14x1024xf32>
      %315 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x1024xf32>
      %317 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x1024xf32>
      %319 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x1024xf32>
      %321 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %322 = stablehlo.subtract %arg271, %321 : tensor<128x14x14x1024xf32>
      %323 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x14x14x1024xf32>
      %325 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x14x14x1024xf32>
      %327 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x14x14x1024xf32>
      %329 = stablehlo.add %320, %328 : tensor<128x14x14x1024xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
      %331 = stablehlo.maximum %329, %330 : tensor<128x14x14x1024xf32>
      mhlo.return %331 : tensor<128x14x14x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x1024xf32>) -> tensor<25088x1024xf32>
    %185 = "mhlo.fusion"(%arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x1024x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x1024x256xf32>
      mhlo.return %314 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %187 = stablehlo.dot %184, %186, precision = [DEFAULT, DEFAULT] : (tensor<25088x1024xf32>, tensor<1024x256xf32>) -> tensor<25088x256xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x256xf32>) -> tensor<128x14x14x256xf32>
    %189 = "mhlo.fusion"(%arg86, %arg85, %21, %188, %arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %190 = "mhlo.fusion"(%arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x256x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
      %314 = mhlo.copy %313 : tensor<3x3x256x256xf32>
      mhlo.return %314 : tensor<3x3x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %191 = stablehlo.convolution(%189, %190) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<128x14x14x256xf32>
    %192 = "mhlo.fusion"(%arg89, %arg88, %20, %191, %arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %193 = mhlo.bitcast %192 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x256xf32>) -> tensor<25088x256xf32>
    %194 = "mhlo.fusion"(%arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x1024xf32>
      mhlo.return %314 : tensor<1x1x256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %195 = mhlo.bitcast %194 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %196 = stablehlo.dot %193, %195, precision = [DEFAULT, DEFAULT] : (tensor<25088x256xf32>, tensor<256x1024xf32>) -> tensor<25088x1024xf32>
    %197 = mhlo.bitcast %196 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x1024xf32>) -> tensor<128x14x14x1024xf32>
    %198 = "mhlo.fusion"(%arg92, %arg91, %19, %197, %arg221, %arg74, %arg73, %22, %165, %arg209, %arg83, %arg82, %166, %182, %arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<128x14x14x1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<128x14x14x1024xf32>, %arg277: tensor<1024xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<1024xf32>, %arg280: tensor<1024xf32>, %arg281: tensor<128x14x14x1024xf32>, %arg282: tensor<1024xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x1024xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x1024xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x1024xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x1024xf32>
      %321 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %322 = stablehlo.subtract %arg281, %321 : tensor<128x14x14x1024xf32>
      %323 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x14x14x1024xf32>
      %325 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x14x14x1024xf32>
      %327 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x14x14x1024xf32>
      %329 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %330 = stablehlo.subtract %arg276, %329 : tensor<128x14x14x1024xf32>
      %331 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x14x14x1024xf32>
      %333 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x14x14x1024xf32>
      %335 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x14x14x1024xf32>
      %337 = stablehlo.add %328, %336 : tensor<128x14x14x1024xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
      %339 = stablehlo.maximum %337, %338 : tensor<128x14x14x1024xf32>
      %340 = stablehlo.add %320, %339 : tensor<128x14x14x1024xf32>
      %341 = stablehlo.maximum %340, %338 : tensor<128x14x14x1024xf32>
      mhlo.return %341 : tensor<128x14x14x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x1024xf32>) -> tensor<25088x1024xf32>
    %200 = "mhlo.fusion"(%arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x1024x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x1024x256xf32>
      mhlo.return %314 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %202 = stablehlo.dot %199, %201, precision = [DEFAULT, DEFAULT] : (tensor<25088x1024xf32>, tensor<1024x256xf32>) -> tensor<25088x256xf32>
    %203 = mhlo.bitcast %202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x256xf32>) -> tensor<128x14x14x256xf32>
    %204 = "mhlo.fusion"(%arg95, %arg94, %18, %203, %arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %205 = "mhlo.fusion"(%arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x256x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
      %314 = mhlo.copy %313 : tensor<3x3x256x256xf32>
      mhlo.return %314 : tensor<3x3x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %206 = stablehlo.convolution(%204, %205) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<128x14x14x256xf32>
    %207 = "mhlo.fusion"(%arg98, %arg97, %17, %206, %arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %208 = mhlo.bitcast %207 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x256xf32>) -> tensor<25088x256xf32>
    %209 = "mhlo.fusion"(%arg99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x1024xf32>
      mhlo.return %314 : tensor<1x1x256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %210 = mhlo.bitcast %209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %211 = stablehlo.dot %208, %210, precision = [DEFAULT, DEFAULT] : (tensor<25088x256xf32>, tensor<256x1024xf32>) -> tensor<25088x1024xf32>
    %212 = mhlo.bitcast %211 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x1024xf32>) -> tensor<128x14x14x1024xf32>
    %213 = "mhlo.fusion"(%arg101, %arg100, %16, %212, %arg227, %arg92, %arg91, %19, %197, %arg221, %arg74, %arg73, %22, %165, %arg209, %arg83, %arg82, %166, %182, %arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<128x14x14x1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<128x14x14x1024xf32>, %arg277: tensor<1024xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<1024xf32>, %arg280: tensor<1024xf32>, %arg281: tensor<128x14x14x1024xf32>, %arg282: tensor<1024xf32>, %arg283: tensor<1024xf32>, %arg284: tensor<1024xf32>, %arg285: tensor<1024xf32>, %arg286: tensor<128x14x14x1024xf32>, %arg287: tensor<1024xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x1024xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x1024xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x1024xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x1024xf32>
      %321 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %322 = stablehlo.subtract %arg276, %321 : tensor<128x14x14x1024xf32>
      %323 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x14x14x1024xf32>
      %325 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x14x14x1024xf32>
      %327 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x14x14x1024xf32>
      %329 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %330 = stablehlo.subtract %arg286, %329 : tensor<128x14x14x1024xf32>
      %331 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x14x14x1024xf32>
      %333 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x14x14x1024xf32>
      %335 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x14x14x1024xf32>
      %337 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %338 = stablehlo.subtract %arg281, %337 : tensor<128x14x14x1024xf32>
      %339 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<128x14x14x1024xf32>
      %341 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<128x14x14x1024xf32>
      %343 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %344 = stablehlo.add %342, %343 : tensor<128x14x14x1024xf32>
      %345 = stablehlo.add %336, %344 : tensor<128x14x14x1024xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %346 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
      %347 = stablehlo.maximum %345, %346 : tensor<128x14x14x1024xf32>
      %348 = stablehlo.add %328, %347 : tensor<128x14x14x1024xf32>
      %349 = stablehlo.maximum %348, %346 : tensor<128x14x14x1024xf32>
      %350 = stablehlo.add %320, %349 : tensor<128x14x14x1024xf32>
      %351 = stablehlo.maximum %350, %346 : tensor<128x14x14x1024xf32>
      mhlo.return %351 : tensor<128x14x14x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %214 = mhlo.bitcast %213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x1024xf32>) -> tensor<25088x1024xf32>
    %215 = "mhlo.fusion"(%arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x1024x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x1024x256xf32>
      mhlo.return %314 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %216 = mhlo.bitcast %215 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %217 = stablehlo.dot %214, %216, precision = [DEFAULT, DEFAULT] : (tensor<25088x1024xf32>, tensor<1024x256xf32>) -> tensor<25088x256xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x256xf32>) -> tensor<128x14x14x256xf32>
    %219 = "mhlo.fusion"(%arg104, %arg103, %15, %218, %arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %220 = "mhlo.fusion"(%arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x256x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
      %314 = mhlo.copy %313 : tensor<3x3x256x256xf32>
      mhlo.return %314 : tensor<3x3x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %221 = stablehlo.convolution(%219, %220) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<128x14x14x256xf32>
    %222 = "mhlo.fusion"(%arg107, %arg106, %14, %221, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %223 = mhlo.bitcast %222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x256xf32>) -> tensor<25088x256xf32>
    %224 = "mhlo.fusion"(%arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x1024xf32>
      mhlo.return %314 : tensor<1x1x256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %225 = mhlo.bitcast %224 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %226 = stablehlo.dot %223, %225, precision = [DEFAULT, DEFAULT] : (tensor<25088x256xf32>, tensor<256x1024xf32>) -> tensor<25088x1024xf32>
    %227 = mhlo.bitcast %226 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x1024xf32>) -> tensor<128x14x14x1024xf32>
    %228 = "mhlo.fusion"(%arg110, %arg109, %13, %227, %arg233, %arg101, %arg100, %16, %212, %arg227, %arg92, %arg91, %19, %197, %arg221, %arg74, %arg73, %22, %165, %arg209, %arg83, %arg82, %166, %182, %arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<128x14x14x1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<128x14x14x1024xf32>, %arg277: tensor<1024xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<1024xf32>, %arg280: tensor<1024xf32>, %arg281: tensor<128x14x14x1024xf32>, %arg282: tensor<1024xf32>, %arg283: tensor<1024xf32>, %arg284: tensor<1024xf32>, %arg285: tensor<1024xf32>, %arg286: tensor<128x14x14x1024xf32>, %arg287: tensor<1024xf32>, %arg288: tensor<1024xf32>, %arg289: tensor<1024xf32>, %arg290: tensor<1024xf32>, %arg291: tensor<128x14x14x1024xf32>, %arg292: tensor<1024xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x1024xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x1024xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x1024xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x1024xf32>
      %321 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %322 = stablehlo.subtract %arg276, %321 : tensor<128x14x14x1024xf32>
      %323 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x14x14x1024xf32>
      %325 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x14x14x1024xf32>
      %327 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x14x14x1024xf32>
      %329 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %330 = stablehlo.subtract %arg281, %329 : tensor<128x14x14x1024xf32>
      %331 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x14x14x1024xf32>
      %333 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x14x14x1024xf32>
      %335 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x14x14x1024xf32>
      %337 = stablehlo.broadcast_in_dim %arg292, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %338 = stablehlo.subtract %arg291, %337 : tensor<128x14x14x1024xf32>
      %339 = stablehlo.broadcast_in_dim %arg290, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<128x14x14x1024xf32>
      %341 = stablehlo.broadcast_in_dim %arg289, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<128x14x14x1024xf32>
      %343 = stablehlo.broadcast_in_dim %arg288, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %344 = stablehlo.add %342, %343 : tensor<128x14x14x1024xf32>
      %345 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %346 = stablehlo.subtract %arg286, %345 : tensor<128x14x14x1024xf32>
      %347 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<128x14x14x1024xf32>
      %349 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<128x14x14x1024xf32>
      %351 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x14x14x1024xf32>
      %353 = stablehlo.add %344, %352 : tensor<128x14x14x1024xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
      %355 = stablehlo.maximum %353, %354 : tensor<128x14x14x1024xf32>
      %356 = stablehlo.add %336, %355 : tensor<128x14x14x1024xf32>
      %357 = stablehlo.maximum %356, %354 : tensor<128x14x14x1024xf32>
      %358 = stablehlo.add %328, %357 : tensor<128x14x14x1024xf32>
      %359 = stablehlo.maximum %358, %354 : tensor<128x14x14x1024xf32>
      %360 = stablehlo.add %320, %359 : tensor<128x14x14x1024xf32>
      %361 = stablehlo.maximum %360, %354 : tensor<128x14x14x1024xf32>
      mhlo.return %361 : tensor<128x14x14x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x1024xf32>) -> tensor<25088x1024xf32>
    %230 = "mhlo.fusion"(%arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x1024x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x1024x256xf32>
      mhlo.return %314 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %231 = mhlo.bitcast %230 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %232 = stablehlo.dot %229, %231, precision = [DEFAULT, DEFAULT] : (tensor<25088x1024xf32>, tensor<1024x256xf32>) -> tensor<25088x256xf32>
    %233 = mhlo.bitcast %232 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x256xf32>) -> tensor<128x14x14x256xf32>
    %234 = "mhlo.fusion"(%arg113, %arg112, %12, %233, %arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %235 = "mhlo.fusion"(%arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x256x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
      %314 = mhlo.copy %313 : tensor<3x3x256x256xf32>
      mhlo.return %314 : tensor<3x3x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %236 = stablehlo.convolution(%234, %235) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<128x14x14x256xf32>
    %237 = "mhlo.fusion"(%arg116, %arg115, %11, %236, %arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %238 = mhlo.bitcast %237 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x256xf32>) -> tensor<25088x256xf32>
    %239 = "mhlo.fusion"(%arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x1024xf32>
      mhlo.return %314 : tensor<1x1x256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %240 = mhlo.bitcast %239 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %241 = stablehlo.dot %238, %240, precision = [DEFAULT, DEFAULT] : (tensor<25088x256xf32>, tensor<256x1024xf32>) -> tensor<25088x1024xf32>
    %242 = mhlo.bitcast %241 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x1024xf32>) -> tensor<128x14x14x1024xf32>
    %243 = "mhlo.fusion"(%arg119, %arg118, %10, %242, %arg239, %arg110, %arg109, %13, %227, %arg233, %arg101, %arg100, %16, %212, %arg227, %arg92, %arg91, %19, %197, %arg221, %arg74, %arg73, %22, %165, %arg209, %arg83, %arg82, %166, %182, %arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<128x14x14x1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<128x14x14x1024xf32>, %arg277: tensor<1024xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<1024xf32>, %arg280: tensor<1024xf32>, %arg281: tensor<128x14x14x1024xf32>, %arg282: tensor<1024xf32>, %arg283: tensor<1024xf32>, %arg284: tensor<1024xf32>, %arg285: tensor<1024xf32>, %arg286: tensor<128x14x14x1024xf32>, %arg287: tensor<1024xf32>, %arg288: tensor<1024xf32>, %arg289: tensor<1024xf32>, %arg290: tensor<1024xf32>, %arg291: tensor<128x14x14x1024xf32>, %arg292: tensor<1024xf32>, %arg293: tensor<1024xf32>, %arg294: tensor<1024xf32>, %arg295: tensor<1024xf32>, %arg296: tensor<128x14x14x1024xf32>, %arg297: tensor<1024xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x1024xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x1024xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x1024xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x1024xf32>
      %321 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %322 = stablehlo.subtract %arg276, %321 : tensor<128x14x14x1024xf32>
      %323 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x14x14x1024xf32>
      %325 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x14x14x1024xf32>
      %327 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x14x14x1024xf32>
      %329 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %330 = stablehlo.subtract %arg281, %329 : tensor<128x14x14x1024xf32>
      %331 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x14x14x1024xf32>
      %333 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x14x14x1024xf32>
      %335 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x14x14x1024xf32>
      %337 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %338 = stablehlo.subtract %arg286, %337 : tensor<128x14x14x1024xf32>
      %339 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<128x14x14x1024xf32>
      %341 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<128x14x14x1024xf32>
      %343 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %344 = stablehlo.add %342, %343 : tensor<128x14x14x1024xf32>
      %345 = stablehlo.broadcast_in_dim %arg297, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %346 = stablehlo.subtract %arg296, %345 : tensor<128x14x14x1024xf32>
      %347 = stablehlo.broadcast_in_dim %arg295, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<128x14x14x1024xf32>
      %349 = stablehlo.broadcast_in_dim %arg294, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<128x14x14x1024xf32>
      %351 = stablehlo.broadcast_in_dim %arg293, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x14x14x1024xf32>
      %353 = stablehlo.broadcast_in_dim %arg292, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %354 = stablehlo.subtract %arg291, %353 : tensor<128x14x14x1024xf32>
      %355 = stablehlo.broadcast_in_dim %arg290, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %356 = stablehlo.multiply %354, %355 : tensor<128x14x14x1024xf32>
      %357 = stablehlo.broadcast_in_dim %arg289, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %358 = stablehlo.multiply %356, %357 : tensor<128x14x14x1024xf32>
      %359 = stablehlo.broadcast_in_dim %arg288, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %360 = stablehlo.add %358, %359 : tensor<128x14x14x1024xf32>
      %361 = stablehlo.add %352, %360 : tensor<128x14x14x1024xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
      %363 = stablehlo.maximum %361, %362 : tensor<128x14x14x1024xf32>
      %364 = stablehlo.add %344, %363 : tensor<128x14x14x1024xf32>
      %365 = stablehlo.maximum %364, %362 : tensor<128x14x14x1024xf32>
      %366 = stablehlo.add %336, %365 : tensor<128x14x14x1024xf32>
      %367 = stablehlo.maximum %366, %362 : tensor<128x14x14x1024xf32>
      %368 = stablehlo.add %328, %367 : tensor<128x14x14x1024xf32>
      %369 = stablehlo.maximum %368, %362 : tensor<128x14x14x1024xf32>
      %370 = stablehlo.add %320, %369 : tensor<128x14x14x1024xf32>
      %371 = stablehlo.maximum %370, %362 : tensor<128x14x14x1024xf32>
      mhlo.return %371 : tensor<128x14x14x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %244 = mhlo.bitcast %243 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x1024xf32>) -> tensor<25088x1024xf32>
    %245 = "mhlo.fusion"(%arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x1024x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %314 = mhlo.copy %313 : tensor<1x1x1024x256xf32>
      mhlo.return %314 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %246 = mhlo.bitcast %245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %247 = stablehlo.dot %244, %246, precision = [DEFAULT, DEFAULT] : (tensor<25088x1024xf32>, tensor<1024x256xf32>) -> tensor<25088x256xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x256xf32>) -> tensor<128x14x14x256xf32>
    %249 = "mhlo.fusion"(%arg122, %arg121, %9, %248, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %250 = "mhlo.fusion"(%arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256x256x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,256,256]{1,0,2,3}"} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
      %314 = mhlo.copy %313 : tensor<3x3x256x256xf32>
      mhlo.return %314 : tensor<3x3x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x3x3xf32>) -> tensor<3x3x256x256xf32>
    %251 = stablehlo.convolution(%249, %250) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<128x14x14x256xf32>
    %252 = "mhlo.fusion"(%arg125, %arg124, %8, %251, %arg243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<128x14x14x256xf32>, %arg272: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x256xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<256xf32>) -> tensor<128x14x14x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x256xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x256xf32>
      mhlo.return %322 : tensor<128x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x14x14x256xf32>, tensor<256xf32>) -> tensor<128x14x14x256xf32>
    %253 = mhlo.bitcast %252 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x256xf32>) -> tensor<25088x256xf32>
    %254 = "mhlo.fusion"(%arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024x256x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,1024]{1,0,2,3}"} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
      %314 = mhlo.copy %313 : tensor<1x1x256x1024xf32>
      mhlo.return %314 : tensor<1x1x256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x1x1xf32>) -> tensor<1x1x256x1024xf32>
    %255 = mhlo.bitcast %254 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %256 = stablehlo.dot %253, %255, precision = [DEFAULT, DEFAULT] : (tensor<25088x256xf32>, tensor<256x1024xf32>) -> tensor<25088x1024xf32>
    %257 = mhlo.bitcast %256 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x1024xf32>) -> tensor<128x14x14x1024xf32>
    %258 = "mhlo.fusion"(%arg128, %arg127, %7, %257, %arg245, %arg119, %arg118, %10, %242, %arg239, %arg110, %arg109, %13, %227, %arg233, %arg101, %arg100, %16, %212, %arg227, %arg92, %arg91, %19, %197, %arg221, %arg74, %arg73, %22, %165, %arg209, %arg83, %arg82, %166, %182, %arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<1024xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<128x14x14x1024xf32>, %arg272: tensor<1024xf32>, %arg273: tensor<1024xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<128x14x14x1024xf32>, %arg277: tensor<1024xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<1024xf32>, %arg280: tensor<1024xf32>, %arg281: tensor<128x14x14x1024xf32>, %arg282: tensor<1024xf32>, %arg283: tensor<1024xf32>, %arg284: tensor<1024xf32>, %arg285: tensor<1024xf32>, %arg286: tensor<128x14x14x1024xf32>, %arg287: tensor<1024xf32>, %arg288: tensor<1024xf32>, %arg289: tensor<1024xf32>, %arg290: tensor<1024xf32>, %arg291: tensor<128x14x14x1024xf32>, %arg292: tensor<1024xf32>, %arg293: tensor<1024xf32>, %arg294: tensor<1024xf32>, %arg295: tensor<1024xf32>, %arg296: tensor<128x14x14x1024xf32>, %arg297: tensor<1024xf32>, %arg298: tensor<1024xf32>, %arg299: tensor<1024xf32>, %arg300: tensor<1024xf32>, %arg301: tensor<128x14x14x1024xf32>, %arg302: tensor<1024xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x1024xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x1024xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x1024xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x1024xf32>
      %321 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %322 = stablehlo.subtract %arg276, %321 : tensor<128x14x14x1024xf32>
      %323 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x14x14x1024xf32>
      %325 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x14x14x1024xf32>
      %327 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x14x14x1024xf32>
      %329 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %330 = stablehlo.subtract %arg281, %329 : tensor<128x14x14x1024xf32>
      %331 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x14x14x1024xf32>
      %333 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x14x14x1024xf32>
      %335 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x14x14x1024xf32>
      %337 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %338 = stablehlo.subtract %arg286, %337 : tensor<128x14x14x1024xf32>
      %339 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<128x14x14x1024xf32>
      %341 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<128x14x14x1024xf32>
      %343 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %344 = stablehlo.add %342, %343 : tensor<128x14x14x1024xf32>
      %345 = stablehlo.broadcast_in_dim %arg292, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %346 = stablehlo.subtract %arg291, %345 : tensor<128x14x14x1024xf32>
      %347 = stablehlo.broadcast_in_dim %arg290, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<128x14x14x1024xf32>
      %349 = stablehlo.broadcast_in_dim %arg289, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<128x14x14x1024xf32>
      %351 = stablehlo.broadcast_in_dim %arg288, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x14x14x1024xf32>
      %353 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %354 = stablehlo.subtract %arg301, %353 : tensor<128x14x14x1024xf32>
      %355 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %356 = stablehlo.multiply %354, %355 : tensor<128x14x14x1024xf32>
      %357 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %358 = stablehlo.multiply %356, %357 : tensor<128x14x14x1024xf32>
      %359 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %360 = stablehlo.add %358, %359 : tensor<128x14x14x1024xf32>
      %361 = stablehlo.broadcast_in_dim %arg297, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %362 = stablehlo.subtract %arg296, %361 : tensor<128x14x14x1024xf32>
      %363 = stablehlo.broadcast_in_dim %arg295, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %364 = stablehlo.multiply %362, %363 : tensor<128x14x14x1024xf32>
      %365 = stablehlo.broadcast_in_dim %arg294, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %366 = stablehlo.multiply %364, %365 : tensor<128x14x14x1024xf32>
      %367 = stablehlo.broadcast_in_dim %arg293, dims = [3] : (tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
      %368 = stablehlo.add %366, %367 : tensor<128x14x14x1024xf32>
      %369 = stablehlo.add %360, %368 : tensor<128x14x14x1024xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x1024xf32>
      %371 = stablehlo.maximum %369, %370 : tensor<128x14x14x1024xf32>
      %372 = stablehlo.add %352, %371 : tensor<128x14x14x1024xf32>
      %373 = stablehlo.maximum %372, %370 : tensor<128x14x14x1024xf32>
      %374 = stablehlo.add %344, %373 : tensor<128x14x14x1024xf32>
      %375 = stablehlo.maximum %374, %370 : tensor<128x14x14x1024xf32>
      %376 = stablehlo.add %336, %375 : tensor<128x14x14x1024xf32>
      %377 = stablehlo.maximum %376, %370 : tensor<128x14x14x1024xf32>
      %378 = stablehlo.add %328, %377 : tensor<128x14x14x1024xf32>
      %379 = stablehlo.maximum %378, %370 : tensor<128x14x14x1024xf32>
      %380 = stablehlo.add %320, %379 : tensor<128x14x14x1024xf32>
      %381 = stablehlo.maximum %380, %370 : tensor<128x14x14x1024xf32>
      mhlo.return %381 : tensor<128x14x14x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128x14x14x1024xf32>, tensor<1024xf32>) -> tensor<128x14x14x1024xf32>
    %259 = "mhlo.fusion"(%arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048x1024x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,2048]{1,0,2,3}"} : (tensor<2048x1024x1x1xf32>) -> tensor<1x1x1024x2048xf32>
      %314 = mhlo.copy %313 : tensor<1x1x1024x2048xf32>
      mhlo.return %314 : tensor<1x1x1024x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x1024x1x1xf32>) -> tensor<1x1x1024x2048xf32>
    %260 = stablehlo.convolution(%258, %259) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) -> tensor<128x7x7x2048xf32>
    %261 = "mhlo.fusion"(%arg254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<2048xf32>
      %315 = stablehlo.rsqrt %314 : tensor<2048xf32>
      mhlo.return %315 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>) -> tensor<2048xf32>
    %262 = "mhlo.fusion"(%arg252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %263 = "mhlo.fusion"(%arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg268, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %264 = mhlo.bitcast %258 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x14x14x1024xf32>) -> tensor<25088x1024xf32>
    %265 = "mhlo.fusion"(%arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x1024x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,512]{1,0,2,3}"} : (tensor<512x1024x1x1xf32>) -> tensor<1x1x1024x512xf32>
      %314 = mhlo.copy %313 : tensor<1x1x1024x512xf32>
      mhlo.return %314 : tensor<1x1x1024x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1024x1x1xf32>) -> tensor<1x1x1024x512xf32>
    %266 = mhlo.bitcast %265 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x512xf32>) -> tensor<1024x512xf32>
    %267 = stablehlo.dot %264, %266, precision = [DEFAULT, DEFAULT] : (tensor<25088x1024xf32>, tensor<1024x512xf32>) -> tensor<25088x512xf32>
    %268 = mhlo.bitcast %267 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x512xf32>) -> tensor<128x14x14x512xf32>
    %269 = "mhlo.fusion"(%arg134, %arg133, %263, %268, %arg249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x14x14x512xf32>, %arg272: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x14x14x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x14x14x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x14x14x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x14x14x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x14x14x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x14x14x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x14x14x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x14x14x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x14x14x512xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x14x14x512xf32>
      mhlo.return %322 : tensor<128x14x14x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x14x14x512xf32>, tensor<512xf32>) -> tensor<128x14x14x512xf32>
    %270 = "mhlo.fusion"(%arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x512x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
      %314 = mhlo.copy %313 : tensor<3x3x512x512xf32>
      mhlo.return %314 : tensor<3x3x512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
    %271 = stablehlo.convolution(%269, %270) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x14x14x512xf32>, tensor<3x3x512x512xf32>) -> tensor<128x7x7x512xf32>
    %272 = "mhlo.fusion"(%arg137, %arg136, %262, %271, %arg251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x7x7x512xf32>, %arg272: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x7x7x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x7x7x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x7x7x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x7x7x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x7x7x512xf32>
      mhlo.return %322 : tensor<128x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x7x7x512xf32>, tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %273 = mhlo.bitcast %272 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x7x7x512xf32>) -> tensor<6272x512xf32>
    %274 = "mhlo.fusion"(%arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048x512x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,2048]{1,0,2,3}"} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
      %314 = mhlo.copy %313 : tensor<1x1x512x2048xf32>
      mhlo.return %314 : tensor<1x1x512x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
    %275 = mhlo.bitcast %274 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %276 = stablehlo.dot %273, %275, precision = [DEFAULT, DEFAULT] : (tensor<6272x512xf32>, tensor<512x2048xf32>) -> tensor<6272x2048xf32>
    %277 = mhlo.bitcast %276 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x2048xf32>) -> tensor<128x7x7x2048xf32>
    %278 = "mhlo.fusion"(%arg131, %arg130, %6, %260, %arg247, %arg140, %arg139, %261, %277, %arg253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<128x7x7x2048xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>, %arg274: tensor<2048xf32>, %arg275: tensor<2048xf32>, %arg276: tensor<128x7x7x2048xf32>, %arg277: tensor<2048xf32>):
      %313 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %314 = stablehlo.subtract %arg276, %313 : tensor<128x7x7x2048xf32>
      %315 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x7x7x2048xf32>
      %317 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x7x7x2048xf32>
      %319 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x7x7x2048xf32>
      %321 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %322 = stablehlo.subtract %arg271, %321 : tensor<128x7x7x2048xf32>
      %323 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x7x7x2048xf32>
      %325 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x7x7x2048xf32>
      %327 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x7x7x2048xf32>
      %329 = stablehlo.add %320, %328 : tensor<128x7x7x2048xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x7x7x2048xf32>
      %331 = stablehlo.maximum %329, %330 : tensor<128x7x7x2048xf32>
      mhlo.return %331 : tensor<128x7x7x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %279 = mhlo.bitcast %278 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x7x7x2048xf32>) -> tensor<6272x2048xf32>
    %280 = "mhlo.fusion"(%arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x2048x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,2048,512]{1,0,2,3}"} : (tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32>
      %314 = mhlo.copy %313 : tensor<1x1x2048x512xf32>
      mhlo.return %314 : tensor<1x1x2048x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32>
    %281 = mhlo.bitcast %280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2048x512xf32>) -> tensor<2048x512xf32>
    %282 = stablehlo.dot %279, %281, precision = [DEFAULT, DEFAULT] : (tensor<6272x2048xf32>, tensor<2048x512xf32>) -> tensor<6272x512xf32>
    %283 = mhlo.bitcast %282 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x512xf32>) -> tensor<128x7x7x512xf32>
    %284 = "mhlo.fusion"(%arg143, %arg142, %5, %283, %arg255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x7x7x512xf32>, %arg272: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x7x7x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x7x7x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x7x7x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x7x7x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x7x7x512xf32>
      mhlo.return %322 : tensor<128x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x7x7x512xf32>, tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %285 = "mhlo.fusion"(%arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x512x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
      %314 = mhlo.copy %313 : tensor<3x3x512x512xf32>
      mhlo.return %314 : tensor<3x3x512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
    %286 = stablehlo.convolution(%284, %285) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<128x7x7x512xf32>
    %287 = "mhlo.fusion"(%arg146, %arg145, %4, %286, %arg257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x7x7x512xf32>, %arg272: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x7x7x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x7x7x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x7x7x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x7x7x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x7x7x512xf32>
      mhlo.return %322 : tensor<128x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x7x7x512xf32>, tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %288 = mhlo.bitcast %287 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x7x7x512xf32>) -> tensor<6272x512xf32>
    %289 = "mhlo.fusion"(%arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048x512x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,2048]{1,0,2,3}"} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
      %314 = mhlo.copy %313 : tensor<1x1x512x2048xf32>
      mhlo.return %314 : tensor<1x1x512x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
    %290 = mhlo.bitcast %289 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %291 = stablehlo.dot %288, %290, precision = [DEFAULT, DEFAULT] : (tensor<6272x512xf32>, tensor<512x2048xf32>) -> tensor<6272x2048xf32>
    %292 = mhlo.bitcast %291 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x2048xf32>) -> tensor<128x7x7x2048xf32>
    %293 = "mhlo.fusion"(%arg149, %arg148, %3, %292, %arg259, %arg131, %arg130, %6, %260, %arg247, %arg140, %arg139, %261, %277, %arg253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<128x7x7x2048xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>, %arg274: tensor<2048xf32>, %arg275: tensor<2048xf32>, %arg276: tensor<128x7x7x2048xf32>, %arg277: tensor<2048xf32>, %arg278: tensor<2048xf32>, %arg279: tensor<2048xf32>, %arg280: tensor<2048xf32>, %arg281: tensor<128x7x7x2048xf32>, %arg282: tensor<2048xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x7x7x2048xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x7x7x2048xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x7x7x2048xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x7x7x2048xf32>
      %321 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %322 = stablehlo.subtract %arg281, %321 : tensor<128x7x7x2048xf32>
      %323 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x7x7x2048xf32>
      %325 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x7x7x2048xf32>
      %327 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x7x7x2048xf32>
      %329 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %330 = stablehlo.subtract %arg276, %329 : tensor<128x7x7x2048xf32>
      %331 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x7x7x2048xf32>
      %333 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x7x7x2048xf32>
      %335 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x7x7x2048xf32>
      %337 = stablehlo.add %328, %336 : tensor<128x7x7x2048xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x7x7x2048xf32>
      %339 = stablehlo.maximum %337, %338 : tensor<128x7x7x2048xf32>
      %340 = stablehlo.add %320, %339 : tensor<128x7x7x2048xf32>
      %341 = stablehlo.maximum %340, %338 : tensor<128x7x7x2048xf32>
      mhlo.return %341 : tensor<128x7x7x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x7x7x2048xf32>) -> tensor<6272x2048xf32>
    %295 = "mhlo.fusion"(%arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x2048x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,2048,512]{1,0,2,3}"} : (tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32>
      %314 = mhlo.copy %313 : tensor<1x1x2048x512xf32>
      mhlo.return %314 : tensor<1x1x2048x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2048x1x1xf32>) -> tensor<1x1x2048x512xf32>
    %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2048x512xf32>) -> tensor<2048x512xf32>
    %297 = stablehlo.dot %294, %296, precision = [DEFAULT, DEFAULT] : (tensor<6272x2048xf32>, tensor<2048x512xf32>) -> tensor<6272x512xf32>
    %298 = mhlo.bitcast %297 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x512xf32>) -> tensor<128x7x7x512xf32>
    %299 = "mhlo.fusion"(%arg152, %arg151, %2, %298, %arg261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x7x7x512xf32>, %arg272: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x7x7x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x7x7x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x7x7x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x7x7x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x7x7x512xf32>
      mhlo.return %322 : tensor<128x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x7x7x512xf32>, tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %300 = "mhlo.fusion"(%arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512x512x3x3xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,512,512]{1,0,2,3}"} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
      %314 = mhlo.copy %313 : tensor<3x3x512x512xf32>
      mhlo.return %314 : tensor<3x3x512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512x3x3xf32>) -> tensor<3x3x512x512xf32>
    %301 = stablehlo.convolution(%299, %300) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<128x7x7x512xf32>
    %302 = "mhlo.fusion"(%arg155, %arg154, %1, %301, %arg263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<512xf32>, %arg271: tensor<128x7x7x512xf32>, %arg272: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x7x7x512xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x7x7x512xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x7x7x512xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<512xf32>) -> tensor<128x7x7x512xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x7x7x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x7x7x512xf32>
      %322 = stablehlo.maximum %320, %321 : tensor<128x7x7x512xf32>
      mhlo.return %322 : tensor<128x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x7x7x512xf32>, tensor<512xf32>) -> tensor<128x7x7x512xf32>
    %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x7x7x512xf32>) -> tensor<6272x512xf32>
    %304 = "mhlo.fusion"(%arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048x512x1x1xf32>):
      %313 = stablehlo.transpose %arg268, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,2048]{1,0,2,3}"} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
      %314 = mhlo.copy %313 : tensor<1x1x512x2048xf32>
      mhlo.return %314 : tensor<1x1x512x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x512x1x1xf32>) -> tensor<1x1x512x2048xf32>
    %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %306 = stablehlo.dot %303, %305, precision = [DEFAULT, DEFAULT] : (tensor<6272x512xf32>, tensor<512x2048xf32>) -> tensor<6272x2048xf32>
    %307 = mhlo.bitcast %306 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x2048xf32>) -> tensor<128x7x7x2048xf32>
    %308 = "mhlo.fusion"(%arg158, %arg157, %0, %307, %arg265, %arg149, %arg148, %3, %292, %arg259, %arg131, %arg130, %6, %260, %arg247, %arg140, %arg139, %261, %277, %arg253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<2048xf32>, %arg269: tensor<2048xf32>, %arg270: tensor<2048xf32>, %arg271: tensor<128x7x7x2048xf32>, %arg272: tensor<2048xf32>, %arg273: tensor<2048xf32>, %arg274: tensor<2048xf32>, %arg275: tensor<2048xf32>, %arg276: tensor<128x7x7x2048xf32>, %arg277: tensor<2048xf32>, %arg278: tensor<2048xf32>, %arg279: tensor<2048xf32>, %arg280: tensor<2048xf32>, %arg281: tensor<128x7x7x2048xf32>, %arg282: tensor<2048xf32>, %arg283: tensor<2048xf32>, %arg284: tensor<2048xf32>, %arg285: tensor<2048xf32>, %arg286: tensor<128x7x7x2048xf32>, %arg287: tensor<2048xf32>):
      %313 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %314 = stablehlo.subtract %arg271, %313 : tensor<128x7x7x2048xf32>
      %315 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<128x7x7x2048xf32>
      %317 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x7x7x2048xf32>
      %319 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x7x7x2048xf32>
      %321 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %322 = stablehlo.subtract %arg276, %321 : tensor<128x7x7x2048xf32>
      %323 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<128x7x7x2048xf32>
      %325 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<128x7x7x2048xf32>
      %327 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %328 = stablehlo.add %326, %327 : tensor<128x7x7x2048xf32>
      %329 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %330 = stablehlo.subtract %arg286, %329 : tensor<128x7x7x2048xf32>
      %331 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %332 = stablehlo.multiply %330, %331 : tensor<128x7x7x2048xf32>
      %333 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<128x7x7x2048xf32>
      %335 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %336 = stablehlo.add %334, %335 : tensor<128x7x7x2048xf32>
      %337 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %338 = stablehlo.subtract %arg281, %337 : tensor<128x7x7x2048xf32>
      %339 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<128x7x7x2048xf32>
      %341 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<128x7x7x2048xf32>
      %343 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
      %344 = stablehlo.add %342, %343 : tensor<128x7x7x2048xf32>
      %345 = stablehlo.add %336, %344 : tensor<128x7x7x2048xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %346 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x7x7x2048xf32>
      %347 = stablehlo.maximum %345, %346 : tensor<128x7x7x2048xf32>
      %348 = stablehlo.add %328, %347 : tensor<128x7x7x2048xf32>
      %349 = stablehlo.maximum %348, %346 : tensor<128x7x7x2048xf32>
      %350 = stablehlo.add %320, %349 : tensor<128x7x7x2048xf32>
      %351 = stablehlo.maximum %350, %346 : tensor<128x7x7x2048xf32>
      mhlo.return %351 : tensor<128x7x7x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128x7x7x2048xf32>, tensor<2048xf32>) -> tensor<128x7x7x2048xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %309 = "mhlo.fusion"(%308, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg268: tensor<128x7x7x2048xf32>, %arg269: tensor<f32>):
      %313 = stablehlo.reduce(%arg268 init: %arg269) applies stablehlo.add across dimensions = [2, 1] : (tensor<128x7x7x2048xf32>, tensor<f32>) -> tensor<128x2048xf32>
      mhlo.return %313 : tensor<128x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<128x7x7x2048xf32>, tensor<f32>) -> tensor<128x2048xf32>
    %310 = "mhlo.fusion"(%309) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x2048xf32>):
      %cst_1 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x2048xf32>
      %314 = stablehlo.multiply %arg268, %313 : tensor<128x2048xf32>
      mhlo.return %314 : tensor<128x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<128x2048xf32>) -> tensor<128x2048xf32>
    %311 = stablehlo.dot_general %310, %arg159, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x2048xf32>, tensor<1000x2048xf32>) -> tensor<128x1000xf32>
    %312 = "mhlo.fusion"(%311, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg268: tensor<128x1000xf32>, %arg269: tensor<1000xf32>):
      %313 = stablehlo.broadcast_in_dim %arg269, dims = [1] : (tensor<1000xf32>) -> tensor<128x1000xf32>
      %314 = stablehlo.add %313, %arg268 : tensor<128x1000xf32>
      mhlo.return %314 : tensor<128x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1000xf32>, tensor<1000xf32>) -> tensor<128x1000xf32>
    return %312 : tensor<128x1000xf32>
  }
}
