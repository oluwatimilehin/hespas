module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_99.128(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<16x8x8x512xf32>, %arg1: tensor<f32>) -> tensor<16x512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1, 2] : (tensor<16x8x8x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %0 : tensor<16x512xf32>
  }
  func.func private @region_3.3.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<16x4096xf32>, %arg1: tensor<f32>) -> tensor<16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    return %0 : tensor<16x4xf32>
  }
  func.func private @region_3.3.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<16x4096xf32>, %arg1: tensor<f32>) -> tensor<16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    return %0 : tensor<16x4xf32>
  }
  func.func private @region_16.23.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.3(%arg0: tensor<16x128x4x8xf32>, %arg1: tensor<f32>) -> tensor<16x128x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x128x4x8xf32>, tensor<f32>) -> tensor<16x128x4xf32>
    return %0 : tensor<16x128x4xf32>
  }
  func.func private @region_3.3.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.4(%arg0: tensor<16x4096xf32>, %arg1: tensor<f32>) -> tensor<16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    return %0 : tensor<16x4xf32>
  }
  func.func private @region_3.3.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<16x4096xf32>, %arg1: tensor<f32>) -> tensor<16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    return %0 : tensor<16x4xf32>
  }
  func.func private @region_7.11.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.6(%arg0: tensor<16x128x4x8xf32>, %arg1: tensor<f32>) -> tensor<16x128x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x128x4x8xf32>, tensor<f32>) -> tensor<16x128x4xf32>
    return %0 : tensor<16x128x4xf32>
  }
  func.func private @region_3.3.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.7(%arg0: tensor<16x4096xf32>, %arg1: tensor<f32>) -> tensor<16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    return %0 : tensor<16x4xf32>
  }
  func.func private @fused_computation.8(%arg0: tensor<16x2xf32>, %arg1: tensor<2xf32>) -> tensor<16x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<16x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16x2xf32>
    return %1 : tensor<16x2xf32>
  }
  func.func private @fused_computation.9(%arg0: tensor<16x512xf32>) -> tensor<16x512xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16x512xf32>
    return %1 : tensor<16x512xf32>
  }
  func.func private @fused_computation.10(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x8x8x512xf32>, %arg4: tensor<512xf32>) -> tensor<16x8x8x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x8x8x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x8x8x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x8x8x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x8x8x512xf32>
    return %7 : tensor<16x8x8x512xf32>
  }
  func.func private @fused_computation.11(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.12(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.13(%arg0: tensor<16x256x8x8xf32>) -> tensor<16x8x8x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,8,256]{2,1,3,0}"} : (tensor<16x256x8x8xf32>) -> tensor<16x8x8x256xf32>
    %1 = mhlo.copy %0 : tensor<16x8x8x256xf32>
    return %1 : tensor<16x8x8x256xf32>
  }
  func.func private @fused_computation.14(%arg0: tensor<256xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x256x4x16xf32>, %arg4: tensor<16x4x16x256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>) -> tensor<64x16x256x1x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %2 = stablehlo.add %arg4, %1 : tensor<16x4x16x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
    %4 = mhlo.copy %3 : tensor<16x256x4x16xf32>
    %5 = stablehlo.add %4, %arg3 : tensor<16x256x4x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %8 = stablehlo.multiply %arg2, %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<16x16384xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x16384xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<16x256x4x16xf32>
    %16 = stablehlo.add %0, %15 : tensor<16x256x4x16xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x256x2x2x4x4xf32>
    %18 = stablehlo.transpose %17, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,4,2,4,16,256]{3,1,2,0,5,4}"} : (tensor<16x256x2x2x4x4xf32>) -> tensor<2x4x2x4x16x256xf32>
    %19 = mhlo.copy %18 : tensor<2x4x2x4x16x256xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[64,16,256,1,1]{2,1,0,4,3}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<64x16x256x1x1xf32>
    %21 = mhlo.copy %20 : tensor<64x16x256x1x1xf32>
    return %21 : tensor<64x16x256x1x1xf32>
  }
  func.func private @region_69.88.clone.5.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.16(%arg0: tensor<16xf32>, %arg1: tensor<16x256x4x16xf32>, %arg2: tensor<16x4x16x256xf32>, %arg3: tensor<256xf32>) -> tensor<16x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<16x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x16xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<16x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %9 = stablehlo.subtract %5, %8 : tensor<16x16384xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<16x16384xf32>
    return %10 : tensor<16x16384xf32>
  }
  func.func private @fused_computation.17(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<1x16x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
    return %5 : tensor<1x16x16384xf32>
  }
  func.func private @fused_computation.18(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.19(%arg0: tensor<16x4x16x512xf32>, %arg1: tensor<512xf32>) -> tensor<16x4x16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<16x4x16x512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x16x512xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,4,16]{1,3,2,0}"} : (tensor<16x4x16x512xf32>) -> tensor<16x512x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x512x4x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
    %5 = stablehlo.negate %3 : tensor<16x512x4x16xf32>
    %6 = stablehlo.exponential %5 : tensor<16x512x4x16xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x512x4x16xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x512x4x16xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x512x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,512]{2,1,3,0}"} : (tensor<16x512x4x16xf32>) -> tensor<16x4x16x512xf32>
    %11 = mhlo.copy %10 : tensor<16x4x16x512xf32>
    return %11 : tensor<16x4x16x512xf32>
  }
  func.func private @fused_computation.20(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.21(%arg0: tensor<256xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<16x4x16x256xf32>
    return %13 : tensor<16x4x16x256xf32>
  }
  func.func private @region_69.88.clone.4.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.22(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.23(%arg0: tensor<16xf32>, %arg1: tensor<16x256x4x16xf32>) -> tensor<16x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x16384xf32>
    return %5 : tensor<16x16384xf32>
  }
  func.func private @fused_computation.24(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x256x4x16xf32>
    return %4 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_computation.25(%arg0: tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
  func.func private @fused_computation.26(%arg0: tensor<16x256x4xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x256x4x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x256x4x16xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %10 = mhlo.copy %9 : tensor<16x4x16x256xf32>
    return %10 : tensor<16x4x16x256xf32>
  }
  func.func private @region_91.118(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.27(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x256x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
    return %8 : tensor<16x256x4xf32>
  }
  func.func private @fused_computation.28(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x16xf32>
    return %2 : tensor<16x1x4x16xf32>
  }
  func.func private @fused_computation.29(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x1x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x16xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x16xf32>
    return %10 : tensor<16x1x4x16xf32>
  }
  func.func private @region_89.116(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.30(%arg0: tensor<16x4x16x513xf32>, %arg1: tensor<513xf32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %5 : tensor<16x1x4xf32>
  }
  func.func private @fused_computation.31(%arg0: tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x513xf32>
    return %1 : tensor<1x1x256x513xf32>
  }
  func.func private @fused_computation.32(%arg0: tensor<256xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<16x4x16x256xf32>
    return %13 : tensor<16x4x16x256xf32>
  }
  func.func private @region_69.88.clone.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.33(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.34(%arg0: tensor<16xf32>, %arg1: tensor<16x256x4x16xf32>) -> tensor<16x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x16384xf32>
    return %5 : tensor<16x16384xf32>
  }
  func.func private @fused_computation.35(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x256x4x16xf32>
    return %4 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_computation.36(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.37(%arg0: tensor<16x4x16x512xf32>, %arg1: tensor<512xf32>) -> tensor<16x4x16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<16x4x16x512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x16x512xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,4,16]{1,3,2,0}"} : (tensor<16x4x16x512xf32>) -> tensor<16x512x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x512x4x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
    %5 = stablehlo.negate %3 : tensor<16x512x4x16xf32>
    %6 = stablehlo.exponential %5 : tensor<16x512x4x16xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x512x4x16xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x512x4x16xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x512x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,512]{2,1,3,0}"} : (tensor<16x512x4x16xf32>) -> tensor<16x4x16x512xf32>
    %11 = mhlo.copy %10 : tensor<16x4x16x512xf32>
    return %11 : tensor<16x4x16x512xf32>
  }
  func.func private @fused_computation.38(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.39(%arg0: tensor<256xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<16x4x16x256xf32>
    return %13 : tensor<16x4x16x256xf32>
  }
  func.func private @region_69.88.clone.2.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.40(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.41(%arg0: tensor<16xf32>, %arg1: tensor<16x256x4x16xf32>) -> tensor<16x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x16384xf32>
    return %5 : tensor<16x16384xf32>
  }
  func.func private @fused_computation.42(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x256x4x16xf32>
    return %4 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_computation.43(%arg0: tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
  func.func private @fused_computation.44(%arg0: tensor<16x256x4xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x256x4x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x256x4x16xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %10 = mhlo.copy %9 : tensor<16x4x16x256xf32>
    return %10 : tensor<16x4x16x256xf32>
  }
  func.func private @region_82.107(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.45(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x256x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
    return %8 : tensor<16x256x4xf32>
  }
  func.func private @fused_computation.46(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x16xf32>
    return %2 : tensor<16x1x4x16xf32>
  }
  func.func private @fused_computation.47(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x1x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x16xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x16xf32>
    return %10 : tensor<16x1x4x16xf32>
  }
  func.func private @region_80.105(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.48(%arg0: tensor<16x4x16x513xf32>, %arg1: tensor<513xf32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %5 : tensor<16x1x4xf32>
  }
  func.func private @fused_computation.49(%arg0: tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x513xf32>
    return %1 : tensor<1x1x256x513xf32>
  }
  func.func private @fused_computation.50(%arg0: tensor<256xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<16x4x16x256xf32>
    return %13 : tensor<16x4x16x256xf32>
  }
  func.func private @region_69.88.clone.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.51(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.52(%arg0: tensor<16xf32>, %arg1: tensor<16x256x4x16xf32>) -> tensor<16x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x16384xf32>
    return %5 : tensor<16x16384xf32>
  }
  func.func private @fused_computation.53(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x256x4x16xf32>
    return %4 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_computation.54(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.55(%arg0: tensor<16x4x16x512xf32>, %arg1: tensor<512xf32>) -> tensor<16x4x16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<16x4x16x512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x16x512xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,4,16]{1,3,2,0}"} : (tensor<16x4x16x512xf32>) -> tensor<16x512x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x512x4x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
    %5 = stablehlo.negate %3 : tensor<16x512x4x16xf32>
    %6 = stablehlo.exponential %5 : tensor<16x512x4x16xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x512x4x16xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x512x4x16xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x512x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,512]{2,1,3,0}"} : (tensor<16x512x4x16xf32>) -> tensor<16x4x16x512xf32>
    %11 = mhlo.copy %10 : tensor<16x4x16x512xf32>
    return %11 : tensor<16x4x16x512xf32>
  }
  func.func private @fused_computation.56(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.57(%arg0: tensor<256xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<16x4x16x256xf32>
    return %13 : tensor<16x4x16x256xf32>
  }
  func.func private @region_69.88.clone.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.58(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.59(%arg0: tensor<16xf32>, %arg1: tensor<16x256x4x16xf32>) -> tensor<16x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x16384xf32>
    return %5 : tensor<16x16384xf32>
  }
  func.func private @fused_computation.60(%arg0: tensor<16x4x16x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x16x256x1x1xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x16xf32>
    %4 = mhlo.bitcast %arg2 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
    %5 = stablehlo.transpose %4, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
    %6 = mhlo.copy %5 : tensor<16x256x2x2x4x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x256x4x16xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x256x4x16xf32>
    return %8 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_computation.61(%arg0: tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
  func.func private @fused_computation.62(%arg0: tensor<16x256x4xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x256x4x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x256x4x16xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %10 = mhlo.copy %9 : tensor<16x4x16x256xf32>
    return %10 : tensor<16x4x16x256xf32>
  }
  func.func private @region_73.95(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.63(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x256x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
    return %8 : tensor<16x256x4xf32>
  }
  func.func private @fused_computation.64(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x16xf32>
    return %2 : tensor<16x1x4x16xf32>
  }
  func.func private @fused_computation.65(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<16x1x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x16xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x16xf32>
    return %10 : tensor<16x1x4x16xf32>
  }
  func.func private @region_71.93(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.66(%arg0: tensor<16x4x16x513xf32>, %arg1: tensor<513xf32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<16x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %5 : tensor<16x1x4xf32>
  }
  func.func private @fused_computation.67(%arg0: tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x513xf32>
    return %1 : tensor<1x1x256x513xf32>
  }
  func.func private @fused_computation.68(%arg0: tensor<256xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<64x16x256x1x1xf32>, %arg4: tensor<256xf32>) -> tensor<16x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
    %3 = mhlo.copy %2 : tensor<16x256x2x2x4x4xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.multiply %arg2, %5 : tensor<16xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x16384xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x16384xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<16x256x4x16xf32>
    %14 = stablehlo.add %0, %13 : tensor<16x256x4x16xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
    %16 = mhlo.copy %15 : tensor<16x4x16x256xf32>
    return %16 : tensor<16x4x16x256xf32>
  }
  func.func private @region_69.88.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.69(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.70(%arg0: tensor<16xf32>, %arg1: tensor<64x16x256x1x1xf32>) -> tensor<16x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
    %2 = mhlo.copy %1 : tensor<16x256x2x2x4x4xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
    %7 = stablehlo.subtract %3, %6 : tensor<16x16384xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x16384xf32>
    return %8 : tensor<16x16384xf32>
  }
  func.func private @fused_computation.71(%arg0: tensor<64x16x256x1x1xf32>) -> tensor<1x16x16384xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
    %2 = mhlo.copy %1 : tensor<16x256x2x2x4x4xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<1x16x16384xf32>
    return %3 : tensor<1x16x16384xf32>
  }
  func.func private @fused_computation.72(%arg0: tensor<16x8x8x256xf32>, %arg1: tensor<2x4x2x4x2xi32>) -> tensor<64x16x256x1x1xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,8,8]{1,3,2,0}"} : (tensor<16x8x8x256xf32>) -> tensor<16x256x8x8xf32>
    %1 = mhlo.copy %0 : tensor<16x256x8x8xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
    %3 = "stablehlo.gather"(%1, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x16x256x1x1xf32>
    return %3 : tensor<64x16x256x1x1xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.74(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x8x8x512xf32>, %arg4: tensor<512xf32>) -> tensor<16x8x8x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x8x8x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x8x8x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x8x8x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x8x8x512xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,8,8]{1,3,2,0}"} : (tensor<16x8x8x512xf32>) -> tensor<16x512x8x8xf32>
    %9 = mhlo.copy %8 : tensor<16x512x8x8xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x8x8xf32>
    %11 = stablehlo.negate %9 : tensor<16x512x8x8xf32>
    %12 = stablehlo.exponential %11 : tensor<16x512x8x8xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x512x8x8xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x512x8x8xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x512x8x8xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,8,512]{2,1,3,0}"} : (tensor<16x512x8x8xf32>) -> tensor<16x8x8x512xf32>
    %17 = mhlo.copy %16 : tensor<16x8x8x512xf32>
    return %17 : tensor<16x8x8x512xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.76(%arg0: tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x512xf32>
    return %1 : tensor<3x3x1x512xf32>
  }
  func.func private @fused_computation.77(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x8x8x512xf32>, %arg4: tensor<512xf32>) -> tensor<16x8x8x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x8x8x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x8x8x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x8x8x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x8x8x512xf32>
    return %7 : tensor<16x8x8x512xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.79(%arg0: tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,768,512]{1,0,2,3}"} : (tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x768x512xf32>
    return %1 : tensor<1x1x768x512xf32>
  }
  func.func private @fused_computation.80(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<16x8x8x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x8x8x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<768xf32>) -> tensor<16x8x8x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x8x8x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<16x8x8x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x8x8x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<16x8x8x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x8x8x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<16x8x8x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x8x8x768xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,768,8,8]{1,3,2,0}"} : (tensor<16x8x8x768xf32>) -> tensor<16x768x8x8xf32>
    %9 = mhlo.copy %8 : tensor<16x768x8x8xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x768x8x8xf32>
    %11 = stablehlo.negate %9 : tensor<16x768x8x8xf32>
    %12 = stablehlo.exponential %11 : tensor<16x768x8x8xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x768x8x8xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x768x8x8xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x768x8x8xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,8,768]{2,1,3,0}"} : (tensor<16x768x8x8xf32>) -> tensor<16x8x8x768xf32>
    %17 = mhlo.copy %16 : tensor<16x8x8x768xf32>
    return %17 : tensor<16x8x8x768xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.82(%arg0: tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,768]{1,0,2,3}"} : (tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x768xf32>
    return %1 : tensor<3x3x1x768xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<16x16x16x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x16x16x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<768xf32>) -> tensor<16x16x16x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x16x16x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<16x16x16x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x16x16x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<16x16x16x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x16x16x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<16x16x16x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x16x16x768xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,768,16,16]{1,3,2,0}"} : (tensor<16x16x16x768xf32>) -> tensor<16x768x16x16xf32>
    %9 = mhlo.copy %8 : tensor<16x768x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x768x16x16xf32>
    %11 = stablehlo.negate %9 : tensor<16x768x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<16x768x16x16xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x768x16x16xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x768x16x16xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x768x16x16xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,768]{2,1,3,0}"} : (tensor<16x768x16x16xf32>) -> tensor<16x16x16x768xf32>
    %17 = mhlo.copy %16 : tensor<16x16x16x768xf32>
    return %17 : tensor<16x16x16x768xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.85(%arg0: tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,768]{1,0,2,3}"} : (tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x768xf32>
    return %1 : tensor<1x1x384x768xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<16x16x16x384xf32>, %arg4: tensor<384xf32>) -> tensor<16x16x16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x16x16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x16x16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x16x16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x16x16x384xf32>
    return %7 : tensor<16x16x16x384xf32>
  }
  func.func private @fused_computation.87(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.88(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.89(%arg0: tensor<16x192x16x16xf32>) -> tensor<16x16x16x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,192]{2,1,3,0}"} : (tensor<16x192x16x16xf32>) -> tensor<16x16x16x192xf32>
    %1 = mhlo.copy %0 : tensor<16x16x16x192xf32>
    return %1 : tensor<16x16x16x192xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x192x4x64xf32>, %arg4: tensor<16x4x64x192xf32>, %arg5: tensor<192xf32>, %arg6: tensor<192xf32>) -> tensor<256x16x192x1x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %2 = stablehlo.add %arg4, %1 : tensor<16x4x64x192xf32>
    %3 = stablehlo.transpose %2, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %4 = mhlo.copy %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.add %4, %arg3 : tensor<16x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %8 = stablehlo.multiply %arg2, %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<16x49152xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x49152xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<16x192x4x64xf32>
    %16 = stablehlo.add %0, %15 : tensor<16x192x4x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x192x2x2x8x8xf32>
    %18 = stablehlo.transpose %17, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,8,2,8,16,192]{3,1,2,0,5,4}"} : (tensor<16x192x2x2x8x8xf32>) -> tensor<2x8x2x8x16x192xf32>
    %19 = mhlo.copy %18 : tensor<2x8x2x8x16x192xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[256,16,192,1,1]{2,1,0,4,3}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<256x16x192x1x1xf32>
    %21 = mhlo.copy %20 : tensor<256x16x192x1x1xf32>
    return %21 : tensor<256x16x192x1x1xf32>
  }
  func.func private @region_27.35.clone.7.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.91(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<16xf32>, %arg1: tensor<16x192x4x64xf32>, %arg2: tensor<16x4x64x192xf32>, %arg3: tensor<192xf32>) -> tensor<16x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %9 = stablehlo.subtract %5, %8 : tensor<16x49152xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<16x49152xf32>
    return %10 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.93(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x16x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    return %5 : tensor<1x16x49152xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<16x4x64x384xf32>, %arg1: tensor<384xf32>) -> tensor<16x4x64x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<16x4x64x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x384xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,4,64]{1,3,2,0}"} : (tensor<16x4x64x384xf32>) -> tensor<16x384x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x384x4x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
    %5 = stablehlo.negate %3 : tensor<16x384x4x64xf32>
    %6 = stablehlo.exponential %5 : tensor<16x384x4x64xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x384x4x64xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x384x4x64xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x384x4x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,384]{2,1,3,0}"} : (tensor<16x384x4x64xf32>) -> tensor<16x4x64x384xf32>
    %11 = mhlo.copy %10 : tensor<16x4x64x384xf32>
    return %11 : tensor<16x4x64x384xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.97(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<16x4x64x192xf32>
    return %13 : tensor<16x4x64x192xf32>
  }
  func.func private @region_27.35.clone.6.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.98(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<16xf32>, %arg1: tensor<16x192x4x64xf32>) -> tensor<16x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x49152xf32>
    return %5 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.100(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x192x4x64xf32>
    return %4 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x192xf32>
    return %1 : tensor<1x1x192x192xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<16x192x4xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x192x4x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x192x4x64xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %10 = mhlo.copy %9 : tensor<16x4x64x192xf32>
    return %10 : tensor<16x4x64x192xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x192x4x64xf32>
    return %7 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x64xf32>
    return %2 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x64xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x64xf32>
    return %10 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<16x4x64x385xf32>, %arg1: tensor<385xf32>) -> tensor<16x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    return %4 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.107(%arg0: tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x385xf32>
    return %1 : tensor<1x1x192x385xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<16x4x64x192xf32>
    return %13 : tensor<16x4x64x192xf32>
  }
  func.func private @region_27.35.clone.5.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.109(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.110(%arg0: tensor<16xf32>, %arg1: tensor<16x192x4x64xf32>) -> tensor<16x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x49152xf32>
    return %5 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x192x4x64xf32>
    return %4 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<16x4x64x384xf32>, %arg1: tensor<384xf32>) -> tensor<16x4x64x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<16x4x64x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x384xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,4,64]{1,3,2,0}"} : (tensor<16x4x64x384xf32>) -> tensor<16x384x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x384x4x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
    %5 = stablehlo.negate %3 : tensor<16x384x4x64xf32>
    %6 = stablehlo.exponential %5 : tensor<16x384x4x64xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x384x4x64xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x384x4x64xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x384x4x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,384]{2,1,3,0}"} : (tensor<16x384x4x64xf32>) -> tensor<16x4x64x384xf32>
    %11 = mhlo.copy %10 : tensor<16x4x64x384xf32>
    return %11 : tensor<16x4x64x384xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<16x4x64x192xf32>
    return %13 : tensor<16x4x64x192xf32>
  }
  func.func private @region_27.35.clone.4.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.116(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.117(%arg0: tensor<16xf32>, %arg1: tensor<16x192x4x64xf32>) -> tensor<16x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x49152xf32>
    return %5 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x192x4x64xf32>
    return %4 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.119(%arg0: tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x192xf32>
    return %1 : tensor<1x1x192x192xf32>
  }
  func.func private @fused_computation.120(%arg0: tensor<16x192x4xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x192x4x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x192x4x64xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %10 = mhlo.copy %9 : tensor<16x4x64x192xf32>
    return %10 : tensor<16x4x64x192xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x192x4x64xf32>
    return %7 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x64xf32>
    return %2 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x64xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x64xf32>
    return %10 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<16x4x64x385xf32>, %arg1: tensor<385xf32>) -> tensor<16x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    return %4 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x385xf32>
    return %1 : tensor<1x1x192x385xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<16x4x64x192xf32>
    return %13 : tensor<16x4x64x192xf32>
  }
  func.func private @region_27.35.clone.3.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.127(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<16xf32>, %arg1: tensor<16x192x4x64xf32>) -> tensor<16x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x49152xf32>
    return %5 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.129(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x192x4x64xf32>
    return %4 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<16x4x64x384xf32>, %arg1: tensor<384xf32>) -> tensor<16x4x64x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<16x4x64x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x384xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,4,64]{1,3,2,0}"} : (tensor<16x4x64x384xf32>) -> tensor<16x384x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x384x4x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
    %5 = stablehlo.negate %3 : tensor<16x384x4x64xf32>
    %6 = stablehlo.exponential %5 : tensor<16x384x4x64xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x384x4x64xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x384x4x64xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x384x4x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,384]{2,1,3,0}"} : (tensor<16x384x4x64xf32>) -> tensor<16x4x64x384xf32>
    %11 = mhlo.copy %10 : tensor<16x4x64x384xf32>
    return %11 : tensor<16x4x64x384xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<16x4x64x192xf32>
    return %13 : tensor<16x4x64x192xf32>
  }
  func.func private @region_27.35.clone.2.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.134(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<16xf32>, %arg1: tensor<16x192x4x64xf32>) -> tensor<16x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x49152xf32>
    return %5 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.136(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x192x4x64xf32>
    return %4 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x192xf32>
    return %1 : tensor<1x1x192x192xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<16x192x4xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x192x4x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x192x4x64xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %10 = mhlo.copy %9 : tensor<16x4x64x192xf32>
    return %10 : tensor<16x4x64x192xf32>
  }
  func.func private @fused_computation.139(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x192x4x64xf32>
    return %7 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.140(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x64xf32>
    return %2 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x64xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x64xf32>
    return %10 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<16x4x64x385xf32>, %arg1: tensor<385xf32>) -> tensor<16x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    return %4 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x385xf32>
    return %1 : tensor<1x1x192x385xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<16x4x64x192xf32>
    return %13 : tensor<16x4x64x192xf32>
  }
  func.func private @region_27.35.clone.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.145(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<16xf32>, %arg1: tensor<16x192x4x64xf32>) -> tensor<16x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x49152xf32>
    return %5 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x192x4x64xf32>
    return %4 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.149(%arg0: tensor<16x4x64x384xf32>, %arg1: tensor<384xf32>) -> tensor<16x4x64x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<16x4x64x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x384xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,4,64]{1,3,2,0}"} : (tensor<16x4x64x384xf32>) -> tensor<16x384x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x384x4x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
    %5 = stablehlo.negate %3 : tensor<16x384x4x64xf32>
    %6 = stablehlo.exponential %5 : tensor<16x384x4x64xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x384x4x64xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x384x4x64xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x384x4x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,384]{2,1,3,0}"} : (tensor<16x384x4x64xf32>) -> tensor<16x4x64x384xf32>
    %11 = mhlo.copy %10 : tensor<16x4x64x384xf32>
    return %11 : tensor<16x4x64x384xf32>
  }
  func.func private @fused_computation.150(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<16x4x64x192xf32>
    return %13 : tensor<16x4x64x192xf32>
  }
  func.func private @region_27.35.clone.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.152(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<16xf32>, %arg1: tensor<16x192x4x64xf32>) -> tensor<16x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x49152xf32>
    return %5 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<16x4x64x192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<256x16x192x1x1xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x192x4x64xf32>
    %4 = mhlo.bitcast %arg2 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
    %5 = stablehlo.transpose %4, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
    %6 = mhlo.copy %5 : tensor<16x192x2x2x8x8xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x192x4x64xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x192x4x64xf32>
    return %8 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x192xf32>
    return %1 : tensor<1x1x192x192xf32>
  }
  func.func private @fused_computation.156(%arg0: tensor<16x192x4xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x192x4x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x192x4x64xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %10 = mhlo.copy %9 : tensor<16x4x64x192xf32>
    return %10 : tensor<16x4x64x192xf32>
  }
  func.func private @fused_computation.157(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x192x4x64xf32>
    return %7 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x64xf32>
    return %2 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<16x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x64xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x64xf32>
    return %10 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<16x4x64x385xf32>, %arg1: tensor<385xf32>) -> tensor<16x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<16x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    return %4 : tensor<16x1x4x64xf32>
  }
  func.func private @fused_computation.161(%arg0: tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x385xf32>
    return %1 : tensor<1x1x192x385xf32>
  }
  func.func private @fused_computation.162(%arg0: tensor<192xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<256x16x192x1x1xf32>, %arg4: tensor<192xf32>) -> tensor<16x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
    %2 = stablehlo.transpose %1, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
    %3 = mhlo.copy %2 : tensor<16x192x2x2x8x8xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.multiply %arg2, %5 : tensor<16xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x49152xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x49152xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<16x192x4x64xf32>
    %14 = stablehlo.add %0, %13 : tensor<16x192x4x64xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
    %16 = mhlo.copy %15 : tensor<16x4x64x192xf32>
    return %16 : tensor<16x4x64x192xf32>
  }
  func.func private @region_27.35.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.163(%arg0: tensor<16x2xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.164(%arg0: tensor<16xf32>, %arg1: tensor<256x16x192x1x1xf32>) -> tensor<16x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
    %2 = mhlo.copy %1 : tensor<16x192x2x2x8x8xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
    %7 = stablehlo.subtract %3, %6 : tensor<16x49152xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x49152xf32>
    return %8 : tensor<16x49152xf32>
  }
  func.func private @fused_computation.165(%arg0: tensor<256x16x192x1x1xf32>) -> tensor<1x16x49152xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
    %2 = mhlo.copy %1 : tensor<16x192x2x2x8x8xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<1x16x49152xf32>
    return %3 : tensor<1x16x49152xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<16x16x16x192xf32>, %arg1: tensor<2x8x2x8x2xi32>) -> tensor<256x16x192x1x1xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,16,16]{1,3,2,0}"} : (tensor<16x16x16x192xf32>) -> tensor<16x192x16x16xf32>
    %1 = mhlo.copy %0 : tensor<16x192x16x16xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
    %3 = "stablehlo.gather"(%1, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 192, 1, 1>}> : (tensor<16x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x16x192x1x1xf32>
    return %3 : tensor<256x16x192x1x1xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.168(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<16x16x16x384xf32>, %arg4: tensor<384xf32>) -> tensor<16x16x16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x16x16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x16x16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x16x16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x16x16x384xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,16,16]{1,3,2,0}"} : (tensor<16x16x16x384xf32>) -> tensor<16x384x16x16xf32>
    %9 = mhlo.copy %8 : tensor<16x384x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x16x16xf32>
    %11 = stablehlo.negate %9 : tensor<16x384x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<16x384x16x16xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x384x16x16xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x384x16x16xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x384x16x16xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,384]{2,1,3,0}"} : (tensor<16x384x16x16xf32>) -> tensor<16x16x16x384xf32>
    %17 = mhlo.copy %16 : tensor<16x16x16x384xf32>
    return %17 : tensor<16x16x16x384xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.170(%arg0: tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,384]{1,0,2,3}"} : (tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x384xf32>
    return %1 : tensor<3x3x1x384xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<16x16x16x384xf32>, %arg4: tensor<384xf32>) -> tensor<16x16x16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x16x16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x16x16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x16x16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x16x16x384xf32>
    return %7 : tensor<16x16x16x384xf32>
  }
  func.func private @fused_computation.172(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.173(%arg0: tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,384]{1,0,2,3}"} : (tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x384xf32>
    return %1 : tensor<1x1x512x384xf32>
  }
  func.func private @fused_computation.174(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x16x16x512xf32>, %arg4: tensor<512xf32>) -> tensor<16x16x16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<16x16x16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x16x16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<16x16x16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x16x16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<16x16x16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x16x16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<16x16x16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x16x16x512xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,16,16]{1,3,2,0}"} : (tensor<16x16x16x512xf32>) -> tensor<16x512x16x16xf32>
    %9 = mhlo.copy %8 : tensor<16x512x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x16x16xf32>
    %11 = stablehlo.negate %9 : tensor<16x512x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<16x512x16x16xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x512x16x16xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x512x16x16xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x512x16x16xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,512]{2,1,3,0}"} : (tensor<16x512x16x16xf32>) -> tensor<16x16x16x512xf32>
    %17 = mhlo.copy %16 : tensor<16x16x16x512xf32>
    return %17 : tensor<16x16x16x512xf32>
  }
  func.func private @fused_computation.175(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x512xf32>
    return %1 : tensor<3x3x1x512xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x32x32x512xf32>, %arg4: tensor<512xf32>) -> tensor<16x32x32x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<16x32x32x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x32x32x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<16x32x32x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x32x32x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<16x32x32x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x32x32x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<16x32x32x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x32x32x512xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,32,32]{1,3,2,0}"} : (tensor<16x32x32x512xf32>) -> tensor<16x512x32x32xf32>
    %9 = mhlo.copy %8 : tensor<16x512x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %11 = stablehlo.negate %9 : tensor<16x512x32x32xf32>
    %12 = stablehlo.exponential %11 : tensor<16x512x32x32xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x512x32x32xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x512x32x32xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x512x32x32xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,512]{2,1,3,0}"} : (tensor<16x512x32x32xf32>) -> tensor<16x32x32x512xf32>
    %17 = mhlo.copy %16 : tensor<16x32x32x512xf32>
    return %17 : tensor<16x32x32x512xf32>
  }
  func.func private @fused_computation.178(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.179(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x32x32x256xf32>, %arg4: tensor<256xf32>) -> tensor<16x32x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x32x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x32x32x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x32x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x32x32x256xf32>
    return %7 : tensor<16x32x32x256xf32>
  }
  func.func private @fused_computation.181(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.182(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<16x128x32x32xf32>) -> tensor<16x32x32x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,128]{2,1,3,0}"} : (tensor<16x128x32x32xf32>) -> tensor<16x32x32x128xf32>
    %1 = mhlo.copy %0 : tensor<16x32x32x128xf32>
    return %1 : tensor<16x32x32x128xf32>
  }
  func.func private @fused_computation.184(%arg0: tensor<128xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x4x256xf32>, %arg4: tensor<16x4x256x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>) -> tensor<1024x16x128x1x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %2 = stablehlo.add %arg4, %1 : tensor<16x4x256x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
    %4 = mhlo.copy %3 : tensor<16x128x4x256xf32>
    %5 = stablehlo.add %4, %arg3 : tensor<16x128x4x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %8 = stablehlo.multiply %arg2, %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<16x131072xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x131072xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<16x128x4x256xf32>
    %16 = stablehlo.add %0, %15 : tensor<16x128x4x256xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x128x2x2x16x16xf32>
    %18 = stablehlo.transpose %17, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,16,2,16,16,128]{3,1,2,0,5,4}"} : (tensor<16x128x2x2x16x16xf32>) -> tensor<2x16x2x16x16x128xf32>
    %19 = mhlo.copy %18 : tensor<2x16x2x16x16x128xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1024,16,128,1,1]{2,1,0,4,3}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<1024x16x128x1x1xf32>
    %21 = mhlo.copy %20 : tensor<1024x16x128x1x1xf32>
    return %21 : tensor<1024x16x128x1x1xf32>
  }
  func.func private @region_3.3.clone.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.185(%arg0: tensor<16x4xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.186(%arg0: tensor<16xf32>, %arg1: tensor<16x128x4x256xf32>, %arg2: tensor<16x4x256x128xf32>, %arg3: tensor<128xf32>) -> tensor<16x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<16x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x128x4x256xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<16x128x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %9 = stablehlo.subtract %5, %8 : tensor<16x131072xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<16x131072xf32>
    return %10 : tensor<16x131072xf32>
  }
  func.func private @fused_computation.187(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<16x4x256x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x16x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x128x4x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x128x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
    return %5 : tensor<1x16x131072xf32>
  }
  func.func private @fused_computation.188(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.189(%arg0: tensor<16x4x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<16x4x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x4x256x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x256x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,256]{1,3,2,0}"} : (tensor<16x4x256x256xf32>) -> tensor<16x256x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x256xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x256xf32>
    %5 = stablehlo.negate %3 : tensor<16x256x4x256xf32>
    %6 = stablehlo.exponential %5 : tensor<16x256x4x256xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x256x4x256xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x256x4x256xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x256x4x256xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,256]{2,1,3,0}"} : (tensor<16x256x4x256xf32>) -> tensor<16x4x256x256xf32>
    %11 = mhlo.copy %10 : tensor<16x4x256x256xf32>
    return %11 : tensor<16x4x256x256xf32>
  }
  func.func private @fused_computation.190(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.191(%arg0: tensor<128xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x4x256xf32>, %arg4: tensor<128xf32>) -> tensor<16x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x131072xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x131072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x128x4x256xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x128x4x256xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
    %13 = mhlo.copy %12 : tensor<16x4x256x128xf32>
    return %13 : tensor<16x4x256x128xf32>
  }
  func.func private @region_3.3.clone.2.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.192(%arg0: tensor<16x4xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.193(%arg0: tensor<16xf32>, %arg1: tensor<16x128x4x256xf32>) -> tensor<16x131072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x131072xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x131072xf32>
    return %5 : tensor<16x131072xf32>
  }
  func.func private @fused_computation.194(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<16x4x256x128xf32>, %arg2: tensor<128xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x128x4x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x128x4x256xf32>
    return %4 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_computation.195(%arg0: tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x128xf32>
    return %1 : tensor<1x1x128x128xf32>
  }
  func.func private @fused_computation.196(%arg0: tensor<16x128x4xf32>, %arg1: tensor<16x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<16x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:16, 129:257, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x4x256xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x128x4x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x128x4xf32>) -> tensor<16x128x4x256xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x128x4x256xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
    %10 = mhlo.copy %9 : tensor<16x4x256x128xf32>
    return %10 : tensor<16x4x256x128xf32>
  }
  func.func private @fused_computation.197(%arg0: tensor<16x1x4x256xf32>, %arg1: tensor<16x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:16, 1:129, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x256xf32>) -> tensor<16x128x4x256xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x128x4x256xf32>
    return %7 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_computation.198(%arg0: tensor<16x1x4x256xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x256xf32>
    return %2 : tensor<16x1x4x256xf32>
  }
  func.func private @fused_computation.199(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<16x1x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x256xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x256xf32>
    return %10 : tensor<16x1x4x256xf32>
  }
  func.func private @fused_computation.200(%arg0: tensor<16x4x256x257xf32>, %arg1: tensor<257xf32>) -> tensor<16x1x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
    return %4 : tensor<16x1x4x256xf32>
  }
  func.func private @fused_computation.201(%arg0: tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x257xf32>
    return %1 : tensor<1x1x128x257xf32>
  }
  func.func private @fused_computation.202(%arg0: tensor<128xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x4x256xf32>, %arg4: tensor<128xf32>) -> tensor<16x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x131072xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x131072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x128x4x256xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x128x4x256xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
    %13 = mhlo.copy %12 : tensor<16x4x256x128xf32>
    return %13 : tensor<16x4x256x128xf32>
  }
  func.func private @region_3.3.clone.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.203(%arg0: tensor<16x4xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.204(%arg0: tensor<16xf32>, %arg1: tensor<16x128x4x256xf32>) -> tensor<16x131072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x131072xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x131072xf32>
    return %5 : tensor<16x131072xf32>
  }
  func.func private @fused_computation.205(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<16x4x256x128xf32>, %arg2: tensor<128xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x128x4x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<16x128x4x256xf32>
    return %4 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_computation.206(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.207(%arg0: tensor<16x4x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<16x4x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x4x256x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x256x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,256]{1,3,2,0}"} : (tensor<16x4x256x256xf32>) -> tensor<16x256x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x256x4x256xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x256xf32>
    %5 = stablehlo.negate %3 : tensor<16x256x4x256xf32>
    %6 = stablehlo.exponential %5 : tensor<16x256x4x256xf32>
    %7 = stablehlo.add %4, %6 : tensor<16x256x4x256xf32>
    %8 = stablehlo.divide %4, %7 : tensor<16x256x4x256xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<16x256x4x256xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,256]{2,1,3,0}"} : (tensor<16x256x4x256xf32>) -> tensor<16x4x256x256xf32>
    %11 = mhlo.copy %10 : tensor<16x4x256x256xf32>
    return %11 : tensor<16x4x256x256xf32>
  }
  func.func private @fused_computation.208(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.209(%arg0: tensor<128xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x4x256xf32>, %arg4: tensor<128xf32>) -> tensor<16x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<16x131072xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x131072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x128x4x256xf32>
    %11 = stablehlo.add %0, %10 : tensor<16x128x4x256xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
    %13 = mhlo.copy %12 : tensor<16x4x256x128xf32>
    return %13 : tensor<16x4x256x128xf32>
  }
  func.func private @region_3.3.clone.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.210(%arg0: tensor<16x4xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.211(%arg0: tensor<16xf32>, %arg1: tensor<16x128x4x256xf32>) -> tensor<16x131072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x131072xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x131072xf32>
    return %5 : tensor<16x131072xf32>
  }
  func.func private @fused_computation.212(%arg0: tensor<16x4x256x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1024x16x128x1x1xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x128x4x256xf32>
    %4 = mhlo.bitcast %arg2 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
    %5 = stablehlo.transpose %4, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
    %6 = mhlo.copy %5 : tensor<16x128x2x2x16x16xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x128x4x256xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x128x4x256xf32>
    return %8 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_computation.213(%arg0: tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x128xf32>
    return %1 : tensor<1x1x128x128xf32>
  }
  func.func private @fused_computation.214(%arg0: tensor<16x128x4xf32>, %arg1: tensor<16x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<16x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:16, 129:257, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x4x256xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<16x128x4x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x128x4xf32>) -> tensor<16x128x4x256xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x128x4x256xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
    %10 = mhlo.copy %9 : tensor<16x4x256x128xf32>
    return %10 : tensor<16x4x256x128xf32>
  }
  func.func private @fused_computation.215(%arg0: tensor<16x1x4x256xf32>, %arg1: tensor<16x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:16, 1:129, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<16x4x256xf32>) -> tensor<16x128x4x256xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<16x128x4x256xf32>
    return %7 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_computation.216(%arg0: tensor<16x1x4x256xf32>, %arg1: tensor<16x1x4xf32>) -> tensor<16x1x4x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<16x1x4x256xf32>
    return %2 : tensor<16x1x4x256xf32>
  }
  func.func private @fused_computation.217(%arg0: tensor<16x1x4xf32>, %arg1: tensor<16x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<16x1x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<16x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<16x1x4x256xf32>
    %10 = stablehlo.exponential %9 : tensor<16x1x4x256xf32>
    return %10 : tensor<16x1x4x256xf32>
  }
  func.func private @fused_computation.218(%arg0: tensor<16x4x256x257xf32>, %arg1: tensor<257xf32>) -> tensor<16x1x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<16x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
    return %4 : tensor<16x1x4x256xf32>
  }
  func.func private @fused_computation.219(%arg0: tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x257xf32>
    return %1 : tensor<1x1x128x257xf32>
  }
  func.func private @fused_computation.220(%arg0: tensor<128xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1024x16x128x1x1xf32>, %arg4: tensor<128xf32>) -> tensor<16x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
    %2 = stablehlo.transpose %1, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
    %3 = mhlo.copy %2 : tensor<16x128x2x2x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.multiply %arg2, %5 : tensor<16xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x131072xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<16x131072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<16x128x4x256xf32>
    %14 = stablehlo.add %0, %13 : tensor<16x128x4x256xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
    %16 = mhlo.copy %15 : tensor<16x4x256x128xf32>
    return %16 : tensor<16x4x256x128xf32>
  }
  func.func private @region_3.3.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.221(%arg0: tensor<16x4xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.add %2, %3 : tensor<16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16xf32>
    return %5 : tensor<16xf32>
  }
  func.func private @fused_computation.222(%arg0: tensor<16xf32>, %arg1: tensor<1024x16x128x1x1xf32>) -> tensor<16x131072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
    %2 = mhlo.copy %1 : tensor<16x128x2x2x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
    %7 = stablehlo.subtract %3, %6 : tensor<16x131072xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x131072xf32>
    return %8 : tensor<16x131072xf32>
  }
  func.func private @fused_computation.223(%arg0: tensor<1024x16x128x1x1xf32>) -> tensor<1x16x131072xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
    %2 = mhlo.copy %1 : tensor<16x128x2x2x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<1x16x131072xf32>
    return %3 : tensor<1x16x131072xf32>
  }
  func.func private @fused_computation.224(%arg0: tensor<16x32x32x128xf32>, %arg1: tensor<2x16x2x16x2xi32>) -> tensor<1024x16x128x1x1xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,32,32]{1,3,2,0}"} : (tensor<16x32x32x128xf32>) -> tensor<16x128x32x32xf32>
    %1 = mhlo.copy %0 : tensor<16x128x32x32xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
    %3 = "stablehlo.gather"(%1, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 128, 1, 1>}> : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x16x128x1x1xf32>
    return %3 : tensor<1024x16x128x1x1xf32>
  }
  func.func private @fused_computation.225(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.226(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x32x32x256xf32>, %arg4: tensor<256xf32>) -> tensor<16x32x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x32x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x32x32x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x32x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x32x32x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,32,32]{1,3,2,0}"} : (tensor<16x32x32x256xf32>) -> tensor<16x256x32x32xf32>
    %9 = mhlo.copy %8 : tensor<16x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x32x32xf32>
    %11 = stablehlo.negate %9 : tensor<16x256x32x32xf32>
    %12 = stablehlo.exponential %11 : tensor<16x256x32x32xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x256x32x32xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x256x32x32xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x256x32x32xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,256]{2,1,3,0}"} : (tensor<16x256x32x32xf32>) -> tensor<16x32x32x256xf32>
    %17 = mhlo.copy %16 : tensor<16x32x32x256xf32>
    return %17 : tensor<16x32x32x256xf32>
  }
  func.func private @fused_computation.227(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.228(%arg0: tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x256xf32>
    return %1 : tensor<3x3x1x256xf32>
  }
  func.func private @fused_computation.229(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x32x32x256xf32>, %arg4: tensor<256xf32>) -> tensor<16x32x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x32x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x32x32x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x32x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x32x32x256xf32>
    return %7 : tensor<16x32x32x256xf32>
  }
  func.func private @fused_computation.230(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.231(%arg0: tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
  func.func private @fused_computation.232(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x32x32x256xf32>, %arg4: tensor<256xf32>) -> tensor<16x32x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x32x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x32x32x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x32x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x32x32x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,32,32]{1,3,2,0}"} : (tensor<16x32x32x256xf32>) -> tensor<16x256x32x32xf32>
    %9 = mhlo.copy %8 : tensor<16x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x32x32xf32>
    %11 = stablehlo.negate %9 : tensor<16x256x32x32xf32>
    %12 = stablehlo.exponential %11 : tensor<16x256x32x32xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x256x32x32xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x256x32x32xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x256x32x32xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,256]{2,1,3,0}"} : (tensor<16x256x32x32xf32>) -> tensor<16x32x32x256xf32>
    %17 = mhlo.copy %16 : tensor<16x32x32x256xf32>
    return %17 : tensor<16x32x32x256xf32>
  }
  func.func private @fused_computation.233(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.234(%arg0: tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x256xf32>
    return %1 : tensor<3x3x1x256xf32>
  }
  func.func private @fused_computation.235(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x64x64x256xf32>, %arg4: tensor<256xf32>) -> tensor<16x64x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x64x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x64x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x64x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
    %9 = mhlo.copy %8 : tensor<16x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %11 = stablehlo.negate %9 : tensor<16x256x64x64xf32>
    %12 = stablehlo.exponential %11 : tensor<16x256x64x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x256x64x64xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x256x64x64xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x256x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,256]{2,1,3,0}"} : (tensor<16x256x64x64xf32>) -> tensor<16x64x64x256xf32>
    %17 = mhlo.copy %16 : tensor<16x64x64x256xf32>
    return %17 : tensor<16x64x64x256xf32>
  }
  func.func private @fused_computation.236(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.237(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.238(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<16x64x64x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<16x64x64x128xf32>, %arg9: tensor<128xf32>) -> tensor<16x64x64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<16x64x64x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x64x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x64x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x64x128xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<16x64x64x128xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<16x64x64x128xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<16x64x64x128xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %15 = stablehlo.add %13, %14 : tensor<16x64x64x128xf32>
    %16 = stablehlo.add %7, %15 : tensor<16x64x64x128xf32>
    return %16 : tensor<16x64x64x128xf32>
  }
  func.func private @fused_computation.239(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.240(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.241(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x64x64x256xf32>, %arg4: tensor<256xf32>) -> tensor<16x64x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x64x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x64x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x64x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
    %9 = mhlo.copy %8 : tensor<16x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %11 = stablehlo.negate %9 : tensor<16x256x64x64xf32>
    %12 = stablehlo.exponential %11 : tensor<16x256x64x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x256x64x64xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x256x64x64xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x256x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,256]{2,1,3,0}"} : (tensor<16x256x64x64xf32>) -> tensor<16x64x64x256xf32>
    %17 = mhlo.copy %16 : tensor<16x64x64x256xf32>
    return %17 : tensor<16x64x64x256xf32>
  }
  func.func private @fused_computation.242(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.243(%arg0: tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x256xf32>
    return %1 : tensor<3x3x1x256xf32>
  }
  func.func private @fused_computation.244(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x64x64x256xf32>, %arg4: tensor<256xf32>) -> tensor<16x64x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x64x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x64x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x64x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
    %9 = mhlo.copy %8 : tensor<16x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %11 = stablehlo.negate %9 : tensor<16x256x64x64xf32>
    %12 = stablehlo.exponential %11 : tensor<16x256x64x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x256x64x64xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x256x64x64xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x256x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,256]{2,1,3,0}"} : (tensor<16x256x64x64xf32>) -> tensor<16x64x64x256xf32>
    %17 = mhlo.copy %16 : tensor<16x64x64x256xf32>
    return %17 : tensor<16x64x64x256xf32>
  }
  func.func private @fused_computation.245(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.246(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.247(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<16x64x64x128xf32>, %arg4: tensor<128xf32>) -> tensor<16x64x64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x64x64x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x64x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x64x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x64x128xf32>
    return %7 : tensor<16x64x64x128xf32>
  }
  func.func private @fused_computation.248(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.249(%arg0: tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x128xf32>
    return %1 : tensor<1x1x128x128xf32>
  }
  func.func private @fused_computation.250(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<16x64x64x128xf32>, %arg4: tensor<128xf32>) -> tensor<16x64x64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x64x64x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x64x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x64x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x64x128xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,64,64]{1,3,2,0}"} : (tensor<16x64x64x128xf32>) -> tensor<16x128x64x64xf32>
    %9 = mhlo.copy %8 : tensor<16x128x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x64x64xf32>
    %11 = stablehlo.negate %9 : tensor<16x128x64x64xf32>
    %12 = stablehlo.exponential %11 : tensor<16x128x64x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x128x64x64xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x128x64x64xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x128x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,128]{2,1,3,0}"} : (tensor<16x128x64x64xf32>) -> tensor<16x64x64x128xf32>
    %17 = mhlo.copy %16 : tensor<16x64x64x128xf32>
    return %17 : tensor<16x64x64x128xf32>
  }
  func.func private @fused_computation.251(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.252(%arg0: tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x128xf32>
    return %1 : tensor<3x3x1x128xf32>
  }
  func.func private @fused_computation.253(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<16x128x128x128xf32>, %arg4: tensor<128xf32>) -> tensor<16x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x128x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x128x128xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{1,3,2,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
    %9 = mhlo.copy %8 : tensor<16x128x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x128x128xf32>
    %11 = stablehlo.negate %9 : tensor<16x128x128x128xf32>
    %12 = stablehlo.exponential %11 : tensor<16x128x128x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x128x128x128xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x128x128x128xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x128x128x128xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{2,1,3,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
    %17 = mhlo.copy %16 : tensor<16x128x128x128xf32>
    return %17 : tensor<16x128x128x128xf32>
  }
  func.func private @fused_computation.254(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.255(%arg0: tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x128xf32>
    return %1 : tensor<1x1x64x128xf32>
  }
  func.func private @fused_computation.256(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<16x128x128x64xf32>, %arg4: tensor<64xf32>) -> tensor<16x128x128x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x128x128x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x128x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x128x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x128x64xf32>
    return %7 : tensor<16x128x128x64xf32>
  }
  func.func private @fused_computation.257(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.258(%arg0: tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x64xf32>
    return %1 : tensor<1x1x64x64xf32>
  }
  func.func private @fused_computation.259(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<16x128x128x64xf32>, %arg4: tensor<64xf32>) -> tensor<16x128x128x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x128x128x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x128x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x128x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x128x64xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,128,128]{1,3,2,0}"} : (tensor<16x128x128x64xf32>) -> tensor<16x64x128x128xf32>
    %9 = mhlo.copy %8 : tensor<16x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x64x128x128xf32>
    %11 = stablehlo.negate %9 : tensor<16x64x128x128xf32>
    %12 = stablehlo.exponential %11 : tensor<16x64x128x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x64x128x128xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x64x128x128xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x64x128x128xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,64]{2,1,3,0}"} : (tensor<16x64x128x128xf32>) -> tensor<16x128x128x64xf32>
    %17 = mhlo.copy %16 : tensor<16x128x128x64xf32>
    return %17 : tensor<16x128x128x64xf32>
  }
  func.func private @fused_computation.260(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.261(%arg0: tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,64]{1,0,2,3}"} : (tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x64xf32>
    return %1 : tensor<3x3x1x64xf32>
  }
  func.func private @fused_computation.262(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<16x128x128x64xf32>, %arg4: tensor<64xf32>) -> tensor<16x128x128x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x128x128x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x128x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x128x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x128x64xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,128,128]{1,3,2,0}"} : (tensor<16x128x128x64xf32>) -> tensor<16x64x128x128xf32>
    %9 = mhlo.copy %8 : tensor<16x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x64x128x128xf32>
    %11 = stablehlo.negate %9 : tensor<16x64x128x128xf32>
    %12 = stablehlo.exponential %11 : tensor<16x64x128x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x64x128x128xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x64x128x128xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x64x128x128xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,64]{2,1,3,0}"} : (tensor<16x64x128x128xf32>) -> tensor<16x128x128x64xf32>
    %17 = mhlo.copy %16 : tensor<16x128x128x64xf32>
    return %17 : tensor<16x128x128x64xf32>
  }
  func.func private @fused_computation.263(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.264(%arg0: tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,32,64]{1,0,2,3}"} : (tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32>
    %1 = mhlo.copy %0 : tensor<1x1x32x64xf32>
    return %1 : tensor<1x1x32x64xf32>
  }
  func.func private @fused_computation.265(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<16x128x128x32xf32>, %arg4: tensor<32xf32>) -> tensor<16x128x128x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x128x128x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x128x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x128x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x128x32xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
    %9 = mhlo.copy %8 : tensor<16x32x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32x128x128xf32>
    %11 = stablehlo.negate %9 : tensor<16x32x128x128xf32>
    %12 = stablehlo.exponential %11 : tensor<16x32x128x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<16x32x128x128xf32>
    %14 = stablehlo.divide %10, %13 : tensor<16x32x128x128xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<16x32x128x128xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,32]{2,1,3,0}"} : (tensor<16x32x128x128xf32>) -> tensor<16x128x128x32xf32>
    %17 = mhlo.copy %16 : tensor<16x128x128x32xf32>
    return %17 : tensor<16x128x128x32xf32>
  }
  func.func private @fused_computation.266(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<32xf32>
    %2 = stablehlo.rsqrt %1 : tensor<32xf32>
    return %2 : tensor<32xf32>
  }
  func.func private @fused_computation.267(%arg0: tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,3,32]{1,0,2,3}"} : (tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32>
    %1 = mhlo.copy %0 : tensor<3x3x3x32xf32>
    return %1 : tensor<3x3x3x32xf32>
  }
  func.func private @fused_computation.268(%arg0: tensor<16x3x256x256xf32>) -> tensor<16x256x256x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,256,3]{2,1,3,0}"} : (tensor<16x3x256x256xf32>) -> tensor<16x256x256x3xf32>
    %1 = mhlo.copy %0 : tensor<16x256x256x3xf32>
    return %1 : tensor<16x256x256x3xf32>
  }
  func.func private @fused_computation.269() -> tensor<1024x2xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x16xi32>
    %1 = stablehlo.iota dim = 0 : tensor<2x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<2x16xi32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
    %4 = mhlo.copy %3 : tensor<2x16x1x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
    %6 = stablehlo.compare LT, %4, %5 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
    %c_0 = stablehlo.constant dense<32> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
    %8 = stablehlo.add %4, %7 : tensor<2x16x1x1xi32>
    %9 = stablehlo.select %6, %8, %4 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %13 = stablehlo.compare LT, %2, %12 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %15 = stablehlo.add %2, %14 : tensor<2x16xi32>
    %16 = stablehlo.select %13, %15, %2 : tensor<2x16xi1>, tensor<2x16xi32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %18 = stablehlo.concatenate %11, %17, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
    return %19 : tensor<1024x2xi32>
  }
  func.func private @fused_computation.270() -> tensor<2x16x2x16x2xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x16xi32>
    %1 = stablehlo.iota dim = 0 : tensor<2x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<2x16xi32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
    %4 = mhlo.copy %3 : tensor<2x16x1x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
    %6 = stablehlo.compare LT, %4, %5 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
    %c_0 = stablehlo.constant dense<32> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
    %8 = stablehlo.add %4, %7 : tensor<2x16x1x1xi32>
    %9 = stablehlo.select %6, %8, %4 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %13 = stablehlo.compare LT, %2, %12 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %15 = stablehlo.add %2, %14 : tensor<2x16xi32>
    %16 = stablehlo.select %13, %15, %2 : tensor<2x16xi1>, tensor<2x16xi32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %18 = stablehlo.concatenate %11, %17, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
    return %18 : tensor<2x16x2x16x2xi32>
  }
  func.func private @fused_computation.271() -> tensor<256x2xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<8xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x8xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x8xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
    %7 = mhlo.copy %6 : tensor<2x8x1x1xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
    %c_1 = stablehlo.constant dense<16> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
    %11 = stablehlo.add %7, %10 : tensor<2x8x1x1xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %15 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %16 = stablehlo.compare LT, %5, %15 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %18 = stablehlo.add %5, %17 : tensor<2x8xi32>
    %19 = stablehlo.select %16, %18, %5 : tensor<2x8xi1>, tensor<2x8xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %21 = stablehlo.concatenate %14, %20, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
    return %22 : tensor<256x2xi32>
  }
  func.func private @fused_computation.272() -> tensor<2x8x2x8x2xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<8xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x8xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x8xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
    %7 = mhlo.copy %6 : tensor<2x8x1x1xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
    %c_1 = stablehlo.constant dense<16> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
    %11 = stablehlo.add %7, %10 : tensor<2x8x1x1xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %15 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %16 = stablehlo.compare LT, %5, %15 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %18 = stablehlo.add %5, %17 : tensor<2x8xi32>
    %19 = stablehlo.select %16, %18, %5 : tensor<2x8xi1>, tensor<2x8xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %21 = stablehlo.concatenate %14, %20, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
    return %21 : tensor<2x8x2x8x2xi32>
  }
  func.func private @fused_computation.273() -> tensor<64x2xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<4xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x4xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x4xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
    %7 = mhlo.copy %6 : tensor<2x4x1x1xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
    %c_1 = stablehlo.constant dense<8> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
    %11 = stablehlo.add %7, %10 : tensor<2x4x1x1xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %15 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %16 = stablehlo.compare LT, %5, %15 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %18 = stablehlo.add %5, %17 : tensor<2x4xi32>
    %19 = stablehlo.select %16, %18, %5 : tensor<2x4xi1>, tensor<2x4xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %21 = stablehlo.concatenate %14, %20, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
    return %22 : tensor<64x2xi32>
  }
  func.func private @fused_computation.274() -> tensor<2x4x2x4x2xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<4xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x4xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x4xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
    %7 = mhlo.copy %6 : tensor<2x4x1x1xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
    %c_1 = stablehlo.constant dense<8> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
    %11 = stablehlo.add %7, %10 : tensor<2x4x1x1xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %15 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %16 = stablehlo.compare LT, %5, %15 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %18 = stablehlo.add %5, %17 : tensor<2x4xi32>
    %19 = stablehlo.select %16, %18, %5 : tensor<2x4xi1>, tensor<2x4xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %21 = stablehlo.concatenate %14, %20, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
    return %21 : tensor<2x4x2x4x2xi32>
  }
  func.func private @wrapped_broadcast_computation(%arg0: tensor<f32>) -> tensor<16x256x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<16x256x8x8xf32>
    return %0 : tensor<16x256x8x8xf32>
  }
  func.func private @wrapped_broadcast_computation.1(%arg0: tensor<f32>) -> tensor<16x192x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<16x192x16x16xf32>
    return %0 : tensor<16x192x16x16xf32>
  }
  func.func private @wrapped_broadcast_computation.2(%arg0: tensor<f32>) -> tensor<16x128x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<16x128x32x32xf32>
    return %0 : tensor<16x128x32x32xf32>
  }
  func.func private @region_1.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<1x16x131072xf32>, %arg1: tensor<f32>) -> tensor<1x16x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    return %0 : tensor<1x16x128xf32>
  }
  func.func private @region_1.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<1x16x128xf32>, %arg1: tensor<f32>) -> tensor<1x16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    return %0 : tensor<1x16x4xf32>
  }
  func.func private @region_1.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation(%arg0: tensor<1x16x4xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_3.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<16x131072xf32>, %arg1: tensor<f32>) -> tensor<16x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    return %0 : tensor<16x4096xf32>
  }
  func.func private @region_5.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<16x1x4x256xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
    return %0 : tensor<16x1x4x8xf32>
  }
  func.func private @region_5.9.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.1(%arg0: tensor<16x1x4x8xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_6.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<16x1x4x256xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
    return %0 : tensor<16x1x4x8xf32>
  }
  func.func private @region_6.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.2(%arg0: tensor<16x1x4x8xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_7.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<f32>) -> tensor<16x128x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4x8xf32>
    return %0 : tensor<16x128x4x8xf32>
  }
  func.func private @region_9.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<1x16x131072xf32>, %arg1: tensor<f32>) -> tensor<1x16x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    return %0 : tensor<1x16x128xf32>
  }
  func.func private @region_9.13.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<1x16x128xf32>, %arg1: tensor<f32>) -> tensor<1x16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    return %0 : tensor<1x16x4xf32>
  }
  func.func private @region_9.13.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.3(%arg0: tensor<1x16x4xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_3.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<16x131072xf32>, %arg1: tensor<f32>) -> tensor<16x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    return %0 : tensor<16x4096xf32>
  }
  func.func private @region_12.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<1x16x131072xf32>, %arg1: tensor<f32>) -> tensor<1x16x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    return %0 : tensor<1x16x128xf32>
  }
  func.func private @region_12.17.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<1x16x128xf32>, %arg1: tensor<f32>) -> tensor<1x16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    return %0 : tensor<1x16x4xf32>
  }
  func.func private @region_12.17.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.4(%arg0: tensor<1x16x4xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_3.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.11(%arg0: tensor<16x131072xf32>, %arg1: tensor<f32>) -> tensor<16x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    return %0 : tensor<16x4096xf32>
  }
  func.func private @region_14.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.12(%arg0: tensor<16x1x4x256xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
    return %0 : tensor<16x1x4x8xf32>
  }
  func.func private @region_14.21.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.5(%arg0: tensor<16x1x4x8xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_15.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.13(%arg0: tensor<16x1x4x256xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
    return %0 : tensor<16x1x4x8xf32>
  }
  func.func private @region_15.22.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.6(%arg0: tensor<16x1x4x8xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_16.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.14(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<f32>) -> tensor<16x128x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4x8xf32>
    return %0 : tensor<16x128x4x8xf32>
  }
  func.func private @region_18.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.15(%arg0: tensor<1x16x131072xf32>, %arg1: tensor<f32>) -> tensor<1x16x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    return %0 : tensor<1x16x128xf32>
  }
  func.func private @region_18.24.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.16(%arg0: tensor<1x16x128xf32>, %arg1: tensor<f32>) -> tensor<1x16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    return %0 : tensor<1x16x4xf32>
  }
  func.func private @region_18.24.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.7(%arg0: tensor<1x16x4xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_3.3.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.17(%arg0: tensor<16x131072xf32>, %arg1: tensor<f32>) -> tensor<16x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    return %0 : tensor<16x4096xf32>
  }
  func.func private @region_21.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.18(%arg0: tensor<1x16x131072xf32>, %arg1: tensor<f32>) -> tensor<1x16x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    return %0 : tensor<1x16x128xf32>
  }
  func.func private @region_21.28.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.19(%arg0: tensor<1x16x128xf32>, %arg1: tensor<f32>) -> tensor<1x16x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    return %0 : tensor<1x16x4xf32>
  }
  func.func private @region_21.28.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.8(%arg0: tensor<1x16x4xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_3.3.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.20(%arg0: tensor<16x131072xf32>, %arg1: tensor<f32>) -> tensor<16x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    return %0 : tensor<16x4096xf32>
  }
  func.func private @region_23.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_scatter_computation(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<1024x2xi32>, %arg2: tensor<1024x16x128x1x1xf32>) -> tensor<16x128x32x32xf32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x16x128x1x1xf32>) -> tensor<16x128x32x32xf32>
    return %0 : tensor<16x128x32x32xf32>
  }
  func.func private @region_25.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.21(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_25.33.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.22(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_25.33.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.9(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.23(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.24(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.25(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_29.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.26(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    return %0 : tensor<16x1x4x2xf32>
  }
  func.func private @region_29.40.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.10(%arg0: tensor<16x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_30.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.27(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    return %0 : tensor<16x1x4x2xf32>
  }
  func.func private @region_30.41.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.11(%arg0: tensor<16x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_31.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.28(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x192x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
    return %0 : tensor<16x192x4x2xf32>
  }
  func.func private @region_31.42.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.12(%arg0: tensor<16x192x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x192x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    return %0 : tensor<16x192x4xf32>
  }
  func.func private @region_33.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.29(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_33.44.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.30(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_33.44.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.13(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.31(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.32(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.33(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_36.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.34(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_36.48.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.35(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_36.48.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.14(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.36(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.37(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.38(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_38.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.39(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    return %0 : tensor<16x1x4x2xf32>
  }
  func.func private @region_38.52.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.15(%arg0: tensor<16x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_39.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.40(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    return %0 : tensor<16x1x4x2xf32>
  }
  func.func private @region_39.53.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.16(%arg0: tensor<16x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_40.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.41(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x192x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
    return %0 : tensor<16x192x4x2xf32>
  }
  func.func private @region_40.54.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.17(%arg0: tensor<16x192x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x192x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    return %0 : tensor<16x192x4xf32>
  }
  func.func private @region_42.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.42(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_42.55.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.43(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_42.55.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.18(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.44(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.45(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.2.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.46(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_45.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.47(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_45.59.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.48(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_45.59.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.19(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.49(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.50(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.51(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_47.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.52(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    return %0 : tensor<16x1x4x2xf32>
  }
  func.func private @region_47.63.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.20(%arg0: tensor<16x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_48.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.53(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    return %0 : tensor<16x1x4x2xf32>
  }
  func.func private @region_48.64.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.21(%arg0: tensor<16x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_49.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.54(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x192x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
    return %0 : tensor<16x192x4x2xf32>
  }
  func.func private @region_49.65.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.22(%arg0: tensor<16x192x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x192x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    return %0 : tensor<16x192x4xf32>
  }
  func.func private @region_51.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.55(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_51.66.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.56(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_51.66.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.23(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.57(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.58(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.4.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.59(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_54.70(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.60(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_54.70.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.61(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_54.70.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.24(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.62(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.63(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.5.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.64(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_56.74(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.65(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    return %0 : tensor<16x1x4x2xf32>
  }
  func.func private @region_56.74.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.25(%arg0: tensor<16x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_57.75(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.66(%arg0: tensor<16x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    return %0 : tensor<16x1x4x2xf32>
  }
  func.func private @region_57.75.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.26(%arg0: tensor<16x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_58.76(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.67(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<16x192x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
    return %0 : tensor<16x192x4x2xf32>
  }
  func.func private @region_58.76.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.27(%arg0: tensor<16x192x4x2xf32>, %arg1: tensor<f32>) -> tensor<16x192x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    return %0 : tensor<16x192x4xf32>
  }
  func.func private @region_60.77(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.68(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_60.77.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.69(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_60.77.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.28(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.70(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.71(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.6.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.72(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_63.81(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.73(%arg0: tensor<1x16x49152xf32>, %arg1: tensor<f32>) -> tensor<1x16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    return %0 : tensor<1x16x48xf32>
  }
  func.func private @region_63.81.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.74(%arg0: tensor<1x16x48xf32>, %arg1: tensor<f32>) -> tensor<1x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    return %0 : tensor<1x16x2xf32>
  }
  func.func private @region_63.81.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.29(%arg0: tensor<1x16x2xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_27.35.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.75(%arg0: tensor<16x49152xf32>, %arg1: tensor<f32>) -> tensor<16x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    return %0 : tensor<16x1536xf32>
  }
  func.func private @region_27.35.clone.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.76(%arg0: tensor<16x1536xf32>, %arg1: tensor<f32>) -> tensor<16x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    return %0 : tensor<16x48xf32>
  }
  func.func private @region_27.35.clone.7.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.77(%arg0: tensor<16x48xf32>, %arg1: tensor<f32>) -> tensor<16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
  func.func private @region_65.85(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_scatter_computation.1(%arg0: tensor<16x192x16x16xf32>, %arg1: tensor<256x2xi32>, %arg2: tensor<256x16x192x1x1xf32>) -> tensor<16x192x16x16xf32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x192x16x16xf32>, tensor<256x2xi32>, tensor<256x16x192x1x1xf32>) -> tensor<16x192x16x16xf32>
    return %0 : tensor<16x192x16x16xf32>
  }
  func.func private @region_67.86(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.78(%arg0: tensor<1x16x16384xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_67.86.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.30(%arg0: tensor<1x16x16xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_69.88(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.79(%arg0: tensor<16x16384xf32>, %arg1: tensor<f32>) -> tensor<16x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %0 : tensor<16x512xf32>
  }
  func.func private @region_69.88.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.80(%arg0: tensor<16x512xf32>, %arg1: tensor<f32>) -> tensor<16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
  func.func private @region_72.94(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.31(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_75.97(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.81(%arg0: tensor<1x16x16384xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_75.97.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.32(%arg0: tensor<1x16x16xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_69.88.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.82(%arg0: tensor<16x16384xf32>, %arg1: tensor<f32>) -> tensor<16x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %0 : tensor<16x512xf32>
  }
  func.func private @region_69.88.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.83(%arg0: tensor<16x512xf32>, %arg1: tensor<f32>) -> tensor<16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
  func.func private @region_78.101(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.84(%arg0: tensor<1x16x16384xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_78.101.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.33(%arg0: tensor<1x16x16xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_69.88.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.85(%arg0: tensor<16x16384xf32>, %arg1: tensor<f32>) -> tensor<16x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %0 : tensor<16x512xf32>
  }
  func.func private @region_69.88.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.86(%arg0: tensor<16x512xf32>, %arg1: tensor<f32>) -> tensor<16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
  func.func private @region_81.106(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.34(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_84.108(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.87(%arg0: tensor<1x16x16384xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_84.108.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.35(%arg0: tensor<1x16x16xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_69.88.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.88(%arg0: tensor<16x16384xf32>, %arg1: tensor<f32>) -> tensor<16x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %0 : tensor<16x512xf32>
  }
  func.func private @region_69.88.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.89(%arg0: tensor<16x512xf32>, %arg1: tensor<f32>) -> tensor<16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
  func.func private @region_87.112(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.90(%arg0: tensor<1x16x16384xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_87.112.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.36(%arg0: tensor<1x16x16xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_69.88.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.91(%arg0: tensor<16x16384xf32>, %arg1: tensor<f32>) -> tensor<16x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %0 : tensor<16x512xf32>
  }
  func.func private @region_69.88.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.92(%arg0: tensor<16x512xf32>, %arg1: tensor<f32>) -> tensor<16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
  func.func private @region_90.117(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.37(%arg0: tensor<16x1x4x16xf32>, %arg1: tensor<f32>) -> tensor<16x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    return %0 : tensor<16x1x4xf32>
  }
  func.func private @region_93.119(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.93(%arg0: tensor<1x16x16384xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_93.119.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.38(%arg0: tensor<1x16x16xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_69.88.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.94(%arg0: tensor<16x16384xf32>, %arg1: tensor<f32>) -> tensor<16x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %0 : tensor<16x512xf32>
  }
  func.func private @region_69.88.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.95(%arg0: tensor<16x512xf32>, %arg1: tensor<f32>) -> tensor<16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
  func.func private @region_96.123(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.96(%arg0: tensor<1x16x16384xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_96.123.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.39(%arg0: tensor<1x16x16xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @region_69.88.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.97(%arg0: tensor<16x16384xf32>, %arg1: tensor<f32>) -> tensor<16x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %0 : tensor<16x512xf32>
  }
  func.func private @region_69.88.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.98(%arg0: tensor<16x512xf32>, %arg1: tensor<f32>) -> tensor<16x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
  func.func private @region_98.127(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_scatter_computation.2(%arg0: tensor<16x256x8x8xf32>, %arg1: tensor<64x2xi32>, %arg2: tensor<64x16x256x1x1xf32>) -> tensor<16x256x8x8xf32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x256x8x8xf32>, tensor<64x2xi32>, tensor<64x16x256x1x1xf32>) -> tensor<16x256x8x8xf32>
    return %0 : tensor<16x256x8x8xf32>
  }
  func.func @main(%arg0: tensor<32x3x3x3xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x32x1x1xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x1x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x1x1xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<128x64x1x1xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x1x3x3xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x128x1x1xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<256x128x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256x1x3x3xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<128x256x1x1xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<256x128x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256x1x3x3xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x256x1x1xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256x1x3x3xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<128x256x1x1xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<257x128x1x1xf32>, %arg46: tensor<257xf32>, %arg47: tensor<128x128x1x1xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x1x1xf32>, %arg52: tensor<256xf32>, %arg53: tensor<128x256x1x1xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<257x128x1x1xf32>, %arg58: tensor<257xf32>, %arg59: tensor<128x128x1x1xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<256x128x1x1xf32>, %arg64: tensor<256xf32>, %arg65: tensor<128x256x1x1xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<256x128x1x1xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<512x256x1x1xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512x1x3x3xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<384x512x1x1xf32>, %arg79: tensor<384xf32>, %arg80: tensor<384xf32>, %arg81: tensor<384x1x3x3xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<192x384x1x1xf32>, %arg85: tensor<192xf32>, %arg86: tensor<192xf32>, %arg87: tensor<385x192x1x1xf32>, %arg88: tensor<385xf32>, %arg89: tensor<192x192x1x1xf32>, %arg90: tensor<192xf32>, %arg91: tensor<192xf32>, %arg92: tensor<192xf32>, %arg93: tensor<384x192x1x1xf32>, %arg94: tensor<384xf32>, %arg95: tensor<192x384x1x1xf32>, %arg96: tensor<192xf32>, %arg97: tensor<192xf32>, %arg98: tensor<192xf32>, %arg99: tensor<385x192x1x1xf32>, %arg100: tensor<385xf32>, %arg101: tensor<192x192x1x1xf32>, %arg102: tensor<192xf32>, %arg103: tensor<192xf32>, %arg104: tensor<192xf32>, %arg105: tensor<384x192x1x1xf32>, %arg106: tensor<384xf32>, %arg107: tensor<192x384x1x1xf32>, %arg108: tensor<192xf32>, %arg109: tensor<192xf32>, %arg110: tensor<192xf32>, %arg111: tensor<385x192x1x1xf32>, %arg112: tensor<385xf32>, %arg113: tensor<192x192x1x1xf32>, %arg114: tensor<192xf32>, %arg115: tensor<192xf32>, %arg116: tensor<192xf32>, %arg117: tensor<384x192x1x1xf32>, %arg118: tensor<384xf32>, %arg119: tensor<192x384x1x1xf32>, %arg120: tensor<192xf32>, %arg121: tensor<192xf32>, %arg122: tensor<192xf32>, %arg123: tensor<385x192x1x1xf32>, %arg124: tensor<385xf32>, %arg125: tensor<192x192x1x1xf32>, %arg126: tensor<192xf32>, %arg127: tensor<192xf32>, %arg128: tensor<192xf32>, %arg129: tensor<384x192x1x1xf32>, %arg130: tensor<384xf32>, %arg131: tensor<192x384x1x1xf32>, %arg132: tensor<192xf32>, %arg133: tensor<192xf32>, %arg134: tensor<192xf32>, %arg135: tensor<384x192x1x1xf32>, %arg136: tensor<384xf32>, %arg137: tensor<384xf32>, %arg138: tensor<768x384x1x1xf32>, %arg139: tensor<768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x1x3x3xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<512x768x1x1xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<512x1x3x3xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512xf32>, %arg150: tensor<256x512x1x1xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<513x256x1x1xf32>, %arg154: tensor<513xf32>, %arg155: tensor<256x256x1x1xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<512x256x1x1xf32>, %arg160: tensor<512xf32>, %arg161: tensor<256x512x1x1xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<513x256x1x1xf32>, %arg166: tensor<513xf32>, %arg167: tensor<256x256x1x1xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<512x256x1x1xf32>, %arg172: tensor<512xf32>, %arg173: tensor<256x512x1x1xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<513x256x1x1xf32>, %arg178: tensor<513xf32>, %arg179: tensor<256x256x1x1xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512x256x1x1xf32>, %arg184: tensor<512xf32>, %arg185: tensor<256x512x1x1xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<512x256x1x1xf32>, %arg190: tensor<512xf32>, %arg191: tensor<512xf32>, %arg192: tensor<2x512xf32>, %arg193: tensor<2xf32>, %arg194: tensor<32xf32>, %arg195: tensor<32xf32>, %arg196: tensor<64xf32>, %arg197: tensor<64xf32>, %arg198: tensor<64xf32>, %arg199: tensor<64xf32>, %arg200: tensor<64xf32>, %arg201: tensor<64xf32>, %arg202: tensor<128xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<128xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128xf32>, %arg213: tensor<128xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<256xf32>, %arg222: tensor<256xf32>, %arg223: tensor<256xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<384xf32>, %arg229: tensor<384xf32>, %arg230: tensor<384xf32>, %arg231: tensor<384xf32>, %arg232: tensor<384xf32>, %arg233: tensor<384xf32>, %arg234: tensor<768xf32>, %arg235: tensor<768xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<512xf32>, %arg239: tensor<512xf32>, %arg240: tensor<512xf32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<512xf32>, %arg244: tensor<16x3x256x256xf32>) -> tensor<16x2xf32> {
    %0 = "mhlo.fusion"(%arg243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<512xf32>
      %664 = stablehlo.rsqrt %663 : tensor<512xf32>
      mhlo.return %664 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %662 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<16x256x8x8xf32>
      mhlo.return %662 : tensor<16x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<16x256x8x8xf32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %662 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %663 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %664 = stablehlo.multiply %662, %663 : tensor<4xi32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %666 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %667 = stablehlo.add %665, %666 : tensor<2x4xi32>
      %668 = mhlo.bitcast %667 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
      %669 = mhlo.copy %668 : tensor<2x4x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %670 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %671 = stablehlo.compare LT, %669, %670 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
      %c_2 = stablehlo.constant dense<8> : tensor<i32>
      %672 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %673 = stablehlo.add %669, %672 : tensor<2x4x1x1xi32>
      %674 = stablehlo.select %671, %673, %669 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
      %675 = mhlo.bitcast %674 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %677 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %678 = stablehlo.compare LT, %667, %677 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %679 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %680 = stablehlo.add %667, %679 : tensor<2x4xi32>
      %681 = stablehlo.select %678, %680, %667 : tensor<2x4xi1>, tensor<2x4xi32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %683 = stablehlo.concatenate %676, %682, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      mhlo.return %684 : tensor<64x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<64x2xi32>
    %3 = "mhlo.fusion"(%arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<512xf32>
      %664 = stablehlo.rsqrt %663 : tensor<512xf32>
      mhlo.return %664 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.fusion"(%arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<512xf32>
      %664 = stablehlo.rsqrt %663 : tensor<512xf32>
      mhlo.return %664 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %5 = "mhlo.fusion"(%arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<768xf32>
      %664 = stablehlo.rsqrt %663 : tensor<768xf32>
      mhlo.return %664 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %6 = "mhlo.fusion"(%arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<768xf32>
      %664 = stablehlo.rsqrt %663 : tensor<768xf32>
      mhlo.return %664 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %7 = "mhlo.fusion"(%arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<384xf32>
      %664 = stablehlo.rsqrt %663 : tensor<384xf32>
      mhlo.return %664 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %8 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %662 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<16x192x16x16xf32>
      mhlo.return %662 : tensor<16x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<16x192x16x16xf32>
    %9 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %662 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %663 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %664 = stablehlo.multiply %662, %663 : tensor<8xi32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %666 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %667 = stablehlo.add %665, %666 : tensor<2x8xi32>
      %668 = mhlo.bitcast %667 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
      %669 = mhlo.copy %668 : tensor<2x8x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %670 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %671 = stablehlo.compare LT, %669, %670 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
      %c_2 = stablehlo.constant dense<16> : tensor<i32>
      %672 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %673 = stablehlo.add %669, %672 : tensor<2x8x1x1xi32>
      %674 = stablehlo.select %671, %673, %669 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
      %675 = mhlo.bitcast %674 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %677 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %678 = stablehlo.compare LT, %667, %677 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %679 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %680 = stablehlo.add %667, %679 : tensor<2x8xi32>
      %681 = stablehlo.select %678, %680, %667 : tensor<2x8xi1>, tensor<2x8xi32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %683 = stablehlo.concatenate %676, %682, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      mhlo.return %684 : tensor<256x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<256x2xi32>
    %10 = "mhlo.fusion"(%arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<384xf32>
      %664 = stablehlo.rsqrt %663 : tensor<384xf32>
      mhlo.return %664 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %11 = "mhlo.fusion"(%arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<384xf32>
      %664 = stablehlo.rsqrt %663 : tensor<384xf32>
      mhlo.return %664 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %12 = "mhlo.fusion"(%arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<512xf32>
      %664 = stablehlo.rsqrt %663 : tensor<512xf32>
      mhlo.return %664 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %13 = "mhlo.fusion"(%arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<512xf32>
      %664 = stablehlo.rsqrt %663 : tensor<512xf32>
      mhlo.return %664 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %14 = "mhlo.fusion"(%arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<256xf32>
      %664 = stablehlo.rsqrt %663 : tensor<256xf32>
      mhlo.return %664 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %15 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %662 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<16x128x32x32xf32>
      mhlo.return %662 : tensor<16x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<16x128x32x32xf32>
    %16 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %662 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %663 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %664 = stablehlo.add %662, %663 : tensor<2x16xi32>
      %665 = mhlo.bitcast %664 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
      %666 = mhlo.copy %665 : tensor<2x16x1x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %667 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %668 = stablehlo.compare LT, %666, %667 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
      %c_1 = stablehlo.constant dense<32> : tensor<i32>
      %669 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %670 = stablehlo.add %666, %669 : tensor<2x16x1x1xi32>
      %671 = stablehlo.select %668, %670, %666 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
      %672 = mhlo.bitcast %671 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
      %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %674 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %675 = stablehlo.compare LT, %664, %674 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %676 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %677 = stablehlo.add %664, %676 : tensor<2x16xi32>
      %678 = stablehlo.select %675, %677, %664 : tensor<2x16xi1>, tensor<2x16xi32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %680 = stablehlo.concatenate %673, %679, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      mhlo.return %681 : tensor<1024x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<1024x2xi32>
    %17 = "mhlo.fusion"(%arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<256xf32>
      %664 = stablehlo.rsqrt %663 : tensor<256xf32>
      mhlo.return %664 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %18 = "mhlo.fusion"(%arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<256xf32>
      %664 = stablehlo.rsqrt %663 : tensor<256xf32>
      mhlo.return %664 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %19 = "mhlo.fusion"(%arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<256xf32>
      %664 = stablehlo.rsqrt %663 : tensor<256xf32>
      mhlo.return %664 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %20 = "mhlo.fusion"(%arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<256xf32>
      %664 = stablehlo.rsqrt %663 : tensor<256xf32>
      mhlo.return %664 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %21 = "mhlo.fusion"(%arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<128xf32>
      %664 = stablehlo.rsqrt %663 : tensor<128xf32>
      mhlo.return %664 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %22 = "mhlo.fusion"(%arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<256xf32>
      %664 = stablehlo.rsqrt %663 : tensor<256xf32>
      mhlo.return %664 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %23 = "mhlo.fusion"(%arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<256xf32>
      %664 = stablehlo.rsqrt %663 : tensor<256xf32>
      mhlo.return %664 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %24 = "mhlo.fusion"(%arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<128xf32>
      %664 = stablehlo.rsqrt %663 : tensor<128xf32>
      mhlo.return %664 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %25 = "mhlo.fusion"(%arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<128xf32>
      %664 = stablehlo.rsqrt %663 : tensor<128xf32>
      mhlo.return %664 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %26 = "mhlo.fusion"(%arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<128xf32>
      %664 = stablehlo.rsqrt %663 : tensor<128xf32>
      mhlo.return %664 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %27 = "mhlo.fusion"(%arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<64xf32>
      %664 = stablehlo.rsqrt %663 : tensor<64xf32>
      mhlo.return %664 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %28 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<64xf32>
      %664 = stablehlo.rsqrt %663 : tensor<64xf32>
      mhlo.return %664 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %29 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<64xf32>
      %664 = stablehlo.rsqrt %663 : tensor<64xf32>
      mhlo.return %664 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %30 = "mhlo.fusion"(%arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<32xf32>
      %664 = stablehlo.rsqrt %663 : tensor<32xf32>
      mhlo.return %664 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %31 = "mhlo.fusion"(%arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x3x256x256xf32>):
      %662 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,256,3]{2,1,3,0}"} : (tensor<16x3x256x256xf32>) -> tensor<16x256x256x3xf32>
      %663 = mhlo.copy %662 : tensor<16x256x256x3xf32>
      mhlo.return %663 : tensor<16x256x256x3xf32>
    }) {output_operand_aliasing = []} : (tensor<16x3x256x256xf32>) -> tensor<16x256x256x3xf32>
    %32 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32x3x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,3,32]{1,0,2,3}"} : (tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32>
      %663 = mhlo.copy %662 : tensor<3x3x3x32xf32>
      mhlo.return %663 : tensor<3x3x3x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32>
    %33 = stablehlo.convolution(%31, %32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x256x256x3xf32>, tensor<3x3x3x32xf32>) -> tensor<16x128x128x32xf32>
    %34 = "mhlo.fusion"(%arg2, %arg1, %30, %33, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>, %arg246: tensor<32xf32>, %arg247: tensor<32xf32>, %arg248: tensor<16x128x128x32xf32>, %arg249: tensor<32xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x128x128x32xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x128x128x32xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x128x128x32xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x128x128x32xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %671 = mhlo.copy %670 : tensor<16x32x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x32x128x128xf32>
      %673 = stablehlo.negate %671 : tensor<16x32x128x128xf32>
      %674 = stablehlo.exponential %673 : tensor<16x32x128x128xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x32x128x128xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x32x128x128xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x32x128x128xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,32]{2,1,3,0}"} : (tensor<16x32x128x128xf32>) -> tensor<16x128x128x32xf32>
      %679 = mhlo.copy %678 : tensor<16x128x128x32xf32>
      mhlo.return %679 : tensor<16x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x128x128x32xf32>, tensor<32xf32>) -> tensor<16x128x128x32xf32>
    %35 = mhlo.bitcast %34 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x32xf32>) -> tensor<262144x32xf32>
    %36 = "mhlo.fusion"(%arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x32x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,32,64]{1,0,2,3}"} : (tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32>
      %663 = mhlo.copy %662 : tensor<1x1x32x64xf32>
      mhlo.return %663 : tensor<1x1x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32>
    %37 = mhlo.bitcast %36 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x32x64xf32>) -> tensor<32x64xf32>
    %38 = stablehlo.dot %35, %37, precision = [DEFAULT, DEFAULT] : (tensor<262144x32xf32>, tensor<32x64xf32>) -> tensor<262144x64xf32>
    %39 = mhlo.bitcast %38 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<16x128x128x64xf32>
    %40 = "mhlo.fusion"(%arg5, %arg4, %29, %39, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<16x128x128x64xf32>, %arg249: tensor<64xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x128x128x64xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x128x128x64xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x128x128x64xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x128x128x64xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,128,128]{1,3,2,0}"} : (tensor<16x128x128x64xf32>) -> tensor<16x64x128x128xf32>
      %671 = mhlo.copy %670 : tensor<16x64x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x64x128x128xf32>
      %673 = stablehlo.negate %671 : tensor<16x64x128x128xf32>
      %674 = stablehlo.exponential %673 : tensor<16x64x128x128xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x64x128x128xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x64x128x128xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x64x128x128xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,64]{2,1,3,0}"} : (tensor<16x64x128x128xf32>) -> tensor<16x128x128x64xf32>
      %679 = mhlo.copy %678 : tensor<16x128x128x64xf32>
      mhlo.return %679 : tensor<16x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x128x128x64xf32>, tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %41 = "mhlo.fusion"(%arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,64]{1,0,2,3}"} : (tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x64xf32>
      mhlo.return %663 : tensor<3x3x1x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32>
    %42 = stablehlo.convolution(%40, %41) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 64 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x128x128x64xf32>, tensor<3x3x1x64xf32>) -> tensor<16x128x128x64xf32>
    %43 = "mhlo.fusion"(%arg8, %arg7, %28, %42, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<16x128x128x64xf32>, %arg249: tensor<64xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x128x128x64xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x128x128x64xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x128x128x64xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x128x128x64xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,128,128]{1,3,2,0}"} : (tensor<16x128x128x64xf32>) -> tensor<16x64x128x128xf32>
      %671 = mhlo.copy %670 : tensor<16x64x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x64x128x128xf32>
      %673 = stablehlo.negate %671 : tensor<16x64x128x128xf32>
      %674 = stablehlo.exponential %673 : tensor<16x64x128x128xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x64x128x128xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x64x128x128xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x64x128x128xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,64]{2,1,3,0}"} : (tensor<16x64x128x128xf32>) -> tensor<16x128x128x64xf32>
      %679 = mhlo.copy %678 : tensor<16x128x128x64xf32>
      mhlo.return %679 : tensor<16x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x128x128x64xf32>, tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %44 = mhlo.bitcast %43 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x64xf32>) -> tensor<262144x64xf32>
    %45 = "mhlo.fusion"(%arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x64x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
      %663 = mhlo.copy %662 : tensor<1x1x64x64xf32>
      mhlo.return %663 : tensor<1x1x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
    %46 = mhlo.bitcast %45 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x64xf32>) -> tensor<64x64xf32>
    %47 = stablehlo.dot %44, %46, precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %48 = mhlo.bitcast %47 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<16x128x128x64xf32>
    %49 = "mhlo.fusion"(%arg11, %arg10, %27, %48, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<16x128x128x64xf32>, %arg249: tensor<64xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x128x128x64xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x128x128x64xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x128x128x64xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<16x128x128x64xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x128x128x64xf32>
      mhlo.return %669 : tensor<16x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x128x128x64xf32>, tensor<64xf32>) -> tensor<16x128x128x64xf32>
    %50 = mhlo.bitcast %49 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x64xf32>) -> tensor<262144x64xf32>
    %51 = "mhlo.fusion"(%arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x64x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32>
      %663 = mhlo.copy %662 : tensor<1x1x64x128xf32>
      mhlo.return %663 : tensor<1x1x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32>
    %52 = mhlo.bitcast %51 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x128xf32>) -> tensor<64x128xf32>
    %53 = stablehlo.dot %50, %52, precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x128xf32>) -> tensor<262144x128xf32>
    %54 = mhlo.bitcast %53 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x128xf32>) -> tensor<16x128x128x128xf32>
    %55 = "mhlo.fusion"(%arg14, %arg13, %26, %54, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<16x128x128x128xf32>, %arg249: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x128x128x128xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x128x128x128xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x128x128x128xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x128x128x128xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{1,3,2,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
      %671 = mhlo.copy %670 : tensor<16x128x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x128x128x128xf32>
      %673 = stablehlo.negate %671 : tensor<16x128x128x128xf32>
      %674 = stablehlo.exponential %673 : tensor<16x128x128x128xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x128x128x128xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x128x128x128xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x128x128x128xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{2,1,3,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
      %679 = mhlo.copy %678 : tensor<16x128x128x128xf32>
      mhlo.return %679 : tensor<16x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x128x128xf32>, tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %56 = "mhlo.fusion"(%arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x128xf32>
      mhlo.return %663 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %57 = stablehlo.convolution(%55, %56) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<16x64x64x128xf32>
    %58 = "mhlo.fusion"(%arg17, %arg16, %25, %57, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<16x64x64x128xf32>, %arg249: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x64x64x128xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x64x64x128xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x64x64x128xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x64x64x128xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,64,64]{1,3,2,0}"} : (tensor<16x64x64x128xf32>) -> tensor<16x128x64x64xf32>
      %671 = mhlo.copy %670 : tensor<16x128x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x128x64x64xf32>
      %673 = stablehlo.negate %671 : tensor<16x128x64x64xf32>
      %674 = stablehlo.exponential %673 : tensor<16x128x64x64xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x128x64x64xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x128x64x64xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x128x64x64xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,128]{2,1,3,0}"} : (tensor<16x128x64x64xf32>) -> tensor<16x64x64x128xf32>
      %679 = mhlo.copy %678 : tensor<16x64x64x128xf32>
      mhlo.return %679 : tensor<16x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x64x64x128xf32>, tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x128xf32>) -> tensor<65536x128xf32>
    %60 = "mhlo.fusion"(%arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x128xf32>
      mhlo.return %663 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %62 = stablehlo.dot %59, %61, precision = [DEFAULT, DEFAULT] : (tensor<65536x128xf32>, tensor<128x128xf32>) -> tensor<65536x128xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x128xf32>) -> tensor<16x64x64x128xf32>
    %64 = "mhlo.fusion"(%arg20, %arg19, %24, %63, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<16x64x64x128xf32>, %arg249: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x64x64x128xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x64x64x128xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x64x64x128xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x64x64x128xf32>
      mhlo.return %669 : tensor<16x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x64x64x128xf32>, tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x128xf32>) -> tensor<65536x128xf32>
    %66 = "mhlo.fusion"(%arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x256xf32>
      mhlo.return %663 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %67 = mhlo.bitcast %66 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %68 = stablehlo.dot %65, %67, precision = [DEFAULT, DEFAULT] : (tensor<65536x128xf32>, tensor<128x256xf32>) -> tensor<65536x256xf32>
    %69 = mhlo.bitcast %68 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<16x64x64x256xf32>
    %70 = "mhlo.fusion"(%arg23, %arg22, %23, %69, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x64x64x256xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x64x64x256xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x64x64x256xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x64x64x256xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x64x64x256xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
      %671 = mhlo.copy %670 : tensor<16x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
      %673 = stablehlo.negate %671 : tensor<16x256x64x64xf32>
      %674 = stablehlo.exponential %673 : tensor<16x256x64x64xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x256x64x64xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x256x64x64xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x256x64x64xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,256]{2,1,3,0}"} : (tensor<16x256x64x64xf32>) -> tensor<16x64x64x256xf32>
      %679 = mhlo.copy %678 : tensor<16x64x64x256xf32>
      mhlo.return %679 : tensor<16x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x64x64x256xf32>, tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %71 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x256xf32>
      mhlo.return %663 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %72 = stablehlo.convolution(%70, %71) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<16x64x64x256xf32>
    %73 = "mhlo.fusion"(%arg26, %arg25, %22, %72, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x64x64x256xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x64x64x256xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x64x64x256xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x64x64x256xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x64x64x256xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
      %671 = mhlo.copy %670 : tensor<16x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
      %673 = stablehlo.negate %671 : tensor<16x256x64x64xf32>
      %674 = stablehlo.exponential %673 : tensor<16x256x64x64xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x256x64x64xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x256x64x64xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x256x64x64xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,256]{2,1,3,0}"} : (tensor<16x256x64x64xf32>) -> tensor<16x64x64x256xf32>
      %679 = mhlo.copy %678 : tensor<16x64x64x256xf32>
      mhlo.return %679 : tensor<16x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x64x64x256xf32>, tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %74 = mhlo.bitcast %73 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x256xf32>) -> tensor<65536x256xf32>
    %75 = "mhlo.fusion"(%arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x128xf32>
      mhlo.return %663 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %76 = mhlo.bitcast %75 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %77 = stablehlo.dot %74, %76, precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x128xf32>) -> tensor<65536x128xf32>
    %78 = mhlo.bitcast %77 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x128xf32>) -> tensor<16x64x64x128xf32>
    %79 = "mhlo.fusion"(%arg29, %arg28, %21, %78, %arg212, %arg20, %arg19, %24, %63, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<16x64x64x128xf32>, %arg249: tensor<128xf32>, %arg250: tensor<128xf32>, %arg251: tensor<128xf32>, %arg252: tensor<128xf32>, %arg253: tensor<16x64x64x128xf32>, %arg254: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg254, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %663 = stablehlo.subtract %arg253, %662 : tensor<16x64x64x128xf32>
      %664 = stablehlo.broadcast_in_dim %arg252, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x64x64x128xf32>
      %666 = stablehlo.broadcast_in_dim %arg251, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x64x64x128xf32>
      %668 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x64x64x128xf32>
      %670 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %671 = stablehlo.subtract %arg248, %670 : tensor<16x64x64x128xf32>
      %672 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %673 = stablehlo.multiply %671, %672 : tensor<16x64x64x128xf32>
      %674 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %675 = stablehlo.multiply %673, %674 : tensor<16x64x64x128xf32>
      %676 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<16x64x64x128xf32>
      %677 = stablehlo.add %675, %676 : tensor<16x64x64x128xf32>
      %678 = stablehlo.add %669, %677 : tensor<16x64x64x128xf32>
      mhlo.return %678 : tensor<16x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x64x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x64x64x128xf32>, tensor<128xf32>) -> tensor<16x64x64x128xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x128xf32>) -> tensor<65536x128xf32>
    %81 = "mhlo.fusion"(%arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x256xf32>
      mhlo.return %663 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %83 = stablehlo.dot %80, %82, precision = [DEFAULT, DEFAULT] : (tensor<65536x128xf32>, tensor<128x256xf32>) -> tensor<65536x256xf32>
    %84 = mhlo.bitcast %83 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<16x64x64x256xf32>
    %85 = "mhlo.fusion"(%arg32, %arg31, %20, %84, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x64x64x256xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x64x64x256xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x64x64x256xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x64x64x256xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x64x64x256xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
      %671 = mhlo.copy %670 : tensor<16x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
      %673 = stablehlo.negate %671 : tensor<16x256x64x64xf32>
      %674 = stablehlo.exponential %673 : tensor<16x256x64x64xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x256x64x64xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x256x64x64xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x256x64x64xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,256]{2,1,3,0}"} : (tensor<16x256x64x64xf32>) -> tensor<16x64x64x256xf32>
      %679 = mhlo.copy %678 : tensor<16x64x64x256xf32>
      mhlo.return %679 : tensor<16x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x64x64x256xf32>, tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %86 = "mhlo.fusion"(%arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x256xf32>
      mhlo.return %663 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %87 = stablehlo.convolution(%85, %86) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<16x32x32x256xf32>
    %88 = "mhlo.fusion"(%arg35, %arg34, %19, %87, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x32x32x256xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x32x32x256xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x32x32x256xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x32x32x256xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x32x32x256xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,32,32]{1,3,2,0}"} : (tensor<16x32x32x256xf32>) -> tensor<16x256x32x32xf32>
      %671 = mhlo.copy %670 : tensor<16x256x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x32x32xf32>
      %673 = stablehlo.negate %671 : tensor<16x256x32x32xf32>
      %674 = stablehlo.exponential %673 : tensor<16x256x32x32xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x256x32x32xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x256x32x32xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x256x32x32xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,256]{2,1,3,0}"} : (tensor<16x256x32x32xf32>) -> tensor<16x32x32x256xf32>
      %679 = mhlo.copy %678 : tensor<16x32x32x256xf32>
      mhlo.return %679 : tensor<16x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x32x32x256xf32>, tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %89 = mhlo.bitcast %88 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x32x256xf32>) -> tensor<16384x256xf32>
    %90 = "mhlo.fusion"(%arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x256xf32>
      mhlo.return %663 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %91 = mhlo.bitcast %90 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %92 = stablehlo.dot %89, %91, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %93 = mhlo.bitcast %92 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<16x32x32x256xf32>
    %94 = "mhlo.fusion"(%arg38, %arg37, %18, %93, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x32x32x256xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x32x32x256xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x32x32x256xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x32x32x256xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x32x32x256xf32>
      mhlo.return %669 : tensor<16x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x32x32x256xf32>, tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %95 = "mhlo.fusion"(%arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x256xf32>
      mhlo.return %663 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %96 = stablehlo.convolution(%94, %95) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x32x32x256xf32>, tensor<3x3x1x256xf32>) -> tensor<16x32x32x256xf32>
    %97 = "mhlo.fusion"(%arg41, %arg40, %17, %96, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x32x32x256xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x32x32x256xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x32x32x256xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x32x32x256xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x32x32x256xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,32,32]{1,3,2,0}"} : (tensor<16x32x32x256xf32>) -> tensor<16x256x32x32xf32>
      %671 = mhlo.copy %670 : tensor<16x256x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x32x32xf32>
      %673 = stablehlo.negate %671 : tensor<16x256x32x32xf32>
      %674 = stablehlo.exponential %673 : tensor<16x256x32x32xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x256x32x32xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x256x32x32xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x256x32x32xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,256]{2,1,3,0}"} : (tensor<16x256x32x32xf32>) -> tensor<16x32x32x256xf32>
      %679 = mhlo.copy %678 : tensor<16x32x32x256xf32>
      mhlo.return %679 : tensor<16x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x32x32x256xf32>, tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %98 = mhlo.bitcast %97 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x32x256xf32>) -> tensor<16384x256xf32>
    %99 = "mhlo.fusion"(%arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x128xf32>
      mhlo.return %663 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %100 = mhlo.bitcast %99 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %101 = stablehlo.dot %98, %100, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x128xf32>) -> tensor<16384x128xf32>
    %102 = mhlo.bitcast %101 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<16x32x32x128xf32>
    %103 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %662 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %663 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %664 = stablehlo.add %662, %663 : tensor<2x16xi32>
      %665 = mhlo.bitcast %664 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
      %666 = mhlo.copy %665 : tensor<2x16x1x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %667 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %668 = stablehlo.compare LT, %666, %667 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
      %c_1 = stablehlo.constant dense<32> : tensor<i32>
      %669 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %670 = stablehlo.add %666, %669 : tensor<2x16x1x1xi32>
      %671 = stablehlo.select %668, %670, %666 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
      %672 = mhlo.bitcast %671 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
      %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %674 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %675 = stablehlo.compare LT, %664, %674 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %676 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %677 = stablehlo.add %664, %676 : tensor<2x16xi32>
      %678 = stablehlo.select %675, %677, %664 : tensor<2x16xi1>, tensor<2x16xi32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %680 = stablehlo.concatenate %673, %679, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      mhlo.return %680 : tensor<2x16x2x16x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x16x2x16x2xi32>
    %104 = "mhlo.fusion"(%102, %103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x32x32x128xf32>, %arg246: tensor<2x16x2x16x2xi32>):
      %662 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,32,32]{1,3,2,0}"} : (tensor<16x32x32x128xf32>) -> tensor<16x128x32x32xf32>
      %663 = mhlo.copy %662 : tensor<16x128x32x32xf32>
      %664 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %665 = "stablehlo.gather"(%663, %664) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 128, 1, 1>}> : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x16x128x1x1xf32>
      mhlo.return %665 : tensor<1024x16x128x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<16x32x32x128xf32>, tensor<2x16x2x16x2xi32>) -> tensor<1024x16x128x1x1xf32>
    %105 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x16x128x1x1xf32>):
      %662 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
      %663 = stablehlo.transpose %662, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
      %664 = mhlo.copy %663 : tensor<16x128x2x2x16x16xf32>
      %665 = mhlo.bitcast %664 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<1x16x131072xf32>
      mhlo.return %665 : tensor<1x16x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x16x128x1x1xf32>) -> tensor<1x16x131072xf32>
    %106 = "mhlo.fusion"(%105, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
      mhlo.return %662 : tensor<1x16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x128xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
      mhlo.return %662 : tensor<1x16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x4xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %109 = "mhlo.fusion"(%108, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<1024x16x128x1x1xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
      %663 = stablehlo.transpose %662, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
      %664 = mhlo.copy %663 : tensor<16x128x2x2x16x16xf32>
      %665 = mhlo.bitcast %664 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %667 = stablehlo.multiply %arg245, %666 : tensor<16xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %669 = stablehlo.subtract %665, %668 : tensor<16x131072xf32>
      %670 = stablehlo.multiply %669, %669 : tensor<16x131072xf32>
      mhlo.return %670 : tensor<16x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<1024x16x128x1x1xf32>) -> tensor<16x131072xf32>
    %110 = "mhlo.fusion"(%109, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %662 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %111 = "mhlo.fusion"(%110, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x4096xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
      mhlo.return %662 : tensor<16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    %112 = "mhlo.fusion"(%111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4xf32>) -> tensor<16xf32>
    %113 = "mhlo.fusion"(%arg43, %112, %108, %104, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<1024x16x128x1x1xf32>, %arg249: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
      %664 = stablehlo.transpose %663, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
      %665 = mhlo.copy %664 : tensor<16x128x2x2x16x16xf32>
      %666 = mhlo.bitcast %665 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %668 = stablehlo.multiply %arg247, %667 : tensor<16xf32>
      %669 = stablehlo.broadcast_in_dim %668, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %670 = stablehlo.subtract %666, %669 : tensor<16x131072xf32>
      %671 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x131072xf32>
      %673 = mhlo.bitcast %672 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
      %674 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %675 = stablehlo.multiply %673, %674 : tensor<16x128x4x256xf32>
      %676 = stablehlo.add %662, %675 : tensor<16x128x4x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
      %678 = mhlo.copy %677 : tensor<16x4x256x128xf32>
      mhlo.return %678 : tensor<16x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<16xf32>, tensor<16xf32>, tensor<1024x16x128x1x1xf32>, tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %114 = mhlo.bitcast %113 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x256x128xf32>) -> tensor<16384x128xf32>
    %115 = "mhlo.fusion"(%arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<257x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x257xf32>
      mhlo.return %663 : tensor<1x1x128x257xf32>
    }) {output_operand_aliasing = []} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %116 = mhlo.bitcast %115 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x257xf32>) -> tensor<128x257xf32>
    %117 = stablehlo.dot %114, %116, precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<128x257xf32>) -> tensor<16384x257xf32>
    %118 = mhlo.bitcast %117 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x257xf32>) -> tensor<16x4x256x257xf32>
    %119 = "mhlo.fusion"(%118, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x256x257xf32>, %arg246: tensor<257xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x256x257xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x257x4x256xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
      mhlo.return %666 : tensor<16x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x256x257xf32>, tensor<257xf32>) -> tensor<16x1x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %120 = "mhlo.fusion"(%119, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x256xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
      mhlo.return %662 : tensor<16x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
    %121 = "mhlo.fusion"(%120, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x8xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %122 = "mhlo.fusion"(%121, %118, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x256x257xf32>, %arg247: tensor<257xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x257xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x257x4x256xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x256xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x256xf32>
      mhlo.return %672 : tensor<16x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x256x257xf32>, tensor<257xf32>) -> tensor<16x1x4x256xf32>
    %123 = "mhlo.fusion"(%122, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x256xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
      mhlo.return %662 : tensor<16x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
    %124 = "mhlo.fusion"(%123, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x8xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %125 = "mhlo.fusion"(%122, %124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x256xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x256xf32>
      mhlo.return %664 : tensor<16x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x256xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x256xf32>
    %126 = "mhlo.fusion"(%125, %118, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x256xf32>, %arg246: tensor<16x4x256x257xf32>, %arg247: tensor<257xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x257xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x257x4x256xf32>
      %666 = stablehlo.slice %665 [0:16, 1:129, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x256xf32>) -> tensor<16x128x4x256xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x128x4x256xf32>
      mhlo.return %669 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x256xf32>, tensor<16x4x256x257xf32>, tensor<257xf32>) -> tensor<16x128x4x256xf32>
    %127 = "mhlo.fusion"(%126, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4x8xf32>
      mhlo.return %662 : tensor<16x128x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4x8xf32>
    %128 = "mhlo.fusion"(%127, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x128x4x8xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x128x4x8xf32>, tensor<f32>) -> tensor<16x128x4xf32>
      mhlo.return %662 : tensor<16x128x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x8xf32>, tensor<f32>) -> tensor<16x128x4xf32>
    %129 = "mhlo.fusion"(%128, %118, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x4xf32>, %arg246: tensor<16x4x256x257xf32>, %arg247: tensor<257xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x257xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x257x4x256xf32>
      %666 = stablehlo.slice %665 [0:16, 129:257, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x128x4x256xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x128x4x256xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x128x4xf32>) -> tensor<16x128x4x256xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x128x4x256xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
      %672 = mhlo.copy %671 : tensor<16x4x256x128xf32>
      mhlo.return %672 : tensor<16x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4xf32>, tensor<16x4x256x257xf32>, tensor<257xf32>) -> tensor<16x4x256x128xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x256x128xf32>) -> tensor<16384x128xf32>
    %131 = "mhlo.fusion"(%arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x128xf32>
      mhlo.return %663 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %132 = mhlo.bitcast %131 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %133 = stablehlo.dot %130, %132, precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<128x128xf32>) -> tensor<16384x128xf32>
    %134 = mhlo.bitcast %133 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<16x4x256x128xf32>
    %135 = "mhlo.fusion"(%134, %arg48, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x256x128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1024x16x128x1x1xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x256x128xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x128x4x256xf32>
      %666 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
      %667 = stablehlo.transpose %666, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
      %668 = mhlo.copy %667 : tensor<16x128x2x2x16x16xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x128x4x256xf32>
      %670 = stablehlo.add %665, %669 : tensor<16x128x4x256xf32>
      mhlo.return %670 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x256x128xf32>, tensor<128xf32>, tensor<1024x16x128x1x1xf32>) -> tensor<16x128x4x256xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
    %137 = "mhlo.fusion"(%136, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
      mhlo.return %662 : tensor<1x16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    %138 = "mhlo.fusion"(%137, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x128xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
      mhlo.return %662 : tensor<1x16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    %139 = "mhlo.fusion"(%138, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x4xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %140 = "mhlo.fusion"(%139, %135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x128x4x256xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x131072xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x131072xf32>
      mhlo.return %667 : tensor<16x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %141 = "mhlo.fusion"(%140, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %662 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x4096xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
      mhlo.return %662 : tensor<16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4xf32>) -> tensor<16xf32>
    %144 = "mhlo.fusion"(%arg49, %143, %139, %135, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x4x256xf32>, %arg249: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x131072xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x131072xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x128x4x256xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x128x4x256xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
      %675 = mhlo.copy %674 : tensor<16x4x256x128xf32>
      mhlo.return %675 : tensor<16x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %145 = mhlo.bitcast %144 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x256x128xf32>) -> tensor<16384x128xf32>
    %146 = "mhlo.fusion"(%arg51) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x256xf32>
      mhlo.return %663 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %147 = mhlo.bitcast %146 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %148 = stablehlo.dot %145, %147, precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<128x256xf32>) -> tensor<16384x256xf32>
    %149 = mhlo.bitcast %148 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<16x4x256x256xf32>
    %150 = "mhlo.fusion"(%149, %arg52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x256x256xf32>, %arg246: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x4x256x256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x256x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,256]{1,3,2,0}"} : (tensor<16x4x256x256xf32>) -> tensor<16x256x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x4x256xf32>
      %667 = stablehlo.negate %665 : tensor<16x256x4x256xf32>
      %668 = stablehlo.exponential %667 : tensor<16x256x4x256xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x256x4x256xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x256x4x256xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x256x4x256xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,256]{2,1,3,0}"} : (tensor<16x256x4x256xf32>) -> tensor<16x4x256x256xf32>
      %673 = mhlo.copy %672 : tensor<16x4x256x256xf32>
      mhlo.return %673 : tensor<16x4x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x256x256xf32>, tensor<256xf32>) -> tensor<16x4x256x256xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x256x256xf32>) -> tensor<16384x256xf32>
    %152 = "mhlo.fusion"(%arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x128xf32>
      mhlo.return %663 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %153 = mhlo.bitcast %152 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %154 = stablehlo.dot %151, %153, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x128xf32>) -> tensor<16384x128xf32>
    %155 = mhlo.bitcast %154 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<16x4x256x128xf32>
    %156 = "mhlo.fusion"(%135, %155, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<16x4x256x128xf32>, %arg247: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x128xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x128x4x256xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x128x4x256xf32>
      mhlo.return %666 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<16x4x256x128xf32>, tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
    %158 = "mhlo.fusion"(%157, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
      mhlo.return %662 : tensor<1x16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    %159 = "mhlo.fusion"(%158, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x128xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
      mhlo.return %662 : tensor<1x16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    %160 = "mhlo.fusion"(%159, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x4xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %161 = "mhlo.fusion"(%160, %156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x128x4x256xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x131072xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x131072xf32>
      mhlo.return %667 : tensor<16x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %162 = "mhlo.fusion"(%161, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %662 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %163 = "mhlo.fusion"(%162, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x4096xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
      mhlo.return %662 : tensor<16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4xf32>) -> tensor<16xf32>
    %165 = "mhlo.fusion"(%arg55, %164, %160, %156, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x4x256xf32>, %arg249: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x131072xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x131072xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x128x4x256xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x128x4x256xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
      %675 = mhlo.copy %674 : tensor<16x4x256x128xf32>
      mhlo.return %675 : tensor<16x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %166 = mhlo.bitcast %165 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x256x128xf32>) -> tensor<16384x128xf32>
    %167 = "mhlo.fusion"(%arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<257x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x257xf32>
      mhlo.return %663 : tensor<1x1x128x257xf32>
    }) {output_operand_aliasing = []} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %168 = mhlo.bitcast %167 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x257xf32>) -> tensor<128x257xf32>
    %169 = stablehlo.dot %166, %168, precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<128x257xf32>) -> tensor<16384x257xf32>
    %170 = mhlo.bitcast %169 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x257xf32>) -> tensor<16x4x256x257xf32>
    %171 = "mhlo.fusion"(%170, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x256x257xf32>, %arg246: tensor<257xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x256x257xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x257x4x256xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
      mhlo.return %666 : tensor<16x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x256x257xf32>, tensor<257xf32>) -> tensor<16x1x4x256xf32>
    %172 = "mhlo.fusion"(%171, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x256xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
      mhlo.return %662 : tensor<16x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
    %173 = "mhlo.fusion"(%172, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x8xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %174 = "mhlo.fusion"(%173, %170, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x256x257xf32>, %arg247: tensor<257xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x257xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x257x4x256xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x256xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x256xf32>
      mhlo.return %672 : tensor<16x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x256x257xf32>, tensor<257xf32>) -> tensor<16x1x4x256xf32>
    %175 = "mhlo.fusion"(%174, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x256xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
      mhlo.return %662 : tensor<16x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x256xf32>, tensor<f32>) -> tensor<16x1x4x8xf32>
    %176 = "mhlo.fusion"(%175, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x8xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x8xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %177 = "mhlo.fusion"(%174, %176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x256xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x256xf32>
      mhlo.return %664 : tensor<16x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x256xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x256xf32>
    %178 = "mhlo.fusion"(%177, %170, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x256xf32>, %arg246: tensor<16x4x256x257xf32>, %arg247: tensor<257xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x257xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x257x4x256xf32>
      %666 = stablehlo.slice %665 [0:16, 1:129, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x256xf32>) -> tensor<16x128x4x256xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x128x4x256xf32>
      mhlo.return %669 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x256xf32>, tensor<16x4x256x257xf32>, tensor<257xf32>) -> tensor<16x128x4x256xf32>
    %179 = "mhlo.fusion"(%178, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4x8xf32>
      mhlo.return %662 : tensor<16x128x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4x8xf32>
    %180 = "mhlo.fusion"(%179, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x128x4x8xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x128x4x8xf32>, tensor<f32>) -> tensor<16x128x4xf32>
      mhlo.return %662 : tensor<16x128x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x8xf32>, tensor<f32>) -> tensor<16x128x4xf32>
    %181 = "mhlo.fusion"(%180, %170, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x4xf32>, %arg246: tensor<16x4x256x257xf32>, %arg247: tensor<257xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<16x4x256x257xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x257xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,257,4,256]{1,3,2,0}"} : (tensor<16x4x256x257xf32>) -> tensor<16x257x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x257x4x256xf32>
      %666 = stablehlo.slice %665 [0:16, 129:257, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x128x4x256xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x128x4x256xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x128x4xf32>) -> tensor<16x128x4x256xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x128x4x256xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
      %672 = mhlo.copy %671 : tensor<16x4x256x128xf32>
      mhlo.return %672 : tensor<16x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4xf32>, tensor<16x4x256x257xf32>, tensor<257xf32>) -> tensor<16x4x256x128xf32>
    %182 = mhlo.bitcast %181 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x256x128xf32>) -> tensor<16384x128xf32>
    %183 = "mhlo.fusion"(%arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x128xf32>
      mhlo.return %663 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %185 = stablehlo.dot %182, %184, precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<128x128xf32>) -> tensor<16384x128xf32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<16x4x256x128xf32>
    %187 = "mhlo.fusion"(%156, %186, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<16x4x256x128xf32>, %arg247: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x128xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x128x4x256xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x128x4x256xf32>
      mhlo.return %666 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<16x4x256x128xf32>, tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
    %189 = "mhlo.fusion"(%188, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
      mhlo.return %662 : tensor<1x16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    %190 = "mhlo.fusion"(%189, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x128xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
      mhlo.return %662 : tensor<1x16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    %191 = "mhlo.fusion"(%190, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x4xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %192 = "mhlo.fusion"(%191, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x128x4x256xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x131072xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x131072xf32>
      mhlo.return %667 : tensor<16x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
    %193 = "mhlo.fusion"(%192, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %662 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %194 = "mhlo.fusion"(%193, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x4096xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
      mhlo.return %662 : tensor<16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    %195 = "mhlo.fusion"(%194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4xf32>) -> tensor<16xf32>
    %196 = "mhlo.fusion"(%arg61, %195, %191, %187, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x4x256xf32>, %arg249: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x131072xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x131072xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x128x4x256xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x128x4x256xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,128]{2,1,3,0}"} : (tensor<16x128x4x256xf32>) -> tensor<16x4x256x128xf32>
      %675 = mhlo.copy %674 : tensor<16x4x256x128xf32>
      mhlo.return %675 : tensor<16x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<128xf32>) -> tensor<16x4x256x128xf32>
    %197 = mhlo.bitcast %196 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x256x128xf32>) -> tensor<16384x128xf32>
    %198 = "mhlo.fusion"(%arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x256xf32>
      mhlo.return %663 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %200 = stablehlo.dot %197, %199, precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<128x256xf32>) -> tensor<16384x256xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<16x4x256x256xf32>
    %202 = "mhlo.fusion"(%201, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x256x256xf32>, %arg246: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x4x256x256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x256x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,256]{1,3,2,0}"} : (tensor<16x4x256x256xf32>) -> tensor<16x256x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x4x256xf32>
      %667 = stablehlo.negate %665 : tensor<16x256x4x256xf32>
      %668 = stablehlo.exponential %667 : tensor<16x256x4x256xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x256x4x256xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x256x4x256xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x256x4x256xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,256,256]{2,1,3,0}"} : (tensor<16x256x4x256xf32>) -> tensor<16x4x256x256xf32>
      %673 = mhlo.copy %672 : tensor<16x4x256x256xf32>
      mhlo.return %673 : tensor<16x4x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x256x256xf32>, tensor<256xf32>) -> tensor<16x4x256x256xf32>
    %203 = mhlo.bitcast %202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x256x256xf32>) -> tensor<16384x256xf32>
    %204 = "mhlo.fusion"(%arg65) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x128xf32>
      mhlo.return %663 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %205 = mhlo.bitcast %204 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %206 = stablehlo.dot %203, %205, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x128xf32>) -> tensor<16384x128xf32>
    %207 = mhlo.bitcast %206 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<16x4x256x128xf32>
    %208 = "mhlo.fusion"(%187, %207, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<16x4x256x128xf32>, %arg247: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x256x128xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x128x4x256xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x128x4x256xf32>
      %667 = mhlo.bitcast %666 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
      mhlo.return %667 : tensor<1x16x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<16x4x256x128xf32>, tensor<128xf32>) -> tensor<1x16x131072xf32>
    %209 = "mhlo.fusion"(%208, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
      mhlo.return %662 : tensor<1x16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x131072xf32>, tensor<f32>) -> tensor<1x16x128xf32>
    %210 = "mhlo.fusion"(%209, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x128xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
      mhlo.return %662 : tensor<1x16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x128xf32>, tensor<f32>) -> tensor<1x16x4xf32>
    %211 = "mhlo.fusion"(%210, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x4xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<16xf32>
    %212 = "mhlo.fusion"(%211, %187, %207, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x128x4x256xf32>, %arg247: tensor<16x4x256x128xf32>, %arg248: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
      %663 = stablehlo.add %arg247, %662 : tensor<16x4x256x128xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
      %665 = mhlo.copy %664 : tensor<16x128x4x256xf32>
      %666 = stablehlo.add %665, %arg246 : tensor<16x128x4x256xf32>
      %667 = mhlo.bitcast %666 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %668 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %669 = stablehlo.multiply %arg245, %668 : tensor<16xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %671 = stablehlo.subtract %667, %670 : tensor<16x131072xf32>
      %672 = stablehlo.multiply %671, %671 : tensor<16x131072xf32>
      mhlo.return %672 : tensor<16x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<16x4x256x128xf32>, tensor<128xf32>) -> tensor<16x131072xf32>
    %213 = "mhlo.fusion"(%212, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x131072xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %662 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x131072xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %214 = "mhlo.fusion"(%213, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x4096xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
      mhlo.return %662 : tensor<16x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>, tensor<f32>) -> tensor<16x4xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x4xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4xf32>) -> tensor<16xf32>
    %216 = "mhlo.fusion"(%arg67, %215, %211, %187, %207, %arg66, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x4x256xf32>, %arg249: tensor<16x4x256x128xf32>, %arg250: tensor<128xf32>, %arg251: tensor<128xf32>):
      %662 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %663 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<128xf32>) -> tensor<16x4x256x128xf32>
      %664 = stablehlo.add %arg249, %663 : tensor<16x4x256x128xf32>
      %665 = stablehlo.transpose %664, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,4,256]{1,3,2,0}"} : (tensor<16x4x256x128xf32>) -> tensor<16x128x4x256xf32>
      %666 = mhlo.copy %665 : tensor<16x128x4x256xf32>
      %667 = stablehlo.add %666, %arg248 : tensor<16x128x4x256xf32>
      %668 = mhlo.bitcast %667 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %669 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %670 = stablehlo.multiply %arg247, %669 : tensor<16xf32>
      %671 = stablehlo.broadcast_in_dim %670, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %672 = stablehlo.subtract %668, %671 : tensor<16x131072xf32>
      %673 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x131072xf32>
      %674 = stablehlo.multiply %672, %673 : tensor<16x131072xf32>
      %675 = mhlo.bitcast %674 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x131072xf32>) -> tensor<16x128x4x256xf32>
      %676 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %677 = stablehlo.multiply %675, %676 : tensor<16x128x4x256xf32>
      %678 = stablehlo.add %662, %677 : tensor<16x128x4x256xf32>
      %679 = mhlo.bitcast %678 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x128x2x2x16x16xf32>
      %680 = stablehlo.transpose %679, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,16,2,16,16,128]{3,1,2,0,5,4}"} : (tensor<16x128x2x2x16x16xf32>) -> tensor<2x16x2x16x16x128xf32>
      %681 = mhlo.copy %680 : tensor<2x16x2x16x16x128xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1024,16,128,1,1]{2,1,0,4,3}"} : (tensor<2x16x2x16x16x128xf32>) -> tensor<1024x16x128x1x1xf32>
      %683 = mhlo.copy %682 : tensor<1024x16x128x1x1xf32>
      mhlo.return %683 : tensor<1024x16x128x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<16x4x256x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1024x16x128x1x1xf32>
    %217 = "mhlo.fusion"(%15, %16, %216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x32x32xf32>, %arg246: tensor<1024x2xi32>, %arg247: tensor<1024x16x128x1x1xf32>):
      %662 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %663 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x16x128x1x1xf32>) -> tensor<16x128x32x32xf32>
      mhlo.return %662 : tensor<16x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x16x128x1x1xf32>) -> tensor<16x128x32x32xf32>
    %218 = "mhlo.fusion"(%217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x32x32xf32>):
      %662 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,128]{2,1,3,0}"} : (tensor<16x128x32x32xf32>) -> tensor<16x32x32x128xf32>
      %663 = mhlo.copy %662 : tensor<16x32x32x128xf32>
      mhlo.return %663 : tensor<16x32x32x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x32x32xf32>) -> tensor<16x32x32x128xf32>
    %219 = mhlo.bitcast %218 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x32x128xf32>) -> tensor<16384x128xf32>
    %220 = "mhlo.fusion"(%arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x128x256xf32>
      mhlo.return %663 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %221 = mhlo.bitcast %220 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %222 = stablehlo.dot %219, %221, precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<128x256xf32>) -> tensor<16384x256xf32>
    %223 = mhlo.bitcast %222 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<16x32x32x256xf32>
    %224 = "mhlo.fusion"(%arg71, %arg70, %14, %223, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x32x32x256xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x32x32x256xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x32x32x256xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x32x32x256xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<16x32x32x256xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x32x32x256xf32>
      mhlo.return %669 : tensor<16x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x32x32x256xf32>, tensor<256xf32>) -> tensor<16x32x32x256xf32>
    %225 = mhlo.bitcast %224 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x32x256xf32>) -> tensor<16384x256xf32>
    %226 = "mhlo.fusion"(%arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x512xf32>
      mhlo.return %663 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %227 = mhlo.bitcast %226 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %228 = stablehlo.dot %225, %227, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x512xf32>) -> tensor<16384x512xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x512xf32>) -> tensor<16x32x32x512xf32>
    %230 = "mhlo.fusion"(%arg74, %arg73, %13, %229, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<16x32x32x512xf32>, %arg249: tensor<512xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<16x32x32x512xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x32x32x512xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<16x32x32x512xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x32x32x512xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<16x32x32x512xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x32x32x512xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<16x32x32x512xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x32x32x512xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,32,32]{1,3,2,0}"} : (tensor<16x32x32x512xf32>) -> tensor<16x512x32x32xf32>
      %671 = mhlo.copy %670 : tensor<16x512x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512x32x32xf32>
      %673 = stablehlo.negate %671 : tensor<16x512x32x32xf32>
      %674 = stablehlo.exponential %673 : tensor<16x512x32x32xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x512x32x32xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x512x32x32xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x512x32x32xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,512]{2,1,3,0}"} : (tensor<16x512x32x32xf32>) -> tensor<16x32x32x512xf32>
      %679 = mhlo.copy %678 : tensor<16x32x32x512xf32>
      mhlo.return %679 : tensor<16x32x32x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x32x32x512xf32>, tensor<512xf32>) -> tensor<16x32x32x512xf32>
    %231 = "mhlo.fusion"(%arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x512xf32>
      mhlo.return %663 : tensor<3x3x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %232 = stablehlo.convolution(%230, %231) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x32x32x512xf32>, tensor<3x3x1x512xf32>) -> tensor<16x16x16x512xf32>
    %233 = "mhlo.fusion"(%arg77, %arg76, %12, %232, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<16x16x16x512xf32>, %arg249: tensor<512xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<16x16x16x512xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x16x16x512xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<16x16x16x512xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x16x16x512xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<16x16x16x512xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x16x16x512xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<16x16x16x512xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x16x16x512xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,16,16]{1,3,2,0}"} : (tensor<16x16x16x512xf32>) -> tensor<16x512x16x16xf32>
      %671 = mhlo.copy %670 : tensor<16x512x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512x16x16xf32>
      %673 = stablehlo.negate %671 : tensor<16x512x16x16xf32>
      %674 = stablehlo.exponential %673 : tensor<16x512x16x16xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x512x16x16xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x512x16x16xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x512x16x16xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,512]{2,1,3,0}"} : (tensor<16x512x16x16xf32>) -> tensor<16x16x16x512xf32>
      %679 = mhlo.copy %678 : tensor<16x16x16x512xf32>
      mhlo.return %679 : tensor<16x16x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x16x16x512xf32>, tensor<512xf32>) -> tensor<16x16x16x512xf32>
    %234 = mhlo.bitcast %233 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x16x16x512xf32>) -> tensor<4096x512xf32>
    %235 = "mhlo.fusion"(%arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x512x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,384]{1,0,2,3}"} : (tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32>
      %663 = mhlo.copy %662 : tensor<1x1x512x384xf32>
      mhlo.return %663 : tensor<1x1x512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32>
    %236 = mhlo.bitcast %235 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x384xf32>) -> tensor<512x384xf32>
    %237 = stablehlo.dot %234, %236, precision = [DEFAULT, DEFAULT] : (tensor<4096x512xf32>, tensor<512x384xf32>) -> tensor<4096x384xf32>
    %238 = mhlo.bitcast %237 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x384xf32>) -> tensor<16x16x16x384xf32>
    %239 = "mhlo.fusion"(%arg80, %arg79, %11, %238, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<16x16x16x384xf32>, %arg249: tensor<384xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x16x16x384xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x16x16x384xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x16x16x384xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x16x16x384xf32>
      mhlo.return %669 : tensor<16x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x16x16x384xf32>, tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %240 = "mhlo.fusion"(%arg81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,384]{1,0,2,3}"} : (tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x384xf32>
      mhlo.return %663 : tensor<3x3x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32>
    %241 = stablehlo.convolution(%239, %240) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x16x16x384xf32>, tensor<3x3x1x384xf32>) -> tensor<16x16x16x384xf32>
    %242 = "mhlo.fusion"(%arg83, %arg82, %10, %241, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<16x16x16x384xf32>, %arg249: tensor<384xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x16x16x384xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x16x16x384xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x16x16x384xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x16x16x384xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,16,16]{1,3,2,0}"} : (tensor<16x16x16x384xf32>) -> tensor<16x384x16x16xf32>
      %671 = mhlo.copy %670 : tensor<16x384x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x384x16x16xf32>
      %673 = stablehlo.negate %671 : tensor<16x384x16x16xf32>
      %674 = stablehlo.exponential %673 : tensor<16x384x16x16xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x384x16x16xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x384x16x16xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x384x16x16xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,384]{2,1,3,0}"} : (tensor<16x384x16x16xf32>) -> tensor<16x16x16x384xf32>
      %679 = mhlo.copy %678 : tensor<16x16x16x384xf32>
      mhlo.return %679 : tensor<16x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x16x16x384xf32>, tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %243 = mhlo.bitcast %242 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x16x16x384xf32>) -> tensor<4096x384xf32>
    %244 = "mhlo.fusion"(%arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x384x192xf32>
      mhlo.return %663 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %246 = stablehlo.dot %243, %245, precision = [DEFAULT, DEFAULT] : (tensor<4096x384xf32>, tensor<384x192xf32>) -> tensor<4096x192xf32>
    %247 = mhlo.bitcast %246 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x16x16x192xf32>
    %248 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %662 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %663 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %664 = stablehlo.multiply %662, %663 : tensor<8xi32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %666 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %667 = stablehlo.add %665, %666 : tensor<2x8xi32>
      %668 = mhlo.bitcast %667 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
      %669 = mhlo.copy %668 : tensor<2x8x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %670 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %671 = stablehlo.compare LT, %669, %670 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
      %c_2 = stablehlo.constant dense<16> : tensor<i32>
      %672 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %673 = stablehlo.add %669, %672 : tensor<2x8x1x1xi32>
      %674 = stablehlo.select %671, %673, %669 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
      %675 = mhlo.bitcast %674 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %677 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %678 = stablehlo.compare LT, %667, %677 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %679 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %680 = stablehlo.add %667, %679 : tensor<2x8xi32>
      %681 = stablehlo.select %678, %680, %667 : tensor<2x8xi1>, tensor<2x8xi32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %683 = stablehlo.concatenate %676, %682, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      mhlo.return %683 : tensor<2x8x2x8x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x8x2x8x2xi32>
    %249 = "mhlo.fusion"(%247, %248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16x16x192xf32>, %arg246: tensor<2x8x2x8x2xi32>):
      %662 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,16,16]{1,3,2,0}"} : (tensor<16x16x16x192xf32>) -> tensor<16x192x16x16xf32>
      %663 = mhlo.copy %662 : tensor<16x192x16x16xf32>
      %664 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %665 = "stablehlo.gather"(%663, %664) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 192, 1, 1>}> : (tensor<16x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x16x192x1x1xf32>
      mhlo.return %665 : tensor<256x16x192x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x192xf32>, tensor<2x8x2x8x2xi32>) -> tensor<256x16x192x1x1xf32>
    %250 = "mhlo.fusion"(%249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x16x192x1x1xf32>):
      %662 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
      %663 = stablehlo.transpose %662, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
      %664 = mhlo.copy %663 : tensor<16x192x2x2x8x8xf32>
      %665 = mhlo.bitcast %664 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<1x16x49152xf32>
      mhlo.return %665 : tensor<1x16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x192x1x1xf32>) -> tensor<1x16x49152xf32>
    %251 = "mhlo.fusion"(%250, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %252 = "mhlo.fusion"(%251, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %253 = "mhlo.fusion"(%252, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %254 = "mhlo.fusion"(%253, %249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<256x16x192x1x1xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
      %663 = stablehlo.transpose %662, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
      %664 = mhlo.copy %663 : tensor<16x192x2x2x8x8xf32>
      %665 = mhlo.bitcast %664 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %667 = stablehlo.multiply %arg245, %666 : tensor<16xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %669 = stablehlo.subtract %665, %668 : tensor<16x49152xf32>
      %670 = stablehlo.multiply %669, %669 : tensor<16x49152xf32>
      mhlo.return %670 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<256x16x192x1x1xf32>) -> tensor<16x49152xf32>
    %255 = "mhlo.fusion"(%254, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %257 = "mhlo.fusion"(%256, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %258 = "mhlo.fusion"(%257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %259 = "mhlo.fusion"(%arg85, %258, %253, %249, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<256x16x192x1x1xf32>, %arg249: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
      %664 = stablehlo.transpose %663, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
      %665 = mhlo.copy %664 : tensor<16x192x2x2x8x8xf32>
      %666 = mhlo.bitcast %665 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %668 = stablehlo.multiply %arg247, %667 : tensor<16xf32>
      %669 = stablehlo.broadcast_in_dim %668, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %670 = stablehlo.subtract %666, %669 : tensor<16x49152xf32>
      %671 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x49152xf32>
      %673 = mhlo.bitcast %672 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %674 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %675 = stablehlo.multiply %673, %674 : tensor<16x192x4x64xf32>
      %676 = stablehlo.add %662, %675 : tensor<16x192x4x64xf32>
      %677 = stablehlo.transpose %676, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %678 = mhlo.copy %677 : tensor<16x4x64x192xf32>
      mhlo.return %678 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<256x16x192x1x1xf32>, tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %260 = mhlo.bitcast %259 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %261 = "mhlo.fusion"(%arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x385xf32>
      mhlo.return %663 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %262 = mhlo.bitcast %261 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %263 = stablehlo.dot %260, %262, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x385xf32>) -> tensor<4096x385xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x385xf32>) -> tensor<16x4x64x385xf32>
    %265 = "mhlo.fusion"(%264, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x385xf32>, %arg246: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      mhlo.return %666 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x1x4x64xf32>
    %266 = "mhlo.fusion"(%265, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
      mhlo.return %662 : tensor<16x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    %267 = "mhlo.fusion"(%266, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %268 = "mhlo.fusion"(%267, %264, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x64xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x64xf32>
      mhlo.return %672 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x1x4x64xf32>
    %269 = "mhlo.fusion"(%268, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
      mhlo.return %662 : tensor<16x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    %270 = "mhlo.fusion"(%269, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %271 = "mhlo.fusion"(%268, %270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x64xf32>
      mhlo.return %664 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x64xf32>
    %272 = "mhlo.fusion"(%271, %264, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x192x4x64xf32>
      mhlo.return %669 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x192x4x64xf32>
    %273 = "mhlo.fusion"(%272, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
      mhlo.return %662 : tensor<16x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
    %274 = "mhlo.fusion"(%273, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
      mhlo.return %662 : tensor<16x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    %275 = "mhlo.fusion"(%274, %264, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x192x4x64xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x192x4x64xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %672 = mhlo.copy %671 : tensor<16x4x64x192xf32>
      mhlo.return %672 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x4x64x192xf32>
    %276 = mhlo.bitcast %275 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %277 = "mhlo.fusion"(%arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x192xf32>
      mhlo.return %663 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %278 = mhlo.bitcast %277 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %279 = stablehlo.dot %276, %278, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x192xf32>) -> tensor<4096x192xf32>
    %280 = mhlo.bitcast %279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x4x64x192xf32>
    %281 = "mhlo.fusion"(%280, %arg90, %249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<256x16x192x1x1xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
      %667 = stablehlo.transpose %666, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
      %668 = mhlo.copy %667 : tensor<16x192x2x2x8x8xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x192x4x64xf32>
      %670 = stablehlo.add %665, %669 : tensor<16x192x4x64xf32>
      mhlo.return %670 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x192xf32>, tensor<192xf32>, tensor<256x16x192x1x1xf32>) -> tensor<16x192x4x64xf32>
    %282 = mhlo.bitcast %281 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %283 = "mhlo.fusion"(%282, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %284 = "mhlo.fusion"(%283, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %285 = "mhlo.fusion"(%284, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %286 = "mhlo.fusion"(%285, %281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x192x4x64xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x49152xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x49152xf32>
      mhlo.return %667 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %287 = "mhlo.fusion"(%286, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %288 = "mhlo.fusion"(%287, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %289 = "mhlo.fusion"(%288, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %290 = "mhlo.fusion"(%289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %291 = "mhlo.fusion"(%arg91, %290, %285, %281, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x192x4x64xf32>, %arg249: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x49152xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x49152xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x192x4x64xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x192x4x64xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %675 = mhlo.copy %674 : tensor<16x4x64x192xf32>
      mhlo.return %675 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %293 = "mhlo.fusion"(%arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x384xf32>
      mhlo.return %663 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %295 = stablehlo.dot %292, %294, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x384xf32>) -> tensor<4096x384xf32>
    %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x384xf32>) -> tensor<16x4x64x384xf32>
    %297 = "mhlo.fusion"(%296, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x384xf32>, %arg246: tensor<384xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<16x4x64x384xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x384xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,4,64]{1,3,2,0}"} : (tensor<16x4x64x384xf32>) -> tensor<16x384x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
      %667 = stablehlo.negate %665 : tensor<16x384x4x64xf32>
      %668 = stablehlo.exponential %667 : tensor<16x384x4x64xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x384x4x64xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x384x4x64xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x384x4x64xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,384]{2,1,3,0}"} : (tensor<16x384x4x64xf32>) -> tensor<16x4x64x384xf32>
      %673 = mhlo.copy %672 : tensor<16x4x64x384xf32>
      mhlo.return %673 : tensor<16x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x384xf32>, tensor<384xf32>) -> tensor<16x4x64x384xf32>
    %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x384xf32>) -> tensor<4096x384xf32>
    %299 = "mhlo.fusion"(%arg95) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x384x192xf32>
      mhlo.return %663 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %301 = stablehlo.dot %298, %300, precision = [DEFAULT, DEFAULT] : (tensor<4096x384xf32>, tensor<384x192xf32>) -> tensor<4096x192xf32>
    %302 = mhlo.bitcast %301 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x4x64x192xf32>
    %303 = "mhlo.fusion"(%281, %302, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16x4x64x192xf32>, %arg247: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x192x4x64xf32>
      mhlo.return %666 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %305 = "mhlo.fusion"(%304, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %306 = "mhlo.fusion"(%305, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %307 = "mhlo.fusion"(%306, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %308 = "mhlo.fusion"(%307, %303) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x192x4x64xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x49152xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x49152xf32>
      mhlo.return %667 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %309 = "mhlo.fusion"(%308, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %310 = "mhlo.fusion"(%309, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %311 = "mhlo.fusion"(%310, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %312 = "mhlo.fusion"(%311) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %313 = "mhlo.fusion"(%arg97, %312, %307, %303, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x192x4x64xf32>, %arg249: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x49152xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x49152xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x192x4x64xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x192x4x64xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %675 = mhlo.copy %674 : tensor<16x4x64x192xf32>
      mhlo.return %675 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %315 = "mhlo.fusion"(%arg99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x385xf32>
      mhlo.return %663 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %316 = mhlo.bitcast %315 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %317 = stablehlo.dot %314, %316, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x385xf32>) -> tensor<4096x385xf32>
    %318 = mhlo.bitcast %317 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x385xf32>) -> tensor<16x4x64x385xf32>
    %319 = "mhlo.fusion"(%318, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x385xf32>, %arg246: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      mhlo.return %666 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x1x4x64xf32>
    %320 = "mhlo.fusion"(%319, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
      mhlo.return %662 : tensor<16x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    %321 = "mhlo.fusion"(%320, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %322 = "mhlo.fusion"(%321, %318, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x64xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x64xf32>
      mhlo.return %672 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x1x4x64xf32>
    %323 = "mhlo.fusion"(%322, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
      mhlo.return %662 : tensor<16x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    %324 = "mhlo.fusion"(%323, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %325 = "mhlo.fusion"(%322, %324) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x64xf32>
      mhlo.return %664 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x64xf32>
    %326 = "mhlo.fusion"(%325, %318, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x192x4x64xf32>
      mhlo.return %669 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x192x4x64xf32>
    %327 = "mhlo.fusion"(%326, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
      mhlo.return %662 : tensor<16x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
    %328 = "mhlo.fusion"(%327, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
      mhlo.return %662 : tensor<16x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    %329 = "mhlo.fusion"(%328, %318, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x192x4x64xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x192x4x64xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %672 = mhlo.copy %671 : tensor<16x4x64x192xf32>
      mhlo.return %672 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x4x64x192xf32>
    %330 = mhlo.bitcast %329 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %331 = "mhlo.fusion"(%arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x192xf32>
      mhlo.return %663 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %332 = mhlo.bitcast %331 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %333 = stablehlo.dot %330, %332, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x192xf32>) -> tensor<4096x192xf32>
    %334 = mhlo.bitcast %333 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x4x64x192xf32>
    %335 = "mhlo.fusion"(%303, %334, %arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16x4x64x192xf32>, %arg247: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x192x4x64xf32>
      mhlo.return %666 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %336 = mhlo.bitcast %335 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %337 = "mhlo.fusion"(%336, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %338 = "mhlo.fusion"(%337, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %339 = "mhlo.fusion"(%338, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %340 = "mhlo.fusion"(%339, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x192x4x64xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x49152xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x49152xf32>
      mhlo.return %667 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %341 = "mhlo.fusion"(%340, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %342 = "mhlo.fusion"(%341, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %343 = "mhlo.fusion"(%342, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %344 = "mhlo.fusion"(%343) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %345 = "mhlo.fusion"(%arg103, %344, %339, %335, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x192x4x64xf32>, %arg249: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x49152xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x49152xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x192x4x64xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x192x4x64xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %675 = mhlo.copy %674 : tensor<16x4x64x192xf32>
      mhlo.return %675 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %346 = mhlo.bitcast %345 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %347 = "mhlo.fusion"(%arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x384xf32>
      mhlo.return %663 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %348 = mhlo.bitcast %347 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %349 = stablehlo.dot %346, %348, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x384xf32>) -> tensor<4096x384xf32>
    %350 = mhlo.bitcast %349 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x384xf32>) -> tensor<16x4x64x384xf32>
    %351 = "mhlo.fusion"(%350, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x384xf32>, %arg246: tensor<384xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<16x4x64x384xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x384xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,4,64]{1,3,2,0}"} : (tensor<16x4x64x384xf32>) -> tensor<16x384x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
      %667 = stablehlo.negate %665 : tensor<16x384x4x64xf32>
      %668 = stablehlo.exponential %667 : tensor<16x384x4x64xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x384x4x64xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x384x4x64xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x384x4x64xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,384]{2,1,3,0}"} : (tensor<16x384x4x64xf32>) -> tensor<16x4x64x384xf32>
      %673 = mhlo.copy %672 : tensor<16x4x64x384xf32>
      mhlo.return %673 : tensor<16x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x384xf32>, tensor<384xf32>) -> tensor<16x4x64x384xf32>
    %352 = mhlo.bitcast %351 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x384xf32>) -> tensor<4096x384xf32>
    %353 = "mhlo.fusion"(%arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x384x192xf32>
      mhlo.return %663 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %354 = mhlo.bitcast %353 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %355 = stablehlo.dot %352, %354, precision = [DEFAULT, DEFAULT] : (tensor<4096x384xf32>, tensor<384x192xf32>) -> tensor<4096x192xf32>
    %356 = mhlo.bitcast %355 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x4x64x192xf32>
    %357 = "mhlo.fusion"(%335, %356, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16x4x64x192xf32>, %arg247: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x192x4x64xf32>
      mhlo.return %666 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %358 = mhlo.bitcast %357 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %359 = "mhlo.fusion"(%358, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %360 = "mhlo.fusion"(%359, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %361 = "mhlo.fusion"(%360, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %362 = "mhlo.fusion"(%361, %357) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x192x4x64xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x49152xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x49152xf32>
      mhlo.return %667 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %363 = "mhlo.fusion"(%362, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %364 = "mhlo.fusion"(%363, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %365 = "mhlo.fusion"(%364, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %366 = "mhlo.fusion"(%365) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %367 = "mhlo.fusion"(%arg109, %366, %361, %357, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x192x4x64xf32>, %arg249: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x49152xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x49152xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x192x4x64xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x192x4x64xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %675 = mhlo.copy %674 : tensor<16x4x64x192xf32>
      mhlo.return %675 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %368 = mhlo.bitcast %367 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %369 = "mhlo.fusion"(%arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x385xf32>
      mhlo.return %663 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %370 = mhlo.bitcast %369 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %371 = stablehlo.dot %368, %370, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x385xf32>) -> tensor<4096x385xf32>
    %372 = mhlo.bitcast %371 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x385xf32>) -> tensor<16x4x64x385xf32>
    %373 = "mhlo.fusion"(%372, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x385xf32>, %arg246: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      mhlo.return %666 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x1x4x64xf32>
    %374 = "mhlo.fusion"(%373, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
      mhlo.return %662 : tensor<16x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    %375 = "mhlo.fusion"(%374, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %376 = "mhlo.fusion"(%375, %372, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x64xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x64xf32>
      mhlo.return %672 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x1x4x64xf32>
    %377 = "mhlo.fusion"(%376, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
      mhlo.return %662 : tensor<16x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    %378 = "mhlo.fusion"(%377, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %379 = "mhlo.fusion"(%376, %378) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x64xf32>
      mhlo.return %664 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x64xf32>
    %380 = "mhlo.fusion"(%379, %372, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x192x4x64xf32>
      mhlo.return %669 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x192x4x64xf32>
    %381 = "mhlo.fusion"(%380, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
      mhlo.return %662 : tensor<16x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
    %382 = "mhlo.fusion"(%381, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
      mhlo.return %662 : tensor<16x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    %383 = "mhlo.fusion"(%382, %372, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x192x4x64xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x192x4x64xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %672 = mhlo.copy %671 : tensor<16x4x64x192xf32>
      mhlo.return %672 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x4x64x192xf32>
    %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %385 = "mhlo.fusion"(%arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x192xf32>
      mhlo.return %663 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %386 = mhlo.bitcast %385 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %387 = stablehlo.dot %384, %386, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x192xf32>) -> tensor<4096x192xf32>
    %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x4x64x192xf32>
    %389 = "mhlo.fusion"(%357, %388, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16x4x64x192xf32>, %arg247: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x192x4x64xf32>
      mhlo.return %666 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %390 = mhlo.bitcast %389 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %391 = "mhlo.fusion"(%390, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %392 = "mhlo.fusion"(%391, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %393 = "mhlo.fusion"(%392, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %394 = "mhlo.fusion"(%393, %389) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x192x4x64xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x49152xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x49152xf32>
      mhlo.return %667 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %395 = "mhlo.fusion"(%394, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %396 = "mhlo.fusion"(%395, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %397 = "mhlo.fusion"(%396, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %398 = "mhlo.fusion"(%397) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %399 = "mhlo.fusion"(%arg115, %398, %393, %389, %arg116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x192x4x64xf32>, %arg249: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x49152xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x49152xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x192x4x64xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x192x4x64xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %675 = mhlo.copy %674 : tensor<16x4x64x192xf32>
      mhlo.return %675 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %401 = "mhlo.fusion"(%arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x384xf32>
      mhlo.return %663 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %402 = mhlo.bitcast %401 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %403 = stablehlo.dot %400, %402, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x384xf32>) -> tensor<4096x384xf32>
    %404 = mhlo.bitcast %403 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x384xf32>) -> tensor<16x4x64x384xf32>
    %405 = "mhlo.fusion"(%404, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x384xf32>, %arg246: tensor<384xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<16x4x64x384xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x384xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,4,64]{1,3,2,0}"} : (tensor<16x4x64x384xf32>) -> tensor<16x384x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
      %667 = stablehlo.negate %665 : tensor<16x384x4x64xf32>
      %668 = stablehlo.exponential %667 : tensor<16x384x4x64xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x384x4x64xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x384x4x64xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x384x4x64xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,384]{2,1,3,0}"} : (tensor<16x384x4x64xf32>) -> tensor<16x4x64x384xf32>
      %673 = mhlo.copy %672 : tensor<16x4x64x384xf32>
      mhlo.return %673 : tensor<16x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x384xf32>, tensor<384xf32>) -> tensor<16x4x64x384xf32>
    %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x384xf32>) -> tensor<4096x384xf32>
    %407 = "mhlo.fusion"(%arg119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x384x192xf32>
      mhlo.return %663 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %409 = stablehlo.dot %406, %408, precision = [DEFAULT, DEFAULT] : (tensor<4096x384xf32>, tensor<384x192xf32>) -> tensor<4096x192xf32>
    %410 = mhlo.bitcast %409 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x4x64x192xf32>
    %411 = "mhlo.fusion"(%389, %410, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16x4x64x192xf32>, %arg247: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x192x4x64xf32>
      mhlo.return %666 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %413 = "mhlo.fusion"(%412, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %414 = "mhlo.fusion"(%413, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %415 = "mhlo.fusion"(%414, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %416 = "mhlo.fusion"(%415, %411) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x192x4x64xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x49152xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x49152xf32>
      mhlo.return %667 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %417 = "mhlo.fusion"(%416, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %418 = "mhlo.fusion"(%417, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %419 = "mhlo.fusion"(%418, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %420 = "mhlo.fusion"(%419) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %421 = "mhlo.fusion"(%arg121, %420, %415, %411, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x192x4x64xf32>, %arg249: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x49152xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x49152xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x192x4x64xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x192x4x64xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %675 = mhlo.copy %674 : tensor<16x4x64x192xf32>
      mhlo.return %675 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %422 = mhlo.bitcast %421 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %423 = "mhlo.fusion"(%arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x385xf32>
      mhlo.return %663 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %425 = stablehlo.dot %422, %424, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x385xf32>) -> tensor<4096x385xf32>
    %426 = mhlo.bitcast %425 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x385xf32>) -> tensor<16x4x64x385xf32>
    %427 = "mhlo.fusion"(%426, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x385xf32>, %arg246: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      mhlo.return %666 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x1x4x64xf32>
    %428 = "mhlo.fusion"(%427, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
      mhlo.return %662 : tensor<16x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    %429 = "mhlo.fusion"(%428, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %430 = "mhlo.fusion"(%429, %426, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x64xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x64xf32>
      mhlo.return %672 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x1x4x64xf32>
    %431 = "mhlo.fusion"(%430, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
      mhlo.return %662 : tensor<16x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<f32>) -> tensor<16x1x4x2xf32>
    %432 = "mhlo.fusion"(%431, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x2xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %433 = "mhlo.fusion"(%430, %432) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x64xf32>
      mhlo.return %664 : tensor<16x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x64xf32>
    %434 = "mhlo.fusion"(%433, %426, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x64xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x192x4x64xf32>
      mhlo.return %669 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x64xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x192x4x64xf32>
    %435 = "mhlo.fusion"(%434, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
      mhlo.return %662 : tensor<16x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4x2xf32>
    %436 = "mhlo.fusion"(%435, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
      mhlo.return %662 : tensor<16x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x2xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    %437 = "mhlo.fusion"(%436, %426, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4xf32>, %arg246: tensor<16x4x64x385xf32>, %arg247: tensor<385xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<16x4x64x385xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x385xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,385,4,64]{1,3,2,0}"} : (tensor<16x4x64x385xf32>) -> tensor<16x385x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x385x4x64xf32>
      %666 = stablehlo.slice %665 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x192x4x64xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x192x4x64xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %672 = mhlo.copy %671 : tensor<16x4x64x192xf32>
      mhlo.return %672 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4xf32>, tensor<16x4x64x385xf32>, tensor<385xf32>) -> tensor<16x4x64x192xf32>
    %438 = mhlo.bitcast %437 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %439 = "mhlo.fusion"(%arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x192xf32>
      mhlo.return %663 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %440 = mhlo.bitcast %439 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %441 = stablehlo.dot %438, %440, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x192xf32>) -> tensor<4096x192xf32>
    %442 = mhlo.bitcast %441 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x4x64x192xf32>
    %443 = "mhlo.fusion"(%411, %442, %arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16x4x64x192xf32>, %arg247: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x192x4x64xf32>
      mhlo.return %666 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %444 = mhlo.bitcast %443 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %445 = "mhlo.fusion"(%444, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %446 = "mhlo.fusion"(%445, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %447 = "mhlo.fusion"(%446, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %448 = "mhlo.fusion"(%447, %443) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x192x4x64xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x49152xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x49152xf32>
      mhlo.return %667 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
    %449 = "mhlo.fusion"(%448, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %450 = "mhlo.fusion"(%449, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %451 = "mhlo.fusion"(%450, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %452 = "mhlo.fusion"(%451) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %453 = "mhlo.fusion"(%arg127, %452, %447, %443, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x192x4x64xf32>, %arg249: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x49152xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x49152xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x192x4x64xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x192x4x64xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,192]{2,1,3,0}"} : (tensor<16x192x4x64xf32>) -> tensor<16x4x64x192xf32>
      %675 = mhlo.copy %674 : tensor<16x4x64x192xf32>
      mhlo.return %675 : tensor<16x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<192xf32>) -> tensor<16x4x64x192xf32>
    %454 = mhlo.bitcast %453 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x192xf32>) -> tensor<4096x192xf32>
    %455 = "mhlo.fusion"(%arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x384xf32>
      mhlo.return %663 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %456 = mhlo.bitcast %455 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %457 = stablehlo.dot %454, %456, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x384xf32>) -> tensor<4096x384xf32>
    %458 = mhlo.bitcast %457 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x384xf32>) -> tensor<16x4x64x384xf32>
    %459 = "mhlo.fusion"(%458, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x64x384xf32>, %arg246: tensor<384xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<16x4x64x384xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x64x384xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,384,4,64]{1,3,2,0}"} : (tensor<16x4x64x384xf32>) -> tensor<16x384x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
      %667 = stablehlo.negate %665 : tensor<16x384x4x64xf32>
      %668 = stablehlo.exponential %667 : tensor<16x384x4x64xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x384x4x64xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x384x4x64xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x384x4x64xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,64,384]{2,1,3,0}"} : (tensor<16x384x4x64xf32>) -> tensor<16x4x64x384xf32>
      %673 = mhlo.copy %672 : tensor<16x4x64x384xf32>
      mhlo.return %673 : tensor<16x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x64x384xf32>, tensor<384xf32>) -> tensor<16x4x64x384xf32>
    %460 = mhlo.bitcast %459 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x64x384xf32>) -> tensor<4096x384xf32>
    %461 = "mhlo.fusion"(%arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %663 = mhlo.copy %662 : tensor<1x1x384x192xf32>
      mhlo.return %663 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %462 = mhlo.bitcast %461 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %463 = stablehlo.dot %460, %462, precision = [DEFAULT, DEFAULT] : (tensor<4096x384xf32>, tensor<384x192xf32>) -> tensor<4096x192xf32>
    %464 = mhlo.bitcast %463 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x192xf32>) -> tensor<16x4x64x192xf32>
    %465 = "mhlo.fusion"(%443, %464, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16x4x64x192xf32>, %arg247: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x192x4x64xf32>
      %667 = mhlo.bitcast %666 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
      mhlo.return %667 : tensor<1x16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>) -> tensor<1x16x49152xf32>
    %466 = "mhlo.fusion"(%465, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
      mhlo.return %662 : tensor<1x16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x49152xf32>, tensor<f32>) -> tensor<1x16x48xf32>
    %467 = "mhlo.fusion"(%466, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
      mhlo.return %662 : tensor<1x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x48xf32>, tensor<f32>) -> tensor<1x16x2xf32>
    %468 = "mhlo.fusion"(%467, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x2xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %469 = "mhlo.fusion"(%468, %443, %464, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x192x4x64xf32>, %arg247: tensor<16x4x64x192xf32>, %arg248: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %663 = stablehlo.add %arg247, %662 : tensor<16x4x64x192xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %665 = mhlo.copy %664 : tensor<16x192x4x64xf32>
      %666 = stablehlo.add %665, %arg246 : tensor<16x192x4x64xf32>
      %667 = mhlo.bitcast %666 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %668 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %669 = stablehlo.multiply %arg245, %668 : tensor<16xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %671 = stablehlo.subtract %667, %670 : tensor<16x49152xf32>
      %672 = stablehlo.multiply %671, %671 : tensor<16x49152xf32>
      mhlo.return %672 : tensor<16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>) -> tensor<16x49152xf32>
    %470 = "mhlo.fusion"(%469, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x49152xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
      mhlo.return %662 : tensor<16x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49152xf32>, tensor<f32>) -> tensor<16x1536xf32>
    %471 = "mhlo.fusion"(%470, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1536xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
      mhlo.return %662 : tensor<16x48xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1536xf32>, tensor<f32>) -> tensor<16x48xf32>
    %472 = "mhlo.fusion"(%471, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x48xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
      mhlo.return %662 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x48xf32>, tensor<f32>) -> tensor<16x2xf32>
    %473 = "mhlo.fusion"(%472) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>) -> tensor<16xf32>
    %474 = "mhlo.fusion"(%arg133, %473, %468, %443, %464, %arg132, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x192x4x64xf32>, %arg249: tensor<16x4x64x192xf32>, %arg250: tensor<192xf32>, %arg251: tensor<192xf32>):
      %662 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %663 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<192xf32>) -> tensor<16x4x64x192xf32>
      %664 = stablehlo.add %arg249, %663 : tensor<16x4x64x192xf32>
      %665 = stablehlo.transpose %664, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,192,4,64]{1,3,2,0}"} : (tensor<16x4x64x192xf32>) -> tensor<16x192x4x64xf32>
      %666 = mhlo.copy %665 : tensor<16x192x4x64xf32>
      %667 = stablehlo.add %666, %arg248 : tensor<16x192x4x64xf32>
      %668 = mhlo.bitcast %667 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %669 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %670 = stablehlo.multiply %arg247, %669 : tensor<16xf32>
      %671 = stablehlo.broadcast_in_dim %670, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %672 = stablehlo.subtract %668, %671 : tensor<16x49152xf32>
      %673 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x49152xf32>
      %674 = stablehlo.multiply %672, %673 : tensor<16x49152xf32>
      %675 = mhlo.bitcast %674 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x49152xf32>) -> tensor<16x192x4x64xf32>
      %676 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %677 = stablehlo.multiply %675, %676 : tensor<16x192x4x64xf32>
      %678 = stablehlo.add %662, %677 : tensor<16x192x4x64xf32>
      %679 = mhlo.bitcast %678 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x192x2x2x8x8xf32>
      %680 = stablehlo.transpose %679, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,8,2,8,16,192]{3,1,2,0,5,4}"} : (tensor<16x192x2x2x8x8xf32>) -> tensor<2x8x2x8x16x192xf32>
      %681 = mhlo.copy %680 : tensor<2x8x2x8x16x192xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[256,16,192,1,1]{2,1,0,4,3}"} : (tensor<2x8x2x8x16x192xf32>) -> tensor<256x16x192x1x1xf32>
      %683 = mhlo.copy %682 : tensor<256x16x192x1x1xf32>
      mhlo.return %683 : tensor<256x16x192x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x192x4x64xf32>, tensor<16x4x64x192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<256x16x192x1x1xf32>
    %475 = "mhlo.fusion"(%8, %9, %474) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x16x16xf32>, %arg246: tensor<256x2xi32>, %arg247: tensor<256x16x192x1x1xf32>):
      %662 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %663 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x192x16x16xf32>, tensor<256x2xi32>, tensor<256x16x192x1x1xf32>) -> tensor<16x192x16x16xf32>
      mhlo.return %662 : tensor<16x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x16x16xf32>, tensor<256x2xi32>, tensor<256x16x192x1x1xf32>) -> tensor<16x192x16x16xf32>
    %476 = "mhlo.fusion"(%475) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x192x16x16xf32>):
      %662 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,192]{2,1,3,0}"} : (tensor<16x192x16x16xf32>) -> tensor<16x16x16x192xf32>
      %663 = mhlo.copy %662 : tensor<16x16x16x192xf32>
      mhlo.return %663 : tensor<16x16x16x192xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x16x16xf32>) -> tensor<16x16x16x192xf32>
    %477 = mhlo.bitcast %476 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x16x16x192xf32>) -> tensor<4096x192xf32>
    %478 = "mhlo.fusion"(%arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %663 = mhlo.copy %662 : tensor<1x1x192x384xf32>
      mhlo.return %663 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %479 = mhlo.bitcast %478 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %480 = stablehlo.dot %477, %479, precision = [DEFAULT, DEFAULT] : (tensor<4096x192xf32>, tensor<192x384xf32>) -> tensor<4096x384xf32>
    %481 = mhlo.bitcast %480 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x384xf32>) -> tensor<16x16x16x384xf32>
    %482 = "mhlo.fusion"(%arg137, %arg136, %7, %481, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<16x16x16x384xf32>, %arg249: tensor<384xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x16x16x384xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x16x16x384xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x16x16x384xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<16x16x16x384xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x16x16x384xf32>
      mhlo.return %669 : tensor<16x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x16x16x384xf32>, tensor<384xf32>) -> tensor<16x16x16x384xf32>
    %483 = mhlo.bitcast %482 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x16x16x384xf32>) -> tensor<4096x384xf32>
    %484 = "mhlo.fusion"(%arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768x384x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,768]{1,0,2,3}"} : (tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32>
      %663 = mhlo.copy %662 : tensor<1x1x384x768xf32>
      mhlo.return %663 : tensor<1x1x384x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32>
    %485 = mhlo.bitcast %484 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x768xf32>) -> tensor<384x768xf32>
    %486 = stablehlo.dot %483, %485, precision = [DEFAULT, DEFAULT] : (tensor<4096x384xf32>, tensor<384x768xf32>) -> tensor<4096x768xf32>
    %487 = mhlo.bitcast %486 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x768xf32>) -> tensor<16x16x16x768xf32>
    %488 = "mhlo.fusion"(%arg140, %arg139, %6, %487, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<16x16x16x768xf32>, %arg249: tensor<768xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<768xf32>) -> tensor<16x16x16x768xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x16x16x768xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<768xf32>) -> tensor<16x16x16x768xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x16x16x768xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<768xf32>) -> tensor<16x16x16x768xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x16x16x768xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<768xf32>) -> tensor<16x16x16x768xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x16x16x768xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,768,16,16]{1,3,2,0}"} : (tensor<16x16x16x768xf32>) -> tensor<16x768x16x16xf32>
      %671 = mhlo.copy %670 : tensor<16x768x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x768x16x16xf32>
      %673 = stablehlo.negate %671 : tensor<16x768x16x16xf32>
      %674 = stablehlo.exponential %673 : tensor<16x768x16x16xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x768x16x16xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x768x16x16xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x768x16x16xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,768]{2,1,3,0}"} : (tensor<16x768x16x16xf32>) -> tensor<16x16x16x768xf32>
      %679 = mhlo.copy %678 : tensor<16x16x16x768xf32>
      mhlo.return %679 : tensor<16x16x16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x16x16x768xf32>, tensor<768xf32>) -> tensor<16x16x16x768xf32>
    %489 = "mhlo.fusion"(%arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,768]{1,0,2,3}"} : (tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x768xf32>
      mhlo.return %663 : tensor<3x3x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32>
    %490 = stablehlo.convolution(%488, %489) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x16x16x768xf32>, tensor<3x3x1x768xf32>) -> tensor<16x8x8x768xf32>
    %491 = "mhlo.fusion"(%arg143, %arg142, %5, %490, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<16x8x8x768xf32>, %arg249: tensor<768xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<768xf32>) -> tensor<16x8x8x768xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x8x8x768xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<768xf32>) -> tensor<16x8x8x768xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x8x8x768xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<768xf32>) -> tensor<16x8x8x768xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x8x8x768xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<768xf32>) -> tensor<16x8x8x768xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x8x8x768xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,768,8,8]{1,3,2,0}"} : (tensor<16x8x8x768xf32>) -> tensor<16x768x8x8xf32>
      %671 = mhlo.copy %670 : tensor<16x768x8x8xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x768x8x8xf32>
      %673 = stablehlo.negate %671 : tensor<16x768x8x8xf32>
      %674 = stablehlo.exponential %673 : tensor<16x768x8x8xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x768x8x8xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x768x8x8xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x768x8x8xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,8,768]{2,1,3,0}"} : (tensor<16x768x8x8xf32>) -> tensor<16x8x8x768xf32>
      %679 = mhlo.copy %678 : tensor<16x8x8x768xf32>
      mhlo.return %679 : tensor<16x8x8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x8x8x768xf32>, tensor<768xf32>) -> tensor<16x8x8x768xf32>
    %492 = mhlo.bitcast %491 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x8x768xf32>) -> tensor<1024x768xf32>
    %493 = "mhlo.fusion"(%arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x768x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,768,512]{1,0,2,3}"} : (tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32>
      %663 = mhlo.copy %662 : tensor<1x1x768x512xf32>
      mhlo.return %663 : tensor<1x1x768x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32>
    %494 = mhlo.bitcast %493 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x768x512xf32>) -> tensor<768x512xf32>
    %495 = stablehlo.dot %492, %494, precision = [DEFAULT, DEFAULT] : (tensor<1024x768xf32>, tensor<768x512xf32>) -> tensor<1024x512xf32>
    %496 = mhlo.bitcast %495 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x512xf32>) -> tensor<16x8x8x512xf32>
    %497 = "mhlo.fusion"(%arg146, %arg145, %4, %496, %arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<16x8x8x512xf32>, %arg249: tensor<512xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x8x8x512xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x8x8x512xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x8x8x512xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x8x8x512xf32>
      mhlo.return %669 : tensor<16x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x8x8x512xf32>, tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %498 = "mhlo.fusion"(%arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x3x3xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
      %663 = mhlo.copy %662 : tensor<3x3x1x512xf32>
      mhlo.return %663 : tensor<3x3x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %499 = stablehlo.convolution(%497, %498) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x8x8x512xf32>, tensor<3x3x1x512xf32>) -> tensor<16x8x8x512xf32>
    %500 = "mhlo.fusion"(%arg149, %arg148, %3, %499, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<16x8x8x512xf32>, %arg249: tensor<512xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x8x8x512xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x8x8x512xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x8x8x512xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x8x8x512xf32>
      %670 = stablehlo.transpose %669, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,8,8]{1,3,2,0}"} : (tensor<16x8x8x512xf32>) -> tensor<16x512x8x8xf32>
      %671 = mhlo.copy %670 : tensor<16x512x8x8xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512x8x8xf32>
      %673 = stablehlo.negate %671 : tensor<16x512x8x8xf32>
      %674 = stablehlo.exponential %673 : tensor<16x512x8x8xf32>
      %675 = stablehlo.add %672, %674 : tensor<16x512x8x8xf32>
      %676 = stablehlo.divide %672, %675 : tensor<16x512x8x8xf32>
      %677 = stablehlo.multiply %671, %676 : tensor<16x512x8x8xf32>
      %678 = stablehlo.transpose %677, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,8,512]{2,1,3,0}"} : (tensor<16x512x8x8xf32>) -> tensor<16x8x8x512xf32>
      %679 = mhlo.copy %678 : tensor<16x8x8x512xf32>
      mhlo.return %679 : tensor<16x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x8x8x512xf32>, tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %501 = mhlo.bitcast %500 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x8x512xf32>) -> tensor<1024x512xf32>
    %502 = "mhlo.fusion"(%arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x512x256xf32>
      mhlo.return %663 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %503 = mhlo.bitcast %502 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %504 = stablehlo.dot %501, %503, precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32>
    %505 = mhlo.bitcast %504 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<16x8x8x256xf32>
    %506 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %662 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %663 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %664 = stablehlo.multiply %662, %663 : tensor<4xi32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %666 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %667 = stablehlo.add %665, %666 : tensor<2x4xi32>
      %668 = mhlo.bitcast %667 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
      %669 = mhlo.copy %668 : tensor<2x4x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %670 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %671 = stablehlo.compare LT, %669, %670 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
      %c_2 = stablehlo.constant dense<8> : tensor<i32>
      %672 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %673 = stablehlo.add %669, %672 : tensor<2x4x1x1xi32>
      %674 = stablehlo.select %671, %673, %669 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
      %675 = mhlo.bitcast %674 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %677 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %678 = stablehlo.compare LT, %667, %677 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %679 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %680 = stablehlo.add %667, %679 : tensor<2x4xi32>
      %681 = stablehlo.select %678, %680, %667 : tensor<2x4xi1>, tensor<2x4xi32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %683 = stablehlo.concatenate %676, %682, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      mhlo.return %683 : tensor<2x4x2x4x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x4x2x4x2xi32>
    %507 = "mhlo.fusion"(%505, %506) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x8x8x256xf32>, %arg246: tensor<2x4x2x4x2xi32>):
      %662 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,8,8]{1,3,2,0}"} : (tensor<16x8x8x256xf32>) -> tensor<16x256x8x8xf32>
      %663 = mhlo.copy %662 : tensor<16x256x8x8xf32>
      %664 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %665 = "stablehlo.gather"(%663, %664) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x16x256x1x1xf32>
      mhlo.return %665 : tensor<64x16x256x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x8x256xf32>, tensor<2x4x2x4x2xi32>) -> tensor<64x16x256x1x1xf32>
    %508 = "mhlo.fusion"(%507) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x16x256x1x1xf32>):
      %662 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
      %663 = stablehlo.transpose %662, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
      %664 = mhlo.copy %663 : tensor<16x256x2x2x4x4xf32>
      %665 = mhlo.bitcast %664 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<1x16x16384xf32>
      mhlo.return %665 : tensor<1x16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x256x1x1xf32>) -> tensor<1x16x16384xf32>
    %509 = "mhlo.fusion"(%508, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %662 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %510 = "mhlo.fusion"(%509, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %511 = "mhlo.fusion"(%510, %507) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<64x16x256x1x1xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
      %663 = stablehlo.transpose %662, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
      %664 = mhlo.copy %663 : tensor<16x256x2x2x4x4xf32>
      %665 = mhlo.bitcast %664 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %667 = stablehlo.multiply %arg245, %666 : tensor<16xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %669 = stablehlo.subtract %665, %668 : tensor<16x16384xf32>
      %670 = stablehlo.multiply %669, %669 : tensor<16x16384xf32>
      mhlo.return %670 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<64x16x256x1x1xf32>) -> tensor<16x16384xf32>
    %512 = "mhlo.fusion"(%511, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %662 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    %513 = "mhlo.fusion"(%512, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %662 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    %514 = "mhlo.fusion"(%513) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>) -> tensor<16xf32>
    %515 = "mhlo.fusion"(%arg151, %514, %510, %507, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<64x16x256x1x1xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
      %664 = stablehlo.transpose %663, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
      %665 = mhlo.copy %664 : tensor<16x256x2x2x4x4xf32>
      %666 = mhlo.bitcast %665 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %668 = stablehlo.multiply %arg247, %667 : tensor<16xf32>
      %669 = stablehlo.broadcast_in_dim %668, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %670 = stablehlo.subtract %666, %669 : tensor<16x16384xf32>
      %671 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x16384xf32>
      %673 = mhlo.bitcast %672 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
      %674 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %675 = stablehlo.multiply %673, %674 : tensor<16x256x4x16xf32>
      %676 = stablehlo.add %662, %675 : tensor<16x256x4x16xf32>
      %677 = stablehlo.transpose %676, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %678 = mhlo.copy %677 : tensor<16x4x16x256xf32>
      mhlo.return %678 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<16xf32>, tensor<16xf32>, tensor<64x16x256x1x1xf32>, tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %516 = mhlo.bitcast %515 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %517 = "mhlo.fusion"(%arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x513xf32>
      mhlo.return %663 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %518 = mhlo.bitcast %517 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %519 = stablehlo.dot %516, %518, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x513xf32>) -> tensor<1024x513xf32>
    %520 = mhlo.bitcast %519 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x513xf32>) -> tensor<16x4x16x513xf32>
    %521 = "mhlo.fusion"(%520, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x16x513xf32>, %arg246: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.reduce(%666 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %667 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x1x4xf32>
    %522 = "mhlo.fusion"(%521, %520, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x16xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x16xf32>
      mhlo.return %672 : tensor<16x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x1x4x16xf32>
    %523 = "mhlo.fusion"(%522, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %524 = "mhlo.fusion"(%522, %523) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x16xf32>
      mhlo.return %664 : tensor<16x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x16xf32>
    %525 = "mhlo.fusion"(%524, %520, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %670 = stablehlo.reduce(%669 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
      mhlo.return %670 : tensor<16x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x256x4xf32>
    %526 = "mhlo.fusion"(%525, %520, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x256x4x16xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x256x4x16xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %672 = mhlo.copy %671 : tensor<16x4x16x256xf32>
      mhlo.return %672 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x4x16x256xf32>
    %527 = mhlo.bitcast %526 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %528 = "mhlo.fusion"(%arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x256xf32>
      mhlo.return %663 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %529 = mhlo.bitcast %528 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %530 = stablehlo.dot %527, %529, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x256xf32>) -> tensor<1024x256xf32>
    %531 = mhlo.bitcast %530 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<16x4x16x256xf32>
    %532 = "mhlo.fusion"(%531, %arg156, %507) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x16x256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x16x256x1x1xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x16x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x16xf32>
      %666 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
      %667 = stablehlo.transpose %666, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[16,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
      %668 = mhlo.copy %667 : tensor<16x256x2x2x4x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x256x4x16xf32>
      %670 = stablehlo.add %665, %669 : tensor<16x256x4x16xf32>
      mhlo.return %670 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x16x256xf32>, tensor<256xf32>, tensor<64x16x256x1x1xf32>) -> tensor<16x256x4x16xf32>
    %533 = mhlo.bitcast %532 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
    %534 = "mhlo.fusion"(%533, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %662 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %535 = "mhlo.fusion"(%534, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %536 = "mhlo.fusion"(%535, %532) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x256x4x16xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x16384xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x16384xf32>
      mhlo.return %667 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %537 = "mhlo.fusion"(%536, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %662 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    %538 = "mhlo.fusion"(%537, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %662 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    %539 = "mhlo.fusion"(%538) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>) -> tensor<16xf32>
    %540 = "mhlo.fusion"(%arg157, %539, %535, %532, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x256x4x16xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x16384xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x16384xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x256x4x16xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x256x4x16xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %675 = mhlo.copy %674 : tensor<16x4x16x256xf32>
      mhlo.return %675 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x256x4x16xf32>, tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %541 = mhlo.bitcast %540 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %542 = "mhlo.fusion"(%arg159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x512xf32>
      mhlo.return %663 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %543 = mhlo.bitcast %542 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %544 = stablehlo.dot %541, %543, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x512xf32>) -> tensor<1024x512xf32>
    %545 = mhlo.bitcast %544 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x512xf32>) -> tensor<16x4x16x512xf32>
    %546 = "mhlo.fusion"(%545, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x16x512xf32>, %arg246: tensor<512xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<16x4x16x512xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x16x512xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,4,16]{1,3,2,0}"} : (tensor<16x4x16x512xf32>) -> tensor<16x512x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
      %667 = stablehlo.negate %665 : tensor<16x512x4x16xf32>
      %668 = stablehlo.exponential %667 : tensor<16x512x4x16xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x512x4x16xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x512x4x16xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x512x4x16xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,512]{2,1,3,0}"} : (tensor<16x512x4x16xf32>) -> tensor<16x4x16x512xf32>
      %673 = mhlo.copy %672 : tensor<16x4x16x512xf32>
      mhlo.return %673 : tensor<16x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x16x512xf32>, tensor<512xf32>) -> tensor<16x4x16x512xf32>
    %547 = mhlo.bitcast %546 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x512xf32>) -> tensor<1024x512xf32>
    %548 = "mhlo.fusion"(%arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x512x256xf32>
      mhlo.return %663 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %549 = mhlo.bitcast %548 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %550 = stablehlo.dot %547, %549, precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32>
    %551 = mhlo.bitcast %550 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<16x4x16x256xf32>
    %552 = "mhlo.fusion"(%532, %551, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16x4x16x256xf32>, %arg247: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x16xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x256x4x16xf32>
      mhlo.return %666 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16x4x16x256xf32>, tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %553 = mhlo.bitcast %552 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
    %554 = "mhlo.fusion"(%553, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %662 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %555 = "mhlo.fusion"(%554, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %556 = "mhlo.fusion"(%555, %552) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x256x4x16xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x16384xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x16384xf32>
      mhlo.return %667 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %557 = "mhlo.fusion"(%556, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %662 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    %558 = "mhlo.fusion"(%557, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %662 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    %559 = "mhlo.fusion"(%558) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>) -> tensor<16xf32>
    %560 = "mhlo.fusion"(%arg163, %559, %555, %552, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x256x4x16xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x16384xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x16384xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x256x4x16xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x256x4x16xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %675 = mhlo.copy %674 : tensor<16x4x16x256xf32>
      mhlo.return %675 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x256x4x16xf32>, tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %561 = mhlo.bitcast %560 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %562 = "mhlo.fusion"(%arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x513xf32>
      mhlo.return %663 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %563 = mhlo.bitcast %562 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %564 = stablehlo.dot %561, %563, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x513xf32>) -> tensor<1024x513xf32>
    %565 = mhlo.bitcast %564 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x513xf32>) -> tensor<16x4x16x513xf32>
    %566 = "mhlo.fusion"(%565, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x16x513xf32>, %arg246: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.reduce(%666 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %667 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x1x4xf32>
    %567 = "mhlo.fusion"(%566, %565, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x16xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x16xf32>
      mhlo.return %672 : tensor<16x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x1x4x16xf32>
    %568 = "mhlo.fusion"(%567, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %569 = "mhlo.fusion"(%567, %568) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x16xf32>
      mhlo.return %664 : tensor<16x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x16xf32>
    %570 = "mhlo.fusion"(%569, %565, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %670 = stablehlo.reduce(%669 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
      mhlo.return %670 : tensor<16x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x256x4xf32>
    %571 = "mhlo.fusion"(%570, %565, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x256x4x16xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x256x4x16xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %672 = mhlo.copy %671 : tensor<16x4x16x256xf32>
      mhlo.return %672 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x4x16x256xf32>
    %572 = mhlo.bitcast %571 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %573 = "mhlo.fusion"(%arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x256xf32>
      mhlo.return %663 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %574 = mhlo.bitcast %573 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %575 = stablehlo.dot %572, %574, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x256xf32>) -> tensor<1024x256xf32>
    %576 = mhlo.bitcast %575 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<16x4x16x256xf32>
    %577 = "mhlo.fusion"(%552, %576, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16x4x16x256xf32>, %arg247: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x16xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x256x4x16xf32>
      mhlo.return %666 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16x4x16x256xf32>, tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %578 = mhlo.bitcast %577 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
    %579 = "mhlo.fusion"(%578, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %662 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %580 = "mhlo.fusion"(%579, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %581 = "mhlo.fusion"(%580, %577) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x256x4x16xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x16384xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x16384xf32>
      mhlo.return %667 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %582 = "mhlo.fusion"(%581, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %662 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    %583 = "mhlo.fusion"(%582, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %662 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    %584 = "mhlo.fusion"(%583) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>) -> tensor<16xf32>
    %585 = "mhlo.fusion"(%arg169, %584, %580, %577, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x256x4x16xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x16384xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x16384xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x256x4x16xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x256x4x16xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %675 = mhlo.copy %674 : tensor<16x4x16x256xf32>
      mhlo.return %675 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x256x4x16xf32>, tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %586 = mhlo.bitcast %585 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %587 = "mhlo.fusion"(%arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x512xf32>
      mhlo.return %663 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %588 = mhlo.bitcast %587 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %589 = stablehlo.dot %586, %588, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x512xf32>) -> tensor<1024x512xf32>
    %590 = mhlo.bitcast %589 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x512xf32>) -> tensor<16x4x16x512xf32>
    %591 = "mhlo.fusion"(%590, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x16x512xf32>, %arg246: tensor<512xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<16x4x16x512xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x16x512xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,4,16]{1,3,2,0}"} : (tensor<16x4x16x512xf32>) -> tensor<16x512x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
      %667 = stablehlo.negate %665 : tensor<16x512x4x16xf32>
      %668 = stablehlo.exponential %667 : tensor<16x512x4x16xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x512x4x16xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x512x4x16xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x512x4x16xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,512]{2,1,3,0}"} : (tensor<16x512x4x16xf32>) -> tensor<16x4x16x512xf32>
      %673 = mhlo.copy %672 : tensor<16x4x16x512xf32>
      mhlo.return %673 : tensor<16x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x16x512xf32>, tensor<512xf32>) -> tensor<16x4x16x512xf32>
    %592 = mhlo.bitcast %591 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x512xf32>) -> tensor<1024x512xf32>
    %593 = "mhlo.fusion"(%arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x512x256xf32>
      mhlo.return %663 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %594 = mhlo.bitcast %593 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %595 = stablehlo.dot %592, %594, precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32>
    %596 = mhlo.bitcast %595 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<16x4x16x256xf32>
    %597 = "mhlo.fusion"(%577, %596, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16x4x16x256xf32>, %arg247: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x16xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x256x4x16xf32>
      mhlo.return %666 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16x4x16x256xf32>, tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %598 = mhlo.bitcast %597 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
    %599 = "mhlo.fusion"(%598, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %662 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %600 = "mhlo.fusion"(%599, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %601 = "mhlo.fusion"(%600, %597) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x256x4x16xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x16384xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x16384xf32>
      mhlo.return %667 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %602 = "mhlo.fusion"(%601, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %662 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    %603 = "mhlo.fusion"(%602, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %662 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    %604 = "mhlo.fusion"(%603) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>) -> tensor<16xf32>
    %605 = "mhlo.fusion"(%arg175, %604, %600, %597, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x256x4x16xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x16384xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x16384xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x256x4x16xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x256x4x16xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %675 = mhlo.copy %674 : tensor<16x4x16x256xf32>
      mhlo.return %675 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x256x4x16xf32>, tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %606 = mhlo.bitcast %605 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %607 = "mhlo.fusion"(%arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x513xf32>
      mhlo.return %663 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %608 = mhlo.bitcast %607 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %609 = stablehlo.dot %606, %608, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x513xf32>) -> tensor<1024x513xf32>
    %610 = mhlo.bitcast %609 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x513xf32>) -> tensor<16x4x16x513xf32>
    %611 = "mhlo.fusion"(%610, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x16x513xf32>, %arg246: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.reduce(%666 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %667 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x1x4xf32>
    %612 = "mhlo.fusion"(%611, %610, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x1x4xf32>
      %668 = stablehlo.maximum %667, %arg245 : tensor<16x1x4xf32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %671 = stablehlo.subtract %666, %670 : tensor<16x1x4x16xf32>
      %672 = stablehlo.exponential %671 : tensor<16x1x4x16xf32>
      mhlo.return %672 : tensor<16x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x1x4x16xf32>
    %613 = "mhlo.fusion"(%612, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
      mhlo.return %662 : tensor<16x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<f32>) -> tensor<16x1x4xf32>
    %614 = "mhlo.fusion"(%612, %613) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<16x1x4xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x4xf32>) -> tensor<16x4xf32>
      %663 = stablehlo.broadcast_in_dim %662, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %664 = stablehlo.divide %arg245, %663 : tensor<16x1x4x16xf32>
      mhlo.return %664 : tensor<16x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<16x1x4xf32>) -> tensor<16x1x4x16xf32>
    %615 = "mhlo.fusion"(%614, %610, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x1x4x16xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %667 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
      %669 = stablehlo.multiply %666, %668 : tensor<16x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %670 = stablehlo.reduce(%669 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
      mhlo.return %670 : tensor<16x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x4x16xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x256x4xf32>
    %616 = "mhlo.fusion"(%615, %610, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4xf32>, %arg246: tensor<16x4x16x513xf32>, %arg247: tensor<513xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<16x4x16x513xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x513xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,513,4,16]{1,3,2,0}"} : (tensor<16x4x16x513xf32>) -> tensor<16x513x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x513x4x16xf32>
      %666 = stablehlo.slice %665 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %667 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
      %668 = stablehlo.maximum %666, %667 : tensor<16x256x4x16xf32>
      %669 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<16x256x4x16xf32>
      %671 = stablehlo.transpose %670, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %672 = mhlo.copy %671 : tensor<16x4x16x256xf32>
      mhlo.return %672 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4xf32>, tensor<16x4x16x513xf32>, tensor<513xf32>) -> tensor<16x4x16x256xf32>
    %617 = mhlo.bitcast %616 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %618 = "mhlo.fusion"(%arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x256xf32>
      mhlo.return %663 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %619 = mhlo.bitcast %618 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %620 = stablehlo.dot %617, %619, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x256xf32>) -> tensor<1024x256xf32>
    %621 = mhlo.bitcast %620 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<16x4x16x256xf32>
    %622 = "mhlo.fusion"(%597, %621, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16x4x16x256xf32>, %arg247: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x16xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x256x4x16xf32>
      mhlo.return %666 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16x4x16x256xf32>, tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %623 = mhlo.bitcast %622 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
    %624 = "mhlo.fusion"(%623, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %662 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %625 = "mhlo.fusion"(%624, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %626 = "mhlo.fusion"(%625, %622) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x256x4x16xf32>):
      %662 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %arg245, %663 : tensor<16xf32>
      %665 = stablehlo.broadcast_in_dim %664, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %666 = stablehlo.subtract %662, %665 : tensor<16x16384xf32>
      %667 = stablehlo.multiply %666, %666 : tensor<16x16384xf32>
      mhlo.return %667 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %627 = "mhlo.fusion"(%626, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %662 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    %628 = "mhlo.fusion"(%627, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %662 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    %629 = "mhlo.fusion"(%628) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>) -> tensor<16xf32>
    %630 = "mhlo.fusion"(%arg181, %629, %625, %622, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x256x4x16xf32>, %arg249: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %663 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %664 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %665 = stablehlo.multiply %arg247, %664 : tensor<16xf32>
      %666 = stablehlo.broadcast_in_dim %665, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %667 = stablehlo.subtract %663, %666 : tensor<16x16384xf32>
      %668 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<16x16384xf32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
      %671 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %672 = stablehlo.multiply %670, %671 : tensor<16x256x4x16xf32>
      %673 = stablehlo.add %662, %672 : tensor<16x256x4x16xf32>
      %674 = stablehlo.transpose %673, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,256]{2,1,3,0}"} : (tensor<16x256x4x16xf32>) -> tensor<16x4x16x256xf32>
      %675 = mhlo.copy %674 : tensor<16x4x16x256xf32>
      mhlo.return %675 : tensor<16x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x256x4x16xf32>, tensor<256xf32>) -> tensor<16x4x16x256xf32>
    %631 = mhlo.bitcast %630 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x256xf32>) -> tensor<1024x256xf32>
    %632 = "mhlo.fusion"(%arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x512xf32>
      mhlo.return %663 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %633 = mhlo.bitcast %632 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %634 = stablehlo.dot %631, %633, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x512xf32>) -> tensor<1024x512xf32>
    %635 = mhlo.bitcast %634 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x512xf32>) -> tensor<16x4x16x512xf32>
    %636 = "mhlo.fusion"(%635, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x4x16x512xf32>, %arg246: tensor<512xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<16x4x16x512xf32>
      %663 = stablehlo.add %arg245, %662 : tensor<16x4x16x512xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,4,16]{1,3,2,0}"} : (tensor<16x4x16x512xf32>) -> tensor<16x512x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
      %667 = stablehlo.negate %665 : tensor<16x512x4x16xf32>
      %668 = stablehlo.exponential %667 : tensor<16x512x4x16xf32>
      %669 = stablehlo.add %666, %668 : tensor<16x512x4x16xf32>
      %670 = stablehlo.divide %666, %669 : tensor<16x512x4x16xf32>
      %671 = stablehlo.multiply %665, %670 : tensor<16x512x4x16xf32>
      %672 = stablehlo.transpose %671, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,4,16,512]{2,1,3,0}"} : (tensor<16x512x4x16xf32>) -> tensor<16x4x16x512xf32>
      %673 = mhlo.copy %672 : tensor<16x4x16x512xf32>
      mhlo.return %673 : tensor<16x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4x16x512xf32>, tensor<512xf32>) -> tensor<16x4x16x512xf32>
    %637 = mhlo.bitcast %636 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4x16x512xf32>) -> tensor<1024x512xf32>
    %638 = "mhlo.fusion"(%arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %663 = mhlo.copy %662 : tensor<1x1x512x256xf32>
      mhlo.return %663 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %639 = mhlo.bitcast %638 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %640 = stablehlo.dot %637, %639, precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32>
    %641 = mhlo.bitcast %640 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<16x4x16x256xf32>
    %642 = "mhlo.fusion"(%622, %641, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16x4x16x256xf32>, %arg247: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
      %663 = stablehlo.add %arg246, %662 : tensor<16x4x16x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x16xf32>
      %666 = stablehlo.add %665, %arg245 : tensor<16x256x4x16xf32>
      %667 = mhlo.bitcast %666 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
      mhlo.return %667 : tensor<1x16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16x4x16x256xf32>, tensor<256xf32>) -> tensor<1x16x16384xf32>
    %643 = "mhlo.fusion"(%642, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %662 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16384xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %644 = "mhlo.fusion"(%643, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %662 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %645 = "mhlo.fusion"(%644, %622, %641, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>, %arg246: tensor<16x256x4x16xf32>, %arg247: tensor<16x4x16x256xf32>, %arg248: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
      %663 = stablehlo.add %arg247, %662 : tensor<16x4x16x256xf32>
      %664 = stablehlo.transpose %663, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
      %665 = mhlo.copy %664 : tensor<16x256x4x16xf32>
      %666 = stablehlo.add %665, %arg246 : tensor<16x256x4x16xf32>
      %667 = mhlo.bitcast %666 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %668 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %669 = stablehlo.multiply %arg245, %668 : tensor<16xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %671 = stablehlo.subtract %667, %670 : tensor<16x16384xf32>
      %672 = stablehlo.multiply %671, %671 : tensor<16x16384xf32>
      mhlo.return %672 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16x256x4x16xf32>, tensor<16x4x16x256xf32>, tensor<256xf32>) -> tensor<16x16384xf32>
    %646 = "mhlo.fusion"(%645, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16384xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %662 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16x512xf32>
    %647 = "mhlo.fusion"(%646, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512xf32>, %arg246: tensor<f32>):
      %662 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %663 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %662 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>, tensor<f32>) -> tensor<16x16xf32>
    %648 = "mhlo.fusion"(%647) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %664 = stablehlo.multiply %662, %663 : tensor<16xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %666 = stablehlo.add %664, %665 : tensor<16xf32>
      %667 = stablehlo.rsqrt %666 : tensor<16xf32>
      mhlo.return %667 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>) -> tensor<16xf32>
    %649 = "mhlo.fusion"(%arg187, %648, %644, %622, %641, %arg186, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<16xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x256x4x16xf32>, %arg249: tensor<16x4x16x256xf32>, %arg250: tensor<256xf32>, %arg251: tensor<256xf32>):
      %662 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %663 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<256xf32>) -> tensor<16x4x16x256xf32>
      %664 = stablehlo.add %arg249, %663 : tensor<16x4x16x256xf32>
      %665 = stablehlo.transpose %664, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,4,16]{1,3,2,0}"} : (tensor<16x4x16x256xf32>) -> tensor<16x256x4x16xf32>
      %666 = mhlo.copy %665 : tensor<16x256x4x16xf32>
      %667 = stablehlo.add %666, %arg248 : tensor<16x256x4x16xf32>
      %668 = mhlo.bitcast %667 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %669 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %670 = stablehlo.multiply %arg247, %669 : tensor<16xf32>
      %671 = stablehlo.broadcast_in_dim %670, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %672 = stablehlo.subtract %668, %671 : tensor<16x16384xf32>
      %673 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x16384xf32>
      %674 = stablehlo.multiply %672, %673 : tensor<16x16384xf32>
      %675 = mhlo.bitcast %674 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16384xf32>) -> tensor<16x256x4x16xf32>
      %676 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %677 = stablehlo.multiply %675, %676 : tensor<16x256x4x16xf32>
      %678 = stablehlo.add %662, %677 : tensor<16x256x4x16xf32>
      %679 = mhlo.bitcast %678 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x256x2x2x4x4xf32>
      %680 = stablehlo.transpose %679, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,4,2,4,16,256]{3,1,2,0,5,4}"} : (tensor<16x256x2x2x4x4xf32>) -> tensor<2x4x2x4x16x256xf32>
      %681 = mhlo.copy %680 : tensor<2x4x2x4x16x256xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[64,16,256,1,1]{2,1,0,4,3}"} : (tensor<2x4x2x4x16x256xf32>) -> tensor<64x16x256x1x1xf32>
      %683 = mhlo.copy %682 : tensor<64x16x256x1x1xf32>
      mhlo.return %683 : tensor<64x16x256x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x256x4x16xf32>, tensor<16x4x16x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x16x256x1x1xf32>
    %650 = "mhlo.fusion"(%1, %2, %649) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x8x8xf32>, %arg246: tensor<64x2xi32>, %arg247: tensor<64x16x256x1x1xf32>):
      %662 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %663 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %663 : tensor<f32>
      }) : (tensor<16x256x8x8xf32>, tensor<64x2xi32>, tensor<64x16x256x1x1xf32>) -> tensor<16x256x8x8xf32>
      mhlo.return %662 : tensor<16x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8x8xf32>, tensor<64x2xi32>, tensor<64x16x256x1x1xf32>) -> tensor<16x256x8x8xf32>
    %651 = "mhlo.fusion"(%650) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x8x8xf32>):
      %662 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,8,256]{2,1,3,0}"} : (tensor<16x256x8x8xf32>) -> tensor<16x8x8x256xf32>
      %663 = mhlo.copy %662 : tensor<16x8x8x256xf32>
      mhlo.return %663 : tensor<16x8x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8x8xf32>) -> tensor<16x8x8x256xf32>
    %652 = mhlo.bitcast %651 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x8x256xf32>) -> tensor<1024x256xf32>
    %653 = "mhlo.fusion"(%arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %662 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %663 = mhlo.copy %662 : tensor<1x1x256x512xf32>
      mhlo.return %663 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %654 = mhlo.bitcast %653 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %655 = stablehlo.dot %652, %654, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x512xf32>) -> tensor<1024x512xf32>
    %656 = mhlo.bitcast %655 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x512xf32>) -> tensor<16x8x8x512xf32>
    %657 = "mhlo.fusion"(%arg191, %arg190, %0, %656, %arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<16x8x8x512xf32>, %arg249: tensor<512xf32>):
      %662 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %663 = stablehlo.subtract %arg248, %662 : tensor<16x8x8x512xf32>
      %664 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %665 = stablehlo.multiply %663, %664 : tensor<16x8x8x512xf32>
      %666 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %667 = stablehlo.multiply %665, %666 : tensor<16x8x8x512xf32>
      %668 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<16x8x8x512xf32>
      %669 = stablehlo.add %667, %668 : tensor<16x8x8x512xf32>
      mhlo.return %669 : tensor<16x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x8x8x512xf32>, tensor<512xf32>) -> tensor<16x8x8x512xf32>
    %658 = "mhlo.fusion"(%657, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x8x8x512xf32>, %arg246: tensor<f32>):
      %662 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<16x8x8x512xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %662 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x8x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %659 = "mhlo.fusion"(%658) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512xf32>):
      %cst_1 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %662 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
      %663 = stablehlo.multiply %arg245, %662 : tensor<16x512xf32>
      mhlo.return %663 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>) -> tensor<16x512xf32>
    %660 = stablehlo.dot_general %659, %arg192, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x512xf32>, tensor<2x512xf32>) -> tensor<16x2xf32>
    %661 = "mhlo.fusion"(%660, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>, %arg246: tensor<2xf32>):
      %662 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<2xf32>) -> tensor<16x2xf32>
      %663 = stablehlo.add %662, %arg245 : tensor<16x2xf32>
      mhlo.return %663 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>, tensor<2xf32>) -> tensor<16x2xf32>
    return %661 : tensor<16x2xf32>
  }
}
