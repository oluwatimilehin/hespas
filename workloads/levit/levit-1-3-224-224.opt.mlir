module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_computation(%arg0: tensor<196x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<256x128xf32>) -> tensor<196x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<128xf32>) -> tensor<196x128xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : tensor<196x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<128xf32>) -> tensor<196x128xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : tensor<196x128xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,256]{0,1}"} : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    return %4 : tensor<196x256xf32>
  }
  func.func private @gemm_fusion_dot_general.70_computation(%arg0: tensor<4x16x196xf32>, %arg1: tensor<4x16x196xf32>) -> tensor<4x196x196xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x196xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<4x196x196xf32>
    return %2 : tensor<4x196x196xf32>
  }
  func.func private @gemm_fusion_dot_general.71_computation(%arg0: tensor<4x196x196xf32>, %arg1: tensor<4x32x196xf32>) -> tensor<196x4x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x196x196xf32>, tensor<4x32x196xf32>) -> tensor<4x196x32xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<4x196x32xf32>) -> tensor<196x4x32xf32>
    return %1 : tensor<196x4x32xf32>
  }
  func.func private @gemm_fusion_dot_general.72_computation(%arg0: tensor<196x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<196x128xf32> {
    %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x128xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<196x128xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
    return %2 : tensor<196x128xf32>
  }
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<196x128xf32>, %arg1: tensor<256x128xf32>) -> tensor<196x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    return %0 : tensor<196x256xf32>
  }
  func.func private @gemm_fusion_dot_general.74_computation(%arg0: tensor<196x256xf32>, %arg1: tensor<128x256xf32>) -> tensor<196x128xf32> {
    %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<196x256xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
    return %2 : tensor<196x128xf32>
  }
  func.func private @gemm_fusion_dot.2_computation(%arg0: tensor<196x128xf32>, %arg1: tensor<256x128xf32>) -> tensor<196x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,256]{0,1}"} : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    return %0 : tensor<196x256xf32>
  }
  func.func private @gemm_fusion_dot_general.76_computation(%arg0: tensor<4x16x196xf32>, %arg1: tensor<4x16x196xf32>) -> tensor<4x196x196xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x196xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<4x196x196xf32>
    return %2 : tensor<4x196x196xf32>
  }
  func.func private @gemm_fusion_dot_general.77_computation(%arg0: tensor<4x196x196xf32>, %arg1: tensor<4x32x196xf32>) -> tensor<196x4x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x196x196xf32>, tensor<4x32x196xf32>) -> tensor<4x196x32xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<4x196x32xf32>) -> tensor<196x4x32xf32>
    return %1 : tensor<196x4x32xf32>
  }
  func.func private @gemm_fusion_dot_general.78_computation(%arg0: tensor<196x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<196x128xf32> {
    %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x128xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<196x128xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
    return %2 : tensor<196x128xf32>
  }
  func.func private @gemm_fusion_dot.3_computation(%arg0: tensor<196x128xf32>, %arg1: tensor<256x128xf32>) -> tensor<196x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    return %0 : tensor<196x256xf32>
  }
  func.func private @gemm_fusion_dot_general.80_computation(%arg0: tensor<196x256xf32>, %arg1: tensor<128x256xf32>) -> tensor<196x128xf32> {
    %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<196x256xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
    return %2 : tensor<196x128xf32>
  }
  func.func private @gemm_fusion_dot_general.82_computation(%arg0: tensor<49x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,49]{0,1}"} : (tensor<128x128xf32>, tensor<49x128xf32>) -> tensor<128x49xf32>
    return %0 : tensor<128x49xf32>
  }
  func.func private @gemm_fusion_dot.4_computation(%arg0: tensor<196x128xf32>, %arg1: tensor<640x128xf32>) -> tensor<196x640xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,640]{0,1}"} : (tensor<196x128xf32>, tensor<640x128xf32>) -> tensor<196x640xf32>
    return %0 : tensor<196x640xf32>
  }
  func.func private @gemm_fusion_dot_general.83_computation(%arg0: tensor<49x8x16xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<8x16xf32>, %arg3: tensor<8x16x196xf32>) -> tensor<8x196x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<49x8x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<49x8x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<49x8x16xf32>
    %3 = stablehlo.add %1, %2 : tensor<49x8x16xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<49x8x16xf32>) -> tensor<8x49x16xf32>
    %5 = stablehlo.dot_general %arg3, %4, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,196,49]{1,2,0}"} : (tensor<8x16x196xf32>, tensor<8x49x16xf32>) -> tensor<8x196x49xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,196,49]{1,2,0}"} : (tensor<f32>) -> tensor<8x196x49xf32>
    %7 = stablehlo.multiply %5, %6 {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,196,49]{1,2,0}"} : tensor<8x196x49xf32>
    return %7 : tensor<8x196x49xf32>
  }
  func.func private @gemm_fusion_dot_general.84_computation(%arg0: tensor<8x49x196xf32>, %arg1: tensor<8x64x196xf32>) -> tensor<49x64x8xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,64,49]{1,2,0}"} : (tensor<8x64x196xf32>, tensor<8x49x196xf32>) -> tensor<8x64x49xf32>
    %1 = stablehlo.transpose %0, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[49,64,8]{1,2,0}"} : (tensor<8x64x49xf32>) -> tensor<49x64x8xf32>
    return %1 : tensor<49x64x8xf32>
  }
  func.func private @gemm_fusion_dot_general.86_computation(%arg0: tensor<49x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<512x256xf32>) -> tensor<512x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<512x256xf32>, tensor<49x256xf32>) -> tensor<512x49xf32>
    return %4 : tensor<512x49xf32>
  }
  func.func private @gemm_fusion_dot_general.88_computation(%arg0: tensor<49x256xf32>, %arg1: tensor<384x256xf32>) -> tensor<384x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<384x256xf32>, tensor<49x256xf32>) -> tensor<384x49xf32>
    return %0 : tensor<384x49xf32>
  }
  func.func private @gemm_fusion_dot_general.89_computation(%arg0: tensor<6x16x49xf32>, %arg1: tensor<6x16x49xf32>) -> tensor<6x49x49xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<6x49x49xf32>
    return %2 : tensor<6x49x49xf32>
  }
  func.func private @gemm_fusion_dot_general.91_computation(%arg0: tensor<49x192xf32>, %arg1: tensor<256x192xf32>) -> tensor<256x49xf32> {
    %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x192xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<49x192xf32>
    %2 = stablehlo.dot_general %arg1, %1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<256x192xf32>, tensor<49x192xf32>) -> tensor<256x49xf32>
    return %2 : tensor<256x49xf32>
  }
  func.func private @gemm_fusion_dot_general.92_computation(%arg0: tensor<49x256xf32>, %arg1: tensor<512x256xf32>) -> tensor<512x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<512x256xf32>, tensor<49x256xf32>) -> tensor<512x49xf32>
    return %0 : tensor<512x49xf32>
  }
  func.func private @gemm_fusion_dot_general.94_computation(%arg0: tensor<49x256xf32>, %arg1: tensor<384x256xf32>) -> tensor<384x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<384x256xf32>, tensor<49x256xf32>) -> tensor<384x49xf32>
    return %0 : tensor<384x49xf32>
  }
  func.func private @gemm_fusion_dot_general.95_computation(%arg0: tensor<6x16x49xf32>, %arg1: tensor<6x16x49xf32>) -> tensor<6x49x49xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<6x49x49xf32>
    return %2 : tensor<6x49x49xf32>
  }
  func.func private @gemm_fusion_dot_general.97_computation(%arg0: tensor<49x192xf32>, %arg1: tensor<256x192xf32>) -> tensor<256x49xf32> {
    %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x192xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<49x192xf32>
    %2 = stablehlo.dot_general %arg1, %1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<256x192xf32>, tensor<49x192xf32>) -> tensor<256x49xf32>
    return %2 : tensor<256x49xf32>
  }
  func.func private @gemm_fusion_dot_general.98_computation(%arg0: tensor<49x256xf32>, %arg1: tensor<512x256xf32>) -> tensor<512x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<512x256xf32>, tensor<49x256xf32>) -> tensor<512x49xf32>
    return %0 : tensor<512x49xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<49x256xf32>, %arg1: tensor<384x256xf32>) -> tensor<384x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<384x256xf32>, tensor<49x256xf32>) -> tensor<384x49xf32>
    return %0 : tensor<384x49xf32>
  }
  func.func private @gemm_fusion_dot_general.101_computation(%arg0: tensor<6x16x49xf32>, %arg1: tensor<6x16x49xf32>) -> tensor<6x49x49xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<6x49x49xf32>
    return %2 : tensor<6x49x49xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<49x192xf32>, %arg1: tensor<256x192xf32>) -> tensor<256x49xf32> {
    %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x192xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<49x192xf32>
    %2 = stablehlo.dot_general %arg1, %1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<256x192xf32>, tensor<49x192xf32>) -> tensor<256x49xf32>
    return %2 : tensor<256x49xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<49x256xf32>, %arg1: tensor<512x256xf32>) -> tensor<512x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<512x256xf32>, tensor<49x256xf32>) -> tensor<512x49xf32>
    return %0 : tensor<512x49xf32>
  }
  func.func private @gemm_fusion_dot_general.106_computation(%arg0: tensor<49x256xf32>, %arg1: tensor<1280x256xf32>) -> tensor<1280x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x256xf32>, tensor<49x256xf32>) -> tensor<1280x49xf32>
    return %0 : tensor<1280x49xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<16x16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>, %arg3: tensor<16x16x49xf32>) -> tensor<16x16x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<16x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<16x16x16xf32>
    %3 = stablehlo.add %1, %2 : tensor<16x16x16xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %5 = stablehlo.dot_general %4, %arg3, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16x16xf32>, tensor<16x16x49xf32>) -> tensor<16x16x49xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x16x49xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x16x49xf32>
    return %7 : tensor<16x16x49xf32>
  }
  func.func private @gemm_fusion_dot_general.109_computation(%arg0: tensor<16x16x49xf32>, %arg1: tensor<16x64x49xf32>) -> tensor<16x64x16xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,64,16]{1,2,0}"} : (tensor<16x64x49xf32>, tensor<16x16x49xf32>) -> tensor<16x64x16xf32>
    %1 = stablehlo.transpose %0, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,64,16]{1,2,0}"} : (tensor<16x64x16xf32>) -> tensor<16x64x16xf32>
    return %1 : tensor<16x64x16xf32>
  }
  func.func private @gemm_fusion_dot_general.114_computation(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x16x16xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<8x16x16xf32>
    return %2 : tensor<8x16x16xf32>
  }
  func.func private @gemm_fusion_dot_general.115_computation(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x32x16xf32>) -> tensor<16x8x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<8x16x32xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
    return %1 : tensor<16x8x32xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x16x16xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<8x16x16xf32>
    return %2 : tensor<8x16x16xf32>
  }
  func.func private @gemm_fusion_dot_general.121_computation(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x32x16xf32>) -> tensor<16x8x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<8x16x32xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
    return %1 : tensor<16x8x32xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x16x16xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<8x16x16xf32>
    return %2 : tensor<8x16x16xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x32x16xf32>) -> tensor<16x8x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<8x16x32xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
    return %1 : tensor<16x8x32xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x16x16xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<8x16x16xf32>
    return %2 : tensor<8x16x16xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x32x16xf32>) -> tensor<16x8x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<8x16x32xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
    return %1 : tensor<16x8x32xf32>
  }
  func.func private @region_1.8.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.22(%arg0: tensor<8x16x16xf32>, %arg1: tensor<256x8x1xf32>) -> tensor<8x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<8x16x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
    %4 = stablehlo.subtract %1, %3 : tensor<8x16x16xf32>
    %5 = stablehlo.exponential %4 : tensor<8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
    %8 = stablehlo.divide %5, %7 : tensor<8x16x16xf32>
    return %8 : tensor<8x16x16xf32>
  }
  func.func private @region_1.8.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.24(%arg0: tensor<8x16x16xf32>, %arg1: tensor<256x8x1xf32>) -> tensor<8x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<8x16x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
    %4 = stablehlo.subtract %1, %3 : tensor<8x16x16xf32>
    %5 = stablehlo.exponential %4 : tensor<8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
    %8 = stablehlo.divide %5, %7 : tensor<8x16x16xf32>
    return %8 : tensor<8x16x16xf32>
  }
  func.func private @region_1.8.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.26(%arg0: tensor<8x16x16xf32>, %arg1: tensor<256x8x1xf32>) -> tensor<8x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<8x16x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
    %4 = stablehlo.subtract %1, %3 : tensor<8x16x16xf32>
    %5 = stablehlo.exponential %4 : tensor<8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
    %8 = stablehlo.divide %5, %7 : tensor<8x16x16xf32>
    return %8 : tensor<8x16x16xf32>
  }
  func.func private @region_1.8.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.28(%arg0: tensor<8x16x16xf32>, %arg1: tensor<256x8x1xf32>) -> tensor<8x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<8x16x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
    %4 = stablehlo.subtract %1, %3 : tensor<8x16x16xf32>
    %5 = stablehlo.exponential %4 : tensor<8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
    %8 = stablehlo.divide %5, %7 : tensor<8x16x16xf32>
    return %8 : tensor<8x16x16xf32>
  }
  func.func private @region_1.8.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.30(%arg0: tensor<16x16x49xf32>) -> tensor<16x16x49xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<16x16x49xf32>, tensor<f32>) -> tensor<16x16xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<16x16xf32>) -> tensor<16x16x49xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<16x16x49xf32>
    %3 = stablehlo.exponential %2 : tensor<16x16x49xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x16x49xf32>, tensor<f32>) -> tensor<16x16xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<16x16xf32>) -> tensor<16x16x49xf32>
    %6 = stablehlo.divide %3, %5 : tensor<16x16x49xf32>
    return %6 : tensor<16x16x49xf32>
  }
  func.func private @region_1.8.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.32(%arg0: tensor<6x49x49xf32>) -> tensor<6x49x49xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<6x49x49xf32>
    %3 = stablehlo.exponential %2 : tensor<6x49x49xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
    %6 = stablehlo.divide %3, %5 : tensor<6x49x49xf32>
    return %6 : tensor<6x49x49xf32>
  }
  func.func private @region_1.8.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.34(%arg0: tensor<6x49x49xf32>) -> tensor<6x49x49xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<6x49x49xf32>
    %3 = stablehlo.exponential %2 : tensor<6x49x49xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
    %6 = stablehlo.divide %3, %5 : tensor<6x49x49xf32>
    return %6 : tensor<6x49x49xf32>
  }
  func.func private @region_1.8.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.36(%arg0: tensor<6x49x49xf32>) -> tensor<6x49x49xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<6x49x49xf32>
    %3 = stablehlo.exponential %2 : tensor<6x49x49xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
    %6 = stablehlo.divide %3, %5 : tensor<6x49x49xf32>
    return %6 : tensor<6x49x49xf32>
  }
  func.func private @region_1.8.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.38(%arg0: tensor<8x49x196xf32>) -> tensor<8x49x196xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x49x196xf32>, tensor<f32>) -> tensor<8x49xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<8x49xf32>) -> tensor<8x49x196xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<8x49x196xf32>
    %3 = stablehlo.exponential %2 : tensor<8x49x196xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x49x196xf32>, tensor<f32>) -> tensor<8x49xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<8x49xf32>) -> tensor<8x49x196xf32>
    %6 = stablehlo.divide %3, %5 : tensor<8x49x196xf32>
    return %6 : tensor<8x49x196xf32>
  }
  func.func private @region_1.8.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.40(%arg0: tensor<4x196x196xf32>) -> tensor<4x196x196xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<4x196x196xf32>, tensor<f32>) -> tensor<4x196xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<4x196xf32>) -> tensor<4x196x196xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<4x196x196xf32>
    %3 = stablehlo.exponential %2 : tensor<4x196x196xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<4x196x196xf32>, tensor<f32>) -> tensor<4x196xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<4x196xf32>) -> tensor<4x196x196xf32>
    %6 = stablehlo.divide %3, %5 : tensor<4x196x196xf32>
    return %6 : tensor<4x196x196xf32>
  }
  func.func private @region_1.8.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.42(%arg0: tensor<4x196x196xf32>) -> tensor<4x196x196xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<4x196x196xf32>, tensor<f32>) -> tensor<4x196xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<4x196xf32>) -> tensor<4x196x196xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<4x196x196xf32>
    %3 = stablehlo.exponential %2 : tensor<4x196x196xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<4x196x196xf32>, tensor<f32>) -> tensor<4x196xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<4x196xf32>) -> tensor<4x196x196xf32>
    %6 = stablehlo.divide %3, %5 : tensor<4x196x196xf32>
    return %6 : tensor<4x196x196xf32>
  }
  func.func private @fused_gather(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xi32>) -> tensor<256x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x16xi1>, tensor<16x16xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
    return %6 : tensor<256x8x1xf32>
  }
  func.func private @fused_gather.1(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xi32>) -> tensor<256x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x16xi1>, tensor<16x16xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
    return %6 : tensor<256x8x1xf32>
  }
  func.func private @fused_gather.2(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xi32>) -> tensor<256x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x16xi1>, tensor<16x16xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
    return %6 : tensor<256x8x1xf32>
  }
  func.func private @fused_gather.3(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xi32>) -> tensor<256x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x16xi1>, tensor<16x16xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
    return %6 : tensor<256x8x1xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x16xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x768xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<768xf32>
    %5 = stablehlo.rsqrt %4 : tensor<768xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x16x768xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x16x768xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<16x768xf32>
    return %21 : tensor<16x768xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<16x8x32xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x8x32xf32>) -> tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x16x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %5 = stablehlo.clamp %1, %3, %4 : tensor<1x16x256xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<1x16x256xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x256xf32>
    return %9 : tensor<16x256xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x16xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x768xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<768xf32>
    %5 = stablehlo.rsqrt %4 : tensor<768xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x16x768xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x16x768xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<16x768xf32>
    return %21 : tensor<16x768xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<16x8x32xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x8x32xf32>) -> tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x16x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %5 = stablehlo.clamp %1, %3, %4 : tensor<1x16x256xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<1x16x256xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x256xf32>
    return %9 : tensor<16x256xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x16xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x768xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<768xf32>
    %5 = stablehlo.rsqrt %4 : tensor<768xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x16x768xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x16x768xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<16x768xf32>
    return %21 : tensor<16x768xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<16x8x32xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x8x32xf32>) -> tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x16x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %5 = stablehlo.clamp %1, %3, %4 : tensor<1x16x256xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<1x16x256xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x256xf32>
    return %9 : tensor<16x256xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x16xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x768xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<768xf32>
    %5 = stablehlo.rsqrt %4 : tensor<768xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x16x768xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x16x768xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<16x768xf32>
    return %21 : tensor<16x768xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<16x8x32xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x8x32xf32>) -> tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x16x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %5 = stablehlo.clamp %1, %3, %4 : tensor<1x16x256xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<1x16x256xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x256xf32>
    return %9 : tensor<16x256xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x16xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x768xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<768xf32>
    %5 = stablehlo.rsqrt %4 : tensor<768xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x16x768xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x16x768xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<16x768xf32>
    return %21 : tensor<16x768xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<16x64x16xf32>) -> tensor<16x1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<16x64x16xf32>) -> tensor<1x16x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x16x1024xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
    %5 = stablehlo.clamp %1, %3, %4 : tensor<1x16x1024xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<1x16x1024xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x1024xf32>) -> tensor<16x1024xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x1024xf32>
    return %9 : tensor<16x1024xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x49xf32>, %arg4: tensor<512xf32>) -> tensor<49x512xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x49xf32>) -> tensor<49x512xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x512xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x512xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x49x512xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x49x512xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<49x512xf32>
    return %21 : tensor<49x512xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<6x49x32xf32>) -> tensor<1x49x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<6x49x32xf32>) -> tensor<49x6x32xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<49x6x32xf32>) -> tensor<1x49x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %4 = stablehlo.add %1, %3 : tensor<1x49x192xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %6 = stablehlo.clamp %2, %4, %5 : tensor<1x49x192xf32>
    %7 = stablehlo.multiply %1, %6 : tensor<1x49x192xf32>
    return %7 : tensor<1x49x192xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x49xf32>, %arg4: tensor<512xf32>) -> tensor<49x512xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x49xf32>) -> tensor<49x512xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x512xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x512xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x49x512xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x49x512xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<49x512xf32>
    return %21 : tensor<49x512xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<6x49x32xf32>) -> tensor<1x49x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<6x49x32xf32>) -> tensor<49x6x32xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<49x6x32xf32>) -> tensor<1x49x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %4 = stablehlo.add %1, %3 : tensor<1x49x192xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %6 = stablehlo.clamp %2, %4, %5 : tensor<1x49x192xf32>
    %7 = stablehlo.multiply %1, %6 : tensor<1x49x192xf32>
    return %7 : tensor<1x49x192xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x49xf32>, %arg4: tensor<512xf32>) -> tensor<49x512xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x49xf32>) -> tensor<49x512xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x512xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x512xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x49x512xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x49x512xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<49x512xf32>
    return %21 : tensor<49x512xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<6x49x32xf32>) -> tensor<1x49x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<6x49x32xf32>) -> tensor<49x6x32xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<49x6x32xf32>) -> tensor<1x49x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %4 = stablehlo.add %1, %3 : tensor<1x49x192xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %6 = stablehlo.clamp %2, %4, %5 : tensor<1x49x192xf32>
    %7 = stablehlo.multiply %1, %6 : tensor<1x49x192xf32>
    return %7 : tensor<1x49x192xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x49xf32>, %arg4: tensor<512xf32>) -> tensor<49x512xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x49xf32>) -> tensor<49x512xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x512xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x512xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %15 = stablehlo.add %12, %14 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %17 = stablehlo.clamp %13, %15, %16 : tensor<1x49x512xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<1x49x512xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<49x512xf32>
    return %21 : tensor<49x512xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<49x64x8xf32>) -> tensor<49x512xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<49x64x8xf32>) -> tensor<1x49x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %5 = stablehlo.clamp %1, %3, %4 : tensor<1x49x512xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<1x49x512xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x512xf32>
    return %9 : tensor<49x512xf32>
  }
  func.func private @fused_add(%arg0: tensor<8x196x49xf32>, %arg1: tensor<8x196xf32>, %arg2: tensor<49x196xi32>) -> tensor<8x49x196xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<8x196x49xf32>) -> tensor<8x49x196xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
    %2 = stablehlo.compare LT, %arg2, %1 : (tensor<49x196xi32>, tensor<49x196xi32>) -> tensor<49x196xi1>
    %c_0 = stablehlo.constant dense<196> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
    %4 = stablehlo.add %arg2, %3 : tensor<49x196xi32>
    %5 = stablehlo.select %2, %4, %arg2 : tensor<49x196xi1>, tensor<49x196xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[9604,1]{0,1}"} : (tensor<49x196xi32>) -> tensor<9604x1xi32>
    %7 = "stablehlo.gather"(%arg1, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[9604,8,1]{0,1,2}"} : (tensor<8x196xf32>, tensor<9604x1xi32>) -> tensor<9604x8x1xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<9604x8x1xf32>) -> tensor<8x49x196xf32>
    %9 = stablehlo.add %0, %8 : tensor<8x49x196xf32>
    return %9 : tensor<8x49x196xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<196x4x32xf32>) -> tensor<1x196x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<196x4x32xf32>) -> tensor<1x196x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x196x128xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %5 = stablehlo.clamp %1, %3, %4 : tensor<1x196x128xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<1x196x128xf32>
    return %6 : tensor<1x196x128xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<196x4x32xf32>) -> tensor<1x196x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<196x4x32xf32>) -> tensor<1x196x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x196x128xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %5 = stablehlo.clamp %1, %3, %4 : tensor<1x196x128xf32>
    %6 = stablehlo.multiply %0, %5 : tensor<1x196x128xf32>
    return %6 : tensor<1x196x128xf32>
  }
  func.func private @fused_slice(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16xf32>, %arg4: tensor<512xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<512x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<512x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
    %12 = stablehlo.slice %11 [0:1, 0:8, 16:32, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
    %13 = stablehlo.slice %11 [0:1, 0:8, 0:16, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
    return %12, %13 : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
  }
  func.func private @fused_slice.1(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16xf32>, %arg4: tensor<512xf32>) -> tensor<1x8x32x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<512x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<512x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
    %12 = stablehlo.slice %11 [0:1, 0:8, 32:64, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x32x16xf32>
    return %12 : tensor<1x8x32x16xf32>
  }
  func.func private @fused_slice.2(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16xf32>, %arg4: tensor<512xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<512x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<512x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
    %12 = stablehlo.slice %11 [0:1, 0:8, 16:32, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
    %13 = stablehlo.slice %11 [0:1, 0:8, 0:16, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
    return %12, %13 : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
  }
  func.func private @fused_slice.3(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16xf32>, %arg4: tensor<512xf32>) -> tensor<1x8x32x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<512x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<512x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
    %12 = stablehlo.slice %11 [0:1, 0:8, 32:64, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x32x16xf32>
    return %12 : tensor<1x8x32x16xf32>
  }
  func.func private @fused_slice.4(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16xf32>, %arg4: tensor<512xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<512x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<512x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
    %12 = stablehlo.slice %11 [0:1, 0:8, 16:32, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
    %13 = stablehlo.slice %11 [0:1, 0:8, 0:16, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
    return %12, %13 : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
  }
  func.func private @fused_slice.5(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16xf32>, %arg4: tensor<512xf32>) -> tensor<1x8x32x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<512x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<512x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
    %12 = stablehlo.slice %11 [0:1, 0:8, 32:64, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x32x16xf32>
    return %12 : tensor<1x8x32x16xf32>
  }
  func.func private @fused_slice.6(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16xf32>, %arg4: tensor<512xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<512x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<512x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
    %12 = stablehlo.slice %11 [0:1, 0:8, 16:32, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
    %13 = stablehlo.slice %11 [0:1, 0:8, 0:16, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
    return %12, %13 : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
  }
  func.func private @fused_slice.7(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16xf32>, %arg4: tensor<512xf32>) -> tensor<1x8x32x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<512x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<512x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
    %12 = stablehlo.slice %11 [0:1, 0:8, 32:64, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x32x16xf32>
    return %12 : tensor<1x8x32x16xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<16x16x49xf32>, %arg1: tensor<16x49xf32>, %arg2: tensor<16x49xi32>) -> tensor<16x16x49xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
    %1 = stablehlo.compare LT, %arg2, %0 : (tensor<16x49xi32>, tensor<16x49xi32>) -> tensor<16x49xi1>
    %c_0 = stablehlo.constant dense<49> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
    %3 = stablehlo.add %arg2, %2 : tensor<16x49xi32>
    %4 = stablehlo.select %1, %3, %arg2 : tensor<16x49xi1>, tensor<16x49xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[784,1]{0,1}"} : (tensor<16x49xi32>) -> tensor<784x1xi32>
    %6 = "stablehlo.gather"(%arg1, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[784,16,1]{0,1,2}"} : (tensor<16x49xf32>, tensor<784x1xi32>) -> tensor<784x16x1xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<784x16x1xf32>) -> tensor<16x16x49xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x16x49xf32>
    return %8 : tensor<16x16x49xf32>
  }
  func.func private @fused_slice.8(%arg0: tensor<1280x49xf32>, %arg1: tensor<1280xf32>, %arg2: tensor<1280xf32>, %arg3: tensor<1280xf32>, %arg4: tensor<1280xf32>) -> tensor<1x16x16x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1280x49xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<1280xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1280xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1280x49xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1280x49xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
    %10 = stablehlo.add %8, %9 : tensor<1280x49xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1280x49xf32>) -> tensor<1x16x80x49xf32>
    %12 = stablehlo.slice %11 [0:1, 0:16, 0:16, 0:49] : (tensor<1x16x80x49xf32>) -> tensor<1x16x16x49xf32>
    return %12 : tensor<1x16x16x49xf32>
  }
  func.func private @fused_slice.9(%arg0: tensor<1280x49xf32>, %arg1: tensor<1280xf32>, %arg2: tensor<1280xf32>, %arg3: tensor<1280xf32>, %arg4: tensor<1280xf32>) -> tensor<1x16x64x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1280x49xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<1280xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1280xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1280x49xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1280x49xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
    %10 = stablehlo.add %8, %9 : tensor<1280x49xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1280x49xf32>) -> tensor<1x16x80x49xf32>
    %12 = stablehlo.slice %11 [0:1, 0:16, 16:80, 0:49] : (tensor<1x16x80x49xf32>) -> tensor<1x16x64x49xf32>
    return %12 : tensor<1x16x64x49xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<6x49x49xf32>, %arg1: tensor<6x49xf32>, %arg2: tensor<49x49xi32>) -> tensor<6x49x49xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %1 = stablehlo.compare LT, %arg2, %0 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
    %c_0 = stablehlo.constant dense<49> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %3 = stablehlo.add %arg2, %2 : tensor<49x49xi32>
    %4 = stablehlo.select %1, %3, %arg2 : tensor<49x49xi1>, tensor<49x49xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
    %6 = "stablehlo.gather"(%arg1, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[2401,6,1]{0,1,2}"} : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<6x49x49xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<6x49x49xf32>
    return %8 : tensor<6x49x49xf32>
  }
  func.func private @fused_slice.10(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384x49xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> (tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<384x49xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<384x49xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<384x49xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %10 = stablehlo.add %8, %9 : tensor<384x49xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
    %12 = stablehlo.slice %11 [0:1, 0:6, 16:32, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
    %13 = stablehlo.slice %11 [0:1, 0:6, 0:16, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
    return %12, %13 : tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>
  }
  func.func private @fused_slice.11(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384x49xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<1x6x32x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<384x49xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<384x49xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<384x49xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %10 = stablehlo.add %8, %9 : tensor<384x49xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
    %12 = stablehlo.slice %11 [0:1, 0:6, 32:64, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x32x49xf32>
    return %12 : tensor<1x6x32x49xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<6x49x49xf32>, %arg1: tensor<6x49xf32>, %arg2: tensor<49x49xi32>) -> tensor<6x49x49xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %1 = stablehlo.compare LT, %arg2, %0 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
    %c_0 = stablehlo.constant dense<49> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %3 = stablehlo.add %arg2, %2 : tensor<49x49xi32>
    %4 = stablehlo.select %1, %3, %arg2 : tensor<49x49xi1>, tensor<49x49xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
    %6 = "stablehlo.gather"(%arg1, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[2401,6,1]{0,1,2}"} : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<6x49x49xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<6x49x49xf32>
    return %8 : tensor<6x49x49xf32>
  }
  func.func private @fused_slice.12(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384x49xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> (tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<384x49xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<384x49xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<384x49xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %10 = stablehlo.add %8, %9 : tensor<384x49xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
    %12 = stablehlo.slice %11 [0:1, 0:6, 16:32, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
    %13 = stablehlo.slice %11 [0:1, 0:6, 0:16, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
    return %12, %13 : tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>
  }
  func.func private @fused_slice.13(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384x49xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<1x6x32x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<384x49xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<384x49xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<384x49xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %10 = stablehlo.add %8, %9 : tensor<384x49xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
    %12 = stablehlo.slice %11 [0:1, 0:6, 32:64, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x32x49xf32>
    return %12 : tensor<1x6x32x49xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<6x49x49xf32>, %arg1: tensor<6x49xf32>, %arg2: tensor<49x49xi32>) -> tensor<6x49x49xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %1 = stablehlo.compare LT, %arg2, %0 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
    %c_0 = stablehlo.constant dense<49> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %3 = stablehlo.add %arg2, %2 : tensor<49x49xi32>
    %4 = stablehlo.select %1, %3, %arg2 : tensor<49x49xi1>, tensor<49x49xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
    %6 = "stablehlo.gather"(%arg1, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[2401,6,1]{0,1,2}"} : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<6x49x49xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<6x49x49xf32>
    return %8 : tensor<6x49x49xf32>
  }
  func.func private @fused_slice.14(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384x49xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> (tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<384x49xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<384x49xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<384x49xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %10 = stablehlo.add %8, %9 : tensor<384x49xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
    %12 = stablehlo.slice %11 [0:1, 0:6, 16:32, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
    %13 = stablehlo.slice %11 [0:1, 0:6, 0:16, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
    return %12, %13 : tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>
  }
  func.func private @fused_slice.15(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384x49xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<1x6x32x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<384x49xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<384x49xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<384x49xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
    %10 = stablehlo.add %8, %9 : tensor<384x49xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
    %12 = stablehlo.slice %11 [0:1, 0:6, 32:64, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x32x49xf32>
    return %12 : tensor<1x6x32x49xf32>
  }
  func.func private @fused_slice.16(%arg0: tensor<196x640xf32>, %arg1: tensor<640xf32>, %arg2: tensor<640xf32>, %arg3: tensor<640xf32>, %arg4: tensor<640xf32>) -> tensor<1x8x16x196xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<640x196xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<640x196xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<640xf32>
    %5 = stablehlo.rsqrt %4 : tensor<640xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<640x196xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<640x196xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
    %11 = stablehlo.add %9, %10 : tensor<640x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x196xf32>) -> tensor<1x8x80x196xf32>
    %13 = stablehlo.slice %12 [0:1, 0:8, 0:16, 0:196] : (tensor<1x8x80x196xf32>) -> tensor<1x8x16x196xf32>
    return %13 : tensor<1x8x16x196xf32>
  }
  func.func private @fused_slice.17(%arg0: tensor<196x640xf32>, %arg1: tensor<640xf32>, %arg2: tensor<640xf32>, %arg3: tensor<640xf32>, %arg4: tensor<640xf32>) -> tensor<1x8x64x196xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<640x196xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<640x196xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<640xf32>
    %5 = stablehlo.rsqrt %4 : tensor<640xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<640x196xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<640x196xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
    %11 = stablehlo.add %9, %10 : tensor<640x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x196xf32>) -> tensor<1x8x80x196xf32>
    %13 = stablehlo.slice %12 [0:1, 0:8, 16:80, 0:196] : (tensor<1x8x80x196xf32>) -> tensor<1x8x64x196xf32>
    return %13 : tensor<1x8x64x196xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<196x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<196x128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<196x128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<196x128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<1x128x14x14xf32>, %arg24: tensor<128xf32>) -> tensor<1x128x196xf32> {
    %0 = stablehlo.broadcast_in_dim %arg24, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %1 = stablehlo.subtract %arg23, %0 : tensor<1x128x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg22, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg21, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg20, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x14x14xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
    %12 = mhlo.bitcast %arg19 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %13 = stablehlo.broadcast_in_dim %arg18, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %14 = stablehlo.subtract %12, %13 : tensor<128x196xf32>
    %15 = stablehlo.add %arg17, %2 : tensor<128xf32>
    %16 = stablehlo.rsqrt %15 : tensor<128xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %18 = stablehlo.multiply %14, %17 : tensor<128x196xf32>
    %19 = stablehlo.broadcast_in_dim %arg16, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x196xf32>
    %21 = stablehlo.broadcast_in_dim %arg15, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x196xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %24 = stablehlo.add %11, %23 : tensor<1x128x196xf32>
    %25 = mhlo.bitcast %arg14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %26 = stablehlo.broadcast_in_dim %arg13, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %27 = stablehlo.subtract %25, %26 : tensor<128x196xf32>
    %28 = stablehlo.add %arg12, %2 : tensor<128xf32>
    %29 = stablehlo.rsqrt %28 : tensor<128xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %31 = stablehlo.multiply %27, %30 : tensor<128x196xf32>
    %32 = stablehlo.broadcast_in_dim %arg11, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128x196xf32>
    %34 = stablehlo.broadcast_in_dim %arg10, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %35 = stablehlo.add %33, %34 : tensor<128x196xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %37 = stablehlo.add %24, %36 : tensor<1x128x196xf32>
    %38 = mhlo.bitcast %arg7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %39 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %40 = stablehlo.subtract %38, %39 : tensor<128x196xf32>
    %41 = stablehlo.add %arg9, %2 : tensor<128xf32>
    %42 = stablehlo.rsqrt %41 : tensor<128xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %44 = stablehlo.multiply %40, %43 : tensor<128x196xf32>
    %45 = stablehlo.broadcast_in_dim %arg5, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x196xf32>
    %47 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x196xf32>
    %49 = mhlo.bitcast %48 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %50 = stablehlo.add %37, %49 : tensor<1x128x196xf32>
    %51 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %52 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %53 = stablehlo.subtract %51, %52 : tensor<128x196xf32>
    %54 = stablehlo.add %arg8, %2 : tensor<128xf32>
    %55 = stablehlo.rsqrt %54 : tensor<128xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %57 = stablehlo.multiply %53, %56 : tensor<128x196xf32>
    %58 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %59 = stablehlo.multiply %57, %58 : tensor<128x196xf32>
    %60 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %61 = stablehlo.add %59, %60 : tensor<128x196xf32>
    %62 = mhlo.bitcast %61 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %63 = stablehlo.add %50, %62 : tensor<1x128x196xf32>
    return %63 : tensor<1x128x196xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<196x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<196x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<196x128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<1x128x14x14xf32>, %arg19: tensor<128xf32>) -> tensor<1x128x196xf32> {
    %0 = stablehlo.broadcast_in_dim %arg19, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %1 = stablehlo.subtract %arg18, %0 : tensor<1x128x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg17, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x14x14xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
    %12 = mhlo.bitcast %arg14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %13 = stablehlo.broadcast_in_dim %arg13, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %14 = stablehlo.subtract %12, %13 : tensor<128x196xf32>
    %15 = stablehlo.add %arg12, %2 : tensor<128xf32>
    %16 = stablehlo.rsqrt %15 : tensor<128xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %18 = stablehlo.multiply %14, %17 : tensor<128x196xf32>
    %19 = stablehlo.broadcast_in_dim %arg11, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x196xf32>
    %21 = stablehlo.broadcast_in_dim %arg10, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x196xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %24 = stablehlo.add %11, %23 : tensor<1x128x196xf32>
    %25 = mhlo.bitcast %arg9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %26 = stablehlo.broadcast_in_dim %arg8, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %27 = stablehlo.subtract %25, %26 : tensor<128x196xf32>
    %28 = stablehlo.add %arg7, %2 : tensor<128xf32>
    %29 = stablehlo.rsqrt %28 : tensor<128xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %31 = stablehlo.multiply %27, %30 : tensor<128x196xf32>
    %32 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128x196xf32>
    %34 = stablehlo.broadcast_in_dim %arg5, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %35 = stablehlo.add %33, %34 : tensor<128x196xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %37 = stablehlo.add %24, %36 : tensor<1x128x196xf32>
    %38 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %39 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %40 = stablehlo.subtract %38, %39 : tensor<128x196xf32>
    %41 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %42 = stablehlo.rsqrt %41 : tensor<128xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %44 = stablehlo.multiply %40, %43 : tensor<128x196xf32>
    %45 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x196xf32>
    %47 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x196xf32>
    %49 = mhlo.bitcast %48 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %50 = stablehlo.add %37, %49 : tensor<1x128x196xf32>
    return %50 : tensor<1x128x196xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<4x196x196xf32>, %arg1: tensor<4x196xf32>, %arg2: tensor<196x196xi32>) -> tensor<4x196x196xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
    %1 = stablehlo.compare LT, %arg2, %0 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
    %c_0 = stablehlo.constant dense<196> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
    %3 = stablehlo.add %arg2, %2 : tensor<196x196xi32>
    %4 = stablehlo.select %1, %3, %arg2 : tensor<196x196xi1>, tensor<196x196xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
    %6 = "stablehlo.gather"(%arg1, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[38416,4,1]{0,1,2}"} : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<4x196x196xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<4x196x196xf32>
    return %8 : tensor<4x196x196xf32>
  }
  func.func private @fused_slice.18(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> (tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>) {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<256x196xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x196xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<256x196xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<256x196xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %11 = stablehlo.add %9, %10 : tensor<256x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x196xf32>) -> tensor<1x4x64x196xf32>
    %13 = stablehlo.slice %12 [0:1, 0:4, 0:16, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x16x196xf32>
    %14 = stablehlo.slice %12 [0:1, 0:4, 16:32, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x16x196xf32>
    return %13, %14 : tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>
  }
  func.func private @fused_slice.19(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<1x4x32x196xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<256x196xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x196xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<256x196xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<256x196xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %11 = stablehlo.add %9, %10 : tensor<256x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x196xf32>) -> tensor<1x4x64x196xf32>
    %13 = stablehlo.slice %12 [0:1, 0:4, 32:64, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x32x196xf32>
    return %13 : tensor<1x4x32x196xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<196x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<196x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<1x128x14x14xf32>, %arg14: tensor<128xf32>) -> tensor<1x128x196xf32> {
    %0 = stablehlo.broadcast_in_dim %arg14, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %1 = stablehlo.subtract %arg13, %0 : tensor<1x128x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg12, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg10, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x14x14xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
    %12 = mhlo.bitcast %arg9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %13 = stablehlo.broadcast_in_dim %arg8, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %14 = stablehlo.subtract %12, %13 : tensor<128x196xf32>
    %15 = stablehlo.add %arg7, %2 : tensor<128xf32>
    %16 = stablehlo.rsqrt %15 : tensor<128xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %18 = stablehlo.multiply %14, %17 : tensor<128x196xf32>
    %19 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x196xf32>
    %21 = stablehlo.broadcast_in_dim %arg5, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x196xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %24 = stablehlo.add %11, %23 : tensor<1x128x196xf32>
    %25 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %26 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %27 = stablehlo.subtract %25, %26 : tensor<128x196xf32>
    %28 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %29 = stablehlo.rsqrt %28 : tensor<128xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %31 = stablehlo.multiply %27, %30 : tensor<128x196xf32>
    %32 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128x196xf32>
    %34 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %35 = stablehlo.add %33, %34 : tensor<128x196xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %37 = stablehlo.add %24, %36 : tensor<1x128x196xf32>
    return %37 : tensor<1x128x196xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<196x128xf32>, %arg6: tensor<1x128x14x14xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128xf32>) -> tensor<1x128x196xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<1x128x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg9, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x14x14xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
    %12 = mhlo.bitcast %arg5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %13 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %14 = stablehlo.subtract %12, %13 : tensor<128x196xf32>
    %15 = stablehlo.add %arg8, %2 : tensor<128xf32>
    %16 = stablehlo.rsqrt %15 : tensor<128xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %18 = stablehlo.multiply %14, %17 : tensor<128x196xf32>
    %19 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x196xf32>
    %21 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x196xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %24 = stablehlo.add %11, %23 : tensor<1x128x196xf32>
    return %24 : tensor<1x128x196xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<4x196x196xf32>, %arg1: tensor<4x196xf32>, %arg2: tensor<196x196xi32>) -> tensor<4x196x196xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
    %1 = stablehlo.compare LT, %arg2, %0 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
    %c_0 = stablehlo.constant dense<196> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
    %3 = stablehlo.add %arg2, %2 : tensor<196x196xi32>
    %4 = stablehlo.select %1, %3, %arg2 : tensor<196x196xi1>, tensor<196x196xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
    %6 = "stablehlo.gather"(%arg1, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[38416,4,1]{0,1,2}"} : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<4x196x196xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<4x196x196xf32>
    return %8 : tensor<4x196x196xf32>
  }
  func.func private @fused_slice.20(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> (tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>) {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<256x196xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x196xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<256x196xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<256x196xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %11 = stablehlo.add %9, %10 : tensor<256x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x196xf32>) -> tensor<1x4x64x196xf32>
    %13 = stablehlo.slice %12 [0:1, 0:4, 0:16, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x16x196xf32>
    %14 = stablehlo.slice %12 [0:1, 0:4, 16:32, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x16x196xf32>
    return %13, %14 : tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>
  }
  func.func private @fused_slice.21(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<1x4x32x196xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<256x196xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x196xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<256x196xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<256x196xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
    %11 = stablehlo.add %9, %10 : tensor<256x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x196xf32>) -> tensor<1x4x64x196xf32>
    %13 = stablehlo.slice %12 [0:1, 0:4, 32:64, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x32x196xf32>
    return %13 : tensor<1x4x32x196xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<196x256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x196x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<196x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<196x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<196x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %10 = stablehlo.add %8, %9 : tensor<196x256xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x196x256xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %16 = stablehlo.clamp %12, %14, %15 : tensor<1x196x256xf32>
    %17 = stablehlo.multiply %11, %16 : tensor<1x196x256xf32>
    return %17 : tensor<1x196x256xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<196x256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x196x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<196x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<196x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<196x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %10 = stablehlo.add %8, %9 : tensor<196x256xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x196x256xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %16 = stablehlo.clamp %12, %14, %15 : tensor<1x196x256xf32>
    %17 = stablehlo.multiply %11, %16 : tensor<1x196x256xf32>
    return %17 : tensor<1x196x256xf32>
  }
  func.func private @region_1.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<1000x384xf32>, %arg1: tensor<384xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1000x384xf32>
    %1 = stablehlo.multiply %0, %arg0 : tensor<1000x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1000x384xf32>, tensor<f32>) -> tensor<1000xf32>
    return %2 : tensor<1000xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<1x16x112x112xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<1x16x112x112xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<16xf32>) -> tensor<1x16x112x112xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x16x112x112xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<16xf32>
    %4 = stablehlo.rsqrt %3 : tensor<16xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<16xf32>) -> tensor<1x16x112x112xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x16x112x112xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<16xf32>) -> tensor<1x16x112x112xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x16x112x112xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<16xf32>) -> tensor<1x16x112x112xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x16x112x112xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
    %13 = stablehlo.add %10, %12 : tensor<1x16x112x112xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
    %15 = stablehlo.clamp %11, %13, %14 : tensor<1x16x112x112xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x16x112x112xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x16x112x112xf32>
    return %18 : tensor<1x16x112x112xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<1x32x56x56xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>) -> tensor<1x32x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<32xf32>) -> tensor<1x32x56x56xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x32x56x56xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<32xf32>
    %4 = stablehlo.rsqrt %3 : tensor<32xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<32xf32>) -> tensor<1x32x56x56xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x32x56x56xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1x32x56x56xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x32x56x56xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1x32x56x56xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x32x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
    %13 = stablehlo.add %10, %12 : tensor<1x32x56x56xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
    %15 = stablehlo.clamp %11, %13, %14 : tensor<1x32x56x56xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x32x56x56xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x32x56x56xf32>
    return %18 : tensor<1x32x56x56xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x28x28xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x28x28xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x28x28xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x28x28xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x28x28xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x28x28xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x28x28xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x28x28xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
    %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
    %13 = stablehlo.add %10, %12 : tensor<1x64x28x28xf32>
    %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
    %15 = stablehlo.clamp %11, %13, %14 : tensor<1x64x28x28xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x64x28x28xf32>
    %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x64x28x28xf32>
    return %18 : tensor<1x64x28x28xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<1x128x14x14xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1x128x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x14x14xf32>
    return %6 : tensor<1x128x14x14xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<256x49xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg2, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x256xf32>
    return %7 : tensor<49x256xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<49x256xf32>, %arg1: tensor<256x49xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x256xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<49x256xf32>
    return %12 : tensor<49x256xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<49x256xf32>, %arg1: tensor<256x49xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x256xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<49x256xf32>
    return %12 : tensor<49x256xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<49x256xf32>, %arg1: tensor<256x49xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x256xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<49x256xf32>
    return %12 : tensor<49x256xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<49x256xf32>, %arg1: tensor<256x49xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x256xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<49x256xf32>
    return %12 : tensor<49x256xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<49x256xf32>, %arg1: tensor<256x49xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x256xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<49x256xf32>
    return %12 : tensor<49x256xf32>
  }
  func.func private @fused_add.16(%arg0: tensor<49x256xf32>, %arg1: tensor<256x49xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<49x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %11 = stablehlo.add %9, %10 : tensor<49x256xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<49x256xf32>
    return %12 : tensor<49x256xf32>
  }
  func.func private @fused_add.17(%arg0: tensor<49x256xf32>, %arg1: tensor<256x49xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<49x256xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<49x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %8 = stablehlo.add %arg7, %7 : tensor<256xf32>
    %9 = stablehlo.rsqrt %8 : tensor<256xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %11 = stablehlo.multiply %6, %10 : tensor<49x256xf32>
    %12 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<49x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %15 = stablehlo.add %13, %14 : tensor<49x256xf32>
    %16 = stablehlo.add %3, %15 : tensor<49x256xf32>
    return %16 : tensor<49x256xf32>
  }
  func.func private @fused_slice.22(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<196x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<196x128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<196x128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<196x128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<1x128x14x14xf32>, %arg24: tensor<128xf32>) -> tensor<1x128x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg24, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %1 = stablehlo.subtract %arg23, %0 : tensor<1x128x14x14xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg22, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x14x14xf32>
    %7 = stablehlo.broadcast_in_dim %arg21, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg20, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x14x14xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
    %12 = mhlo.bitcast %arg19 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %13 = stablehlo.broadcast_in_dim %arg18, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %14 = stablehlo.subtract %12, %13 : tensor<128x196xf32>
    %15 = stablehlo.add %arg17, %2 : tensor<128xf32>
    %16 = stablehlo.rsqrt %15 : tensor<128xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %18 = stablehlo.multiply %14, %17 : tensor<128x196xf32>
    %19 = stablehlo.broadcast_in_dim %arg16, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x196xf32>
    %21 = stablehlo.broadcast_in_dim %arg15, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x196xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %24 = stablehlo.add %11, %23 : tensor<1x128x196xf32>
    %25 = mhlo.bitcast %arg14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %26 = stablehlo.broadcast_in_dim %arg13, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %27 = stablehlo.subtract %25, %26 : tensor<128x196xf32>
    %28 = stablehlo.add %arg12, %2 : tensor<128xf32>
    %29 = stablehlo.rsqrt %28 : tensor<128xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %31 = stablehlo.multiply %27, %30 : tensor<128x196xf32>
    %32 = stablehlo.broadcast_in_dim %arg11, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128x196xf32>
    %34 = stablehlo.broadcast_in_dim %arg10, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %35 = stablehlo.add %33, %34 : tensor<128x196xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %37 = stablehlo.add %24, %36 : tensor<1x128x196xf32>
    %38 = mhlo.bitcast %arg7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %39 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %40 = stablehlo.subtract %38, %39 : tensor<128x196xf32>
    %41 = stablehlo.add %arg9, %2 : tensor<128xf32>
    %42 = stablehlo.rsqrt %41 : tensor<128xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %44 = stablehlo.multiply %40, %43 : tensor<128x196xf32>
    %45 = stablehlo.broadcast_in_dim %arg5, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x196xf32>
    %47 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x196xf32>
    %49 = mhlo.bitcast %48 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %50 = stablehlo.add %37, %49 : tensor<1x128x196xf32>
    %51 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
    %52 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %53 = stablehlo.subtract %51, %52 : tensor<128x196xf32>
    %54 = stablehlo.add %arg8, %2 : tensor<128xf32>
    %55 = stablehlo.rsqrt %54 : tensor<128xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %57 = stablehlo.multiply %53, %56 : tensor<128x196xf32>
    %58 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %59 = stablehlo.multiply %57, %58 : tensor<128x196xf32>
    %60 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
    %61 = stablehlo.add %59, %60 : tensor<128x196xf32>
    %62 = mhlo.bitcast %61 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
    %63 = stablehlo.add %50, %62 : tensor<1x128x196xf32>
    %64 = mhlo.bitcast %63 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x196xf32>) -> tensor<1x128x14x14xf32>
    %65 = stablehlo.slice %64 [0:1, 0:128, 0:14:2, 0:14] : (tensor<1x128x14x14xf32>) -> tensor<1x128x7x14xf32>
    %66 = stablehlo.slice %65 [0:1, 0:128, 0:7, 0:14:2] : (tensor<1x128x7x14xf32>) -> tensor<1x128x7x7xf32>
    return %66 : tensor<1x128x7x7xf32>
  }
  func.func private @fused_slice.23(%arg0: tensor<49x256xf32>) -> tensor<1x4x4x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x256xf32>) -> tensor<1x7x7x256xf32>
    %1 = stablehlo.slice %0 [0:1, 0:7:2, 0:7, 0:256] : (tensor<1x7x7x256xf32>) -> tensor<1x4x7x256xf32>
    %2 = stablehlo.slice %1 [0:1, 0:4, 0:7:2, 0:256] : (tensor<1x4x7x256xf32>) -> tensor<1x4x4x256xf32>
    return %2 : tensor<1x4x4x256xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<128xf32>, %arg1: tensor<128x49xf32>, %arg2: tensor<128xf32>) -> tensor<49x128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<128x49xf32>) -> tensor<49x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<49x128xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4 = stablehlo.add %arg2, %3 : tensor<128xf32>
    %5 = stablehlo.rsqrt %4 : tensor<128xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<49x128xf32>
    return %7 : tensor<49x128xf32>
  }
  func.func private @fused_add.18(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384x16xf32>, %arg4: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg4, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    return %11 : tensor<16x384xf32>
  }
  func.func private @fused_add.19(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384x16xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<16x384xf32>
    return %12 : tensor<16x384xf32>
  }
  func.func private @fused_add.20(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384x16xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<16x384xf32>
    return %12 : tensor<16x384xf32>
  }
  func.func private @fused_add.21(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384x16xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<16x384xf32>
    return %12 : tensor<16x384xf32>
  }
  func.func private @fused_add.22(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384x16xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<16x384xf32>
    return %12 : tensor<16x384xf32>
  }
  func.func private @fused_add.23(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384x16xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<16x384xf32>
    return %12 : tensor<16x384xf32>
  }
  func.func private @fused_add.24(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384x16xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<16x384xf32>
    return %12 : tensor<16x384xf32>
  }
  func.func private @fused_add.25(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384x16xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<16x384xf32>
    return %12 : tensor<16x384xf32>
  }
  func.func private @fused_add.26(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384x16xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg0, %11 : tensor<16x384xf32>
    return %12 : tensor<16x384xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<256xf32>, %arg1: tensor<256x16xf32>, %arg2: tensor<256xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x16xf32>) -> tensor<16x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x256xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4 = stablehlo.add %arg2, %3 : tensor<256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x256xf32>
    return %7 : tensor<16x256xf32>
  }
  func.func private @region_1.8.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add.27(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>, %arg6: tensor<384xf32>, %arg7: tensor<384xf32>, %arg8: tensor<384xf32>, %arg9: tensor<384x16xf32>) -> tensor<384xf32> {
    %0 = mhlo.bitcast %arg9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg8, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %4 = stablehlo.add %arg7, %3 : tensor<384xf32>
    %5 = stablehlo.rsqrt %4 : tensor<384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.multiply %2, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.add %arg4, %11 : tensor<16x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<16x384xf32>, tensor<f32>) -> tensor<384xf32>
    %cst_1 = stablehlo.constant dense<6.250000e-02> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<384xf32>
    %16 = stablehlo.subtract %15, %arg3 : tensor<384xf32>
    %17 = stablehlo.add %arg2, %3 : tensor<384xf32>
    %18 = stablehlo.rsqrt %17 : tensor<384xf32>
    %19 = stablehlo.multiply %16, %18 : tensor<384xf32>
    %20 = stablehlo.multiply %19, %arg1 : tensor<384xf32>
    %21 = stablehlo.add %20, %arg0 : tensor<384xf32>
    return %21 : tensor<384xf32>
  }
  func.func private @wrapped_add_computation(%arg0: tensor<1000xf32>, %arg1: tensor<1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func.func @main(%arg0: tensor<16x3x3x3xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<32x16x3x3xf32>, %arg4: tensor<32xf32>, %arg5: tensor<32xf32>, %arg6: tensor<64x32x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<128x64x3x3xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<4x196xf32>, %arg13: tensor<256x128xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<128x128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<256x128xf32>, %arg20: tensor<256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<128x256xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<4x196xf32>, %arg26: tensor<256x128xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128xf32>, %arg32: tensor<256x128xf32>, %arg33: tensor<256xf32>, %arg34: tensor<256xf32>, %arg35: tensor<128x256xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128xf32>, %arg38: tensor<8x196xf32>, %arg39: tensor<640x128xf32>, %arg40: tensor<640xf32>, %arg41: tensor<640xf32>, %arg42: tensor<128x128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<256x512xf32>, %arg46: tensor<256xf32>, %arg47: tensor<256xf32>, %arg48: tensor<512x256xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<256x512xf32>, %arg52: tensor<256xf32>, %arg53: tensor<256xf32>, %arg54: tensor<6x49xf32>, %arg55: tensor<384x256xf32>, %arg56: tensor<384xf32>, %arg57: tensor<384xf32>, %arg58: tensor<256x192xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<512x256xf32>, %arg62: tensor<512xf32>, %arg63: tensor<512xf32>, %arg64: tensor<256x512xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<6x49xf32>, %arg68: tensor<384x256xf32>, %arg69: tensor<384xf32>, %arg70: tensor<384xf32>, %arg71: tensor<256x192xf32>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<512x256xf32>, %arg75: tensor<512xf32>, %arg76: tensor<512xf32>, %arg77: tensor<256x512xf32>, %arg78: tensor<256xf32>, %arg79: tensor<256xf32>, %arg80: tensor<6x49xf32>, %arg81: tensor<384x256xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<256x192xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<512x256xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<256x512xf32>, %arg91: tensor<256xf32>, %arg92: tensor<256xf32>, %arg93: tensor<16x49xf32>, %arg94: tensor<1280x256xf32>, %arg95: tensor<1280xf32>, %arg96: tensor<1280xf32>, %arg97: tensor<256x256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<256xf32>, %arg100: tensor<384x1024xf32>, %arg101: tensor<384xf32>, %arg102: tensor<384xf32>, %arg103: tensor<768x384xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768xf32>, %arg106: tensor<384x768xf32>, %arg107: tensor<384xf32>, %arg108: tensor<384xf32>, %arg109: tensor<8x16xf32>, %arg110: tensor<512x384xf32>, %arg111: tensor<512xf32>, %arg112: tensor<512xf32>, %arg113: tensor<384x256xf32>, %arg114: tensor<384xf32>, %arg115: tensor<384xf32>, %arg116: tensor<768x384xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<384x768xf32>, %arg120: tensor<384xf32>, %arg121: tensor<384xf32>, %arg122: tensor<8x16xf32>, %arg123: tensor<512x384xf32>, %arg124: tensor<512xf32>, %arg125: tensor<512xf32>, %arg126: tensor<384x256xf32>, %arg127: tensor<384xf32>, %arg128: tensor<384xf32>, %arg129: tensor<768x384xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<384x768xf32>, %arg133: tensor<384xf32>, %arg134: tensor<384xf32>, %arg135: tensor<8x16xf32>, %arg136: tensor<512x384xf32>, %arg137: tensor<512xf32>, %arg138: tensor<512xf32>, %arg139: tensor<384x256xf32>, %arg140: tensor<384xf32>, %arg141: tensor<384xf32>, %arg142: tensor<768x384xf32>, %arg143: tensor<768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<384x768xf32>, %arg146: tensor<384xf32>, %arg147: tensor<384xf32>, %arg148: tensor<8x16xf32>, %arg149: tensor<512x384xf32>, %arg150: tensor<512xf32>, %arg151: tensor<512xf32>, %arg152: tensor<384x256xf32>, %arg153: tensor<384xf32>, %arg154: tensor<384xf32>, %arg155: tensor<768x384xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<384x768xf32>, %arg159: tensor<384xf32>, %arg160: tensor<384xf32>, %arg161: tensor<384xf32>, %arg162: tensor<384xf32>, %arg163: tensor<1000x384xf32>, %arg164: tensor<1000xf32>, %arg165: tensor<16xf32>, %arg166: tensor<16xf32>, %arg167: tensor<32xf32>, %arg168: tensor<32xf32>, %arg169: tensor<64xf32>, %arg170: tensor<64xf32>, %arg171: tensor<128xf32>, %arg172: tensor<128xf32>, %arg173: tensor<196x196xi32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<128xf32>, %arg177: tensor<128xf32>, %arg178: tensor<256xf32>, %arg179: tensor<256xf32>, %arg180: tensor<128xf32>, %arg181: tensor<128xf32>, %arg182: tensor<196x196xi32>, %arg183: tensor<256xf32>, %arg184: tensor<256xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<128xf32>, %arg190: tensor<128xf32>, %arg191: tensor<49x196xi32>, %arg192: tensor<640xf32>, %arg193: tensor<640xf32>, %arg194: tensor<128xf32>, %arg195: tensor<128xf32>, %arg196: tensor<256xf32>, %arg197: tensor<256xf32>, %arg198: tensor<512xf32>, %arg199: tensor<512xf32>, %arg200: tensor<256xf32>, %arg201: tensor<256xf32>, %arg202: tensor<49x49xi32>, %arg203: tensor<384xf32>, %arg204: tensor<384xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<512xf32>, %arg208: tensor<512xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<49x49xi32>, %arg212: tensor<384xf32>, %arg213: tensor<384xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<512xf32>, %arg217: tensor<512xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<49x49xi32>, %arg221: tensor<384xf32>, %arg222: tensor<384xf32>, %arg223: tensor<256xf32>, %arg224: tensor<256xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<256xf32>, %arg228: tensor<256xf32>, %arg229: tensor<16x49xi32>, %arg230: tensor<1280xf32>, %arg231: tensor<1280xf32>, %arg232: tensor<256xf32>, %arg233: tensor<256xf32>, %arg234: tensor<384xf32>, %arg235: tensor<384xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<384xf32>, %arg239: tensor<384xf32>, %arg240: tensor<16x16xi32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<384xf32>, %arg244: tensor<384xf32>, %arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<384xf32>, %arg248: tensor<384xf32>, %arg249: tensor<16x16xi32>, %arg250: tensor<512xf32>, %arg251: tensor<512xf32>, %arg252: tensor<384xf32>, %arg253: tensor<384xf32>, %arg254: tensor<768xf32>, %arg255: tensor<768xf32>, %arg256: tensor<384xf32>, %arg257: tensor<384xf32>, %arg258: tensor<16x16xi32>, %arg259: tensor<512xf32>, %arg260: tensor<512xf32>, %arg261: tensor<384xf32>, %arg262: tensor<384xf32>, %arg263: tensor<768xf32>, %arg264: tensor<768xf32>, %arg265: tensor<384xf32>, %arg266: tensor<384xf32>, %arg267: tensor<16x16xi32>, %arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<384xf32>, %arg271: tensor<384xf32>, %arg272: tensor<768xf32>, %arg273: tensor<768xf32>, %arg274: tensor<384xf32>, %arg275: tensor<384xf32>, %arg276: tensor<384xf32>, %arg277: tensor<384xf32>, %arg278: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = stablehlo.custom_call @__cudnn$convForward(%arg278, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,16,112,112]{3,2,1,0}, u8[0]{0})"} : (tensor<1x3x224x224xf32>, tensor<16x3x3x3xf32>) -> tuple<tensor<1x16x112x112xf32>, tensor<0xui8>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<1x16x112x112xf32>, tensor<0xui8>>) -> tensor<1x16x112x112xf32>
    %2 = "mhlo.fusion"(%arg2, %arg1, %1, %arg165, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16xf32>, %arg280: tensor<16xf32>, %arg281: tensor<1x16x112x112xf32>, %arg282: tensor<16xf32>, %arg283: tensor<16xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<16xf32>) -> tensor<1x16x112x112xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<1x16x112x112xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<16xf32>
      %267 = stablehlo.rsqrt %266 : tensor<16xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<16xf32>) -> tensor<1x16x112x112xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x16x112x112xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<16xf32>) -> tensor<1x16x112x112xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1x16x112x112xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<16xf32>) -> tensor<1x16x112x112xf32>
      %273 = stablehlo.add %271, %272 : tensor<1x16x112x112xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %274 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %275 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %276 = stablehlo.add %273, %275 : tensor<1x16x112x112xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %278 = stablehlo.clamp %274, %276, %277 : tensor<1x16x112x112xf32>
      %279 = stablehlo.multiply %273, %278 : tensor<1x16x112x112xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %280 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %281 = stablehlo.multiply %279, %280 : tensor<1x16x112x112xf32>
      mhlo.return %281 : tensor<1x16x112x112xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16xf32>, tensor<1x16x112x112xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x16x112x112xf32>
    %3 = stablehlo.custom_call @__cudnn$convForward(%2, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,32,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x16x112x112xf32>, tensor<32x16x3x3xf32>) -> tuple<tensor<1x32x56x56xf32>, tensor<0xui8>>
    %4 = stablehlo.get_tuple_element %3[0] : (tuple<tensor<1x32x56x56xf32>, tensor<0xui8>>) -> tensor<1x32x56x56xf32>
    %5 = "mhlo.fusion"(%arg5, %arg4, %4, %arg167, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<32xf32>, %arg280: tensor<32xf32>, %arg281: tensor<1x32x56x56xf32>, %arg282: tensor<32xf32>, %arg283: tensor<32xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<32xf32>) -> tensor<1x32x56x56xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<1x32x56x56xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<32xf32>
      %267 = stablehlo.rsqrt %266 : tensor<32xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<32xf32>) -> tensor<1x32x56x56xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x32x56x56xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<32xf32>) -> tensor<1x32x56x56xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1x32x56x56xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<32xf32>) -> tensor<1x32x56x56xf32>
      %273 = stablehlo.add %271, %272 : tensor<1x32x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %274 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %275 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %276 = stablehlo.add %273, %275 : tensor<1x32x56x56xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %278 = stablehlo.clamp %274, %276, %277 : tensor<1x32x56x56xf32>
      %279 = stablehlo.multiply %273, %278 : tensor<1x32x56x56xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %280 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %281 = stablehlo.multiply %279, %280 : tensor<1x32x56x56xf32>
      mhlo.return %281 : tensor<1x32x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x32x56x56xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x32x56x56xf32>
    %6 = stablehlo.custom_call @__cudnn$convForward(%5, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x32x56x56xf32>, tensor<64x32x3x3xf32>) -> tuple<tensor<1x64x28x28xf32>, tensor<0xui8>>
    %7 = stablehlo.get_tuple_element %6[0] : (tuple<tensor<1x64x28x28xf32>, tensor<0xui8>>) -> tensor<1x64x28x28xf32>
    %8 = "mhlo.fusion"(%arg8, %arg7, %7, %arg169, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<64xf32>, %arg280: tensor<64xf32>, %arg281: tensor<1x64x28x28xf32>, %arg282: tensor<64xf32>, %arg283: tensor<64xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<64xf32>) -> tensor<1x64x28x28xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<1x64x28x28xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<64xf32>
      %267 = stablehlo.rsqrt %266 : tensor<64xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<64xf32>) -> tensor<1x64x28x28xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x64x28x28xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<64xf32>) -> tensor<1x64x28x28xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1x64x28x28xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<64xf32>) -> tensor<1x64x28x28xf32>
      %273 = stablehlo.add %271, %272 : tensor<1x64x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %274 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %275 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %276 = stablehlo.add %273, %275 : tensor<1x64x28x28xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %278 = stablehlo.clamp %274, %276, %277 : tensor<1x64x28x28xf32>
      %279 = stablehlo.multiply %273, %278 : tensor<1x64x28x28xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %280 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %281 = stablehlo.multiply %279, %280 : tensor<1x64x28x28xf32>
      mhlo.return %281 : tensor<1x64x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x28x28xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x28x28xf32>
    %9 = stablehlo.custom_call @__cudnn$convForward(%8, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x28x28xf32>, tensor<128x64x3x3xf32>) -> tuple<tensor<1x128x14x14xf32>, tensor<0xui8>>
    %10 = stablehlo.get_tuple_element %9[0] : (tuple<tensor<1x128x14x14xf32>, tensor<0xui8>>) -> tensor<1x128x14x14xf32>
    %11 = "mhlo.fusion"(%10, %arg171, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x128x14x14xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>):
      %263 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %264 = stablehlo.subtract %arg279, %263 : tensor<1x128x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<128xf32>
      %267 = stablehlo.rsqrt %266 : tensor<128xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x128x14x14xf32>
      mhlo.return %269 : tensor<1x128x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x14x14xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x14x14xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<1x128x14x14xf32>) -> tensor<196x128xf32>
    %13 = "mhlo.fusion"(%12, %arg10, %arg11, %arg13) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<256x128xf32>):
      %263 = stablehlo.broadcast_in_dim %arg280, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<128xf32>) -> tensor<196x128xf32>
      %264 = stablehlo.multiply %arg279, %263 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : tensor<196x128xf32>
      %265 = stablehlo.broadcast_in_dim %arg281, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<128xf32>) -> tensor<196x128xf32>
      %266 = stablehlo.add %264, %265 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : tensor<196x128xf32>
      %267 = stablehlo.dot_general %266, %arg282, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,256]{0,1}"} : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
      mhlo.return %267 : tensor<196x256xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,256]{0,1}"} : (tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %14:2 = "mhlo.fusion"(%arg15, %arg14, %arg174, %13, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<256x196xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<256x196xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<256x196xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<256x196xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %274 = stablehlo.add %272, %273 : tensor<256x196xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x196xf32>) -> tensor<1x4x64x196xf32>
      %276 = stablehlo.slice %275 [0:1, 0:4, 0:16, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x16x196xf32>
      %277 = stablehlo.slice %275 [0:1, 0:4, 16:32, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x16x196xf32>
      mhlo.return %276, %277 : tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> (tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>)
    %15 = stablehlo.tuple %14#0, %14#1 {xla_shape = "(f32[1,4,16,196]{3,2,1,0}, f32[1,4,16,196]{3,2,1,0})"} : tuple<tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>>
    %16 = mhlo.bitcast %14#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
    %17 = mhlo.bitcast %14#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
    %18 = "mhlo.fusion"(%16, %17) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<4x16x196xf32>, %arg280: tensor<4x16x196xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x196xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<4x196x196xf32>
      mhlo.return %265 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<4x16x196xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
    %19 = "mhlo.fusion"(%18, %arg12, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196x196xf32>, %arg280: tensor<4x196xf32>, %arg281: tensor<196x196xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %264 = stablehlo.compare LT, %arg281, %263 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
      %c_0 = stablehlo.constant dense<196> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %266 = stablehlo.add %arg281, %265 : tensor<196x196xi32>
      %267 = stablehlo.select %264, %266, %arg281 : tensor<196x196xi1>, tensor<196x196xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
      %269 = "stablehlo.gather"(%arg280, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[38416,4,1]{0,1,2}"} : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<4x196x196xf32>
      %271 = stablehlo.add %arg279, %270 : tensor<4x196x196xf32>
      mhlo.return %271 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x196xf32>, tensor<4x196xf32>, tensor<196x196xi32>) -> tensor<4x196x196xf32>
    %20 = "mhlo.fusion"(%19) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<4x196x196xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %263 = stablehlo.reduce(%arg279 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<4x196x196xf32>, tensor<f32>) -> tensor<4x196xf32>
      %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1] : (tensor<4x196xf32>) -> tensor<4x196x196xf32>
      %265 = stablehlo.subtract %arg279, %264 : tensor<4x196x196xf32>
      %266 = stablehlo.exponential %265 : tensor<4x196x196xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %267 = stablehlo.reduce(%266 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<4x196x196xf32>, tensor<f32>) -> tensor<4x196xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1] : (tensor<4x196xf32>) -> tensor<4x196x196xf32>
      %269 = stablehlo.divide %266, %268 : tensor<4x196x196xf32>
      mhlo.return %269 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x196xf32>) -> tensor<4x196x196xf32>
    %21 = "mhlo.fusion"(%arg15, %arg14, %arg174, %13, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<256x196xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<256x196xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<256x196xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<256x196xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %274 = stablehlo.add %272, %273 : tensor<256x196xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x196xf32>) -> tensor<1x4x64x196xf32>
      %276 = stablehlo.slice %275 [0:1, 0:4, 32:64, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x32x196xf32>
      mhlo.return %276 : tensor<1x4x32x196xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<1x4x32x196xf32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x32x196xf32>) -> tensor<4x32x196xf32>
    %23 = "mhlo.fusion"(%20, %22) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<4x196x196xf32>, %arg280: tensor<4x32x196xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x196x196xf32>, tensor<4x32x196xf32>) -> tensor<4x196x32xf32>
      %264 = stablehlo.transpose %263, dims = [1, 0, 2] : (tensor<4x196x32xf32>) -> tensor<196x4x32xf32>
      mhlo.return %264 : tensor<196x4x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x196xf32>, tensor<4x32x196xf32>) -> tensor<196x4x32xf32>
    %24 = "mhlo.fusion"(%23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<196x4x32xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<196x4x32xf32>) -> tensor<1x196x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %266 = stablehlo.add %263, %265 : tensor<1x196x128xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %268 = stablehlo.clamp %264, %266, %267 : tensor<1x196x128xf32>
      %269 = stablehlo.multiply %263, %268 : tensor<1x196x128xf32>
      mhlo.return %269 : tensor<1x196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<196x4x32xf32>) -> tensor<1x196x128xf32>
    %25 = mhlo.bitcast %24 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    %26 = "mhlo.fusion"(%25, %arg16) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x128xf32>, %arg280: tensor<128x128xf32>):
      %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x128xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<196x128xf32>
      %265 = stablehlo.dot_general %264, %arg280, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
      mhlo.return %265 : tensor<196x128xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
    %27 = "mhlo.fusion"(%arg18, %arg11, %arg10, %arg17, %arg176, %26, %10, %arg171, %arg177, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<128xf32>, %arg284: tensor<196x128xf32>, %arg285: tensor<1x128x14x14xf32>, %arg286: tensor<128xf32>, %arg287: tensor<128xf32>, %arg288: tensor<128xf32>):
      %263 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %264 = stablehlo.subtract %arg285, %263 : tensor<1x128x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %266 = stablehlo.add %arg288, %265 : tensor<128xf32>
      %267 = stablehlo.rsqrt %266 : tensor<128xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x128x14x14xf32>
      %270 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1x128x14x14xf32>
      %272 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %273 = stablehlo.add %271, %272 : tensor<1x128x14x14xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %275 = mhlo.bitcast %arg284 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %276 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %277 = stablehlo.subtract %275, %276 : tensor<128x196xf32>
      %278 = stablehlo.add %arg287, %265 : tensor<128xf32>
      %279 = stablehlo.rsqrt %278 : tensor<128xf32>
      %280 = stablehlo.broadcast_in_dim %279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %281 = stablehlo.multiply %277, %280 : tensor<128x196xf32>
      %282 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %283 = stablehlo.multiply %281, %282 : tensor<128x196xf32>
      %284 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %285 = stablehlo.add %283, %284 : tensor<128x196xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %287 = stablehlo.add %274, %286 : tensor<1x128x196xf32>
      mhlo.return %287 : tensor<1x128x196xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<1x128x14x14xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x196xf32>
    %28 = mhlo.bitcast %27 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<1x128x196xf32>) -> tensor<196x128xf32>
    %29 = "mhlo.fusion"(%28, %arg19) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x128xf32>, %arg280: tensor<256x128xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
      mhlo.return %263 : tensor<196x256xf32>
    }) {output_operand_aliasing = []} : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %30 = "mhlo.fusion"(%arg21, %arg20, %29, %arg178, %arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<196x256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<196x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<256xf32>
      %267 = stablehlo.rsqrt %266 : tensor<256xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<196x256xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<196x256xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %273 = stablehlo.add %271, %272 : tensor<196x256xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %275 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %277 = stablehlo.add %274, %276 : tensor<1x196x256xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %278 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %279 = stablehlo.clamp %275, %277, %278 : tensor<1x196x256xf32>
      %280 = stablehlo.multiply %274, %279 : tensor<1x196x256xf32>
      mhlo.return %280 : tensor<1x196x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x196x256xf32>
    %31 = mhlo.bitcast %30 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x256xf32>) -> tensor<196x256xf32>
    %32 = "mhlo.fusion"(%31, %arg22) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x256xf32>, %arg280: tensor<128x256xf32>):
      %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x256xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<196x256xf32>
      %265 = stablehlo.dot_general %264, %arg280, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
      mhlo.return %265 : tensor<196x128xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
    %33 = "mhlo.fusion"(%arg24, %arg23, %arg180, %32, %arg181, %arg18, %arg17, %arg177, %arg176, %26, %arg11, %arg10, %arg172, %10, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<196x128xf32>, %arg283: tensor<128xf32>, %arg284: tensor<128xf32>, %arg285: tensor<128xf32>, %arg286: tensor<128xf32>, %arg287: tensor<128xf32>, %arg288: tensor<196x128xf32>, %arg289: tensor<128xf32>, %arg290: tensor<128xf32>, %arg291: tensor<128xf32>, %arg292: tensor<1x128x14x14xf32>, %arg293: tensor<128xf32>):
      %263 = stablehlo.broadcast_in_dim %arg293, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %264 = stablehlo.subtract %arg292, %263 : tensor<1x128x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %266 = stablehlo.add %arg291, %265 : tensor<128xf32>
      %267 = stablehlo.rsqrt %266 : tensor<128xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x128x14x14xf32>
      %270 = stablehlo.broadcast_in_dim %arg290, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1x128x14x14xf32>
      %272 = stablehlo.broadcast_in_dim %arg289, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %273 = stablehlo.add %271, %272 : tensor<1x128x14x14xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %275 = mhlo.bitcast %arg288 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %276 = stablehlo.broadcast_in_dim %arg287, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %277 = stablehlo.subtract %275, %276 : tensor<128x196xf32>
      %278 = stablehlo.add %arg286, %265 : tensor<128xf32>
      %279 = stablehlo.rsqrt %278 : tensor<128xf32>
      %280 = stablehlo.broadcast_in_dim %279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %281 = stablehlo.multiply %277, %280 : tensor<128x196xf32>
      %282 = stablehlo.broadcast_in_dim %arg285, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %283 = stablehlo.multiply %281, %282 : tensor<128x196xf32>
      %284 = stablehlo.broadcast_in_dim %arg284, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %285 = stablehlo.add %283, %284 : tensor<128x196xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %287 = stablehlo.add %274, %286 : tensor<1x128x196xf32>
      %288 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %289 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %290 = stablehlo.subtract %288, %289 : tensor<128x196xf32>
      %291 = stablehlo.add %arg283, %265 : tensor<128xf32>
      %292 = stablehlo.rsqrt %291 : tensor<128xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %294 = stablehlo.multiply %290, %293 : tensor<128x196xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<128x196xf32>
      %297 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %298 = stablehlo.add %296, %297 : tensor<128x196xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %300 = stablehlo.add %287, %299 : tensor<1x128x196xf32>
      mhlo.return %300 : tensor<1x128x196xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x14x14xf32>, tensor<128xf32>) -> tensor<1x128x196xf32>
    %34 = mhlo.bitcast %33 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<1x128x196xf32>) -> tensor<196x128xf32>
    %35 = "mhlo.fusion"(%34, %arg26) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x128xf32>, %arg280: tensor<256x128xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,256]{0,1}"} : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
      mhlo.return %263 : tensor<196x256xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,256]{0,1}"} : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %36:2 = "mhlo.fusion"(%arg28, %arg27, %arg183, %35, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<256x196xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<256x196xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<256x196xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<256x196xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %274 = stablehlo.add %272, %273 : tensor<256x196xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x196xf32>) -> tensor<1x4x64x196xf32>
      %276 = stablehlo.slice %275 [0:1, 0:4, 0:16, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x16x196xf32>
      %277 = stablehlo.slice %275 [0:1, 0:4, 16:32, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x16x196xf32>
      mhlo.return %276, %277 : tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> (tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>)
    %37 = stablehlo.tuple %36#0, %36#1 {xla_shape = "(f32[1,4,16,196]{3,2,1,0}, f32[1,4,16,196]{3,2,1,0})"} : tuple<tensor<1x4x16x196xf32>, tensor<1x4x16x196xf32>>
    %38 = mhlo.bitcast %36#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
    %39 = mhlo.bitcast %36#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
    %40 = "mhlo.fusion"(%38, %39) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<4x16x196xf32>, %arg280: tensor<4x16x196xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x196xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<4x196x196xf32>
      mhlo.return %265 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<4x16x196xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
    %41 = "mhlo.fusion"(%40, %arg25, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196x196xf32>, %arg280: tensor<4x196xf32>, %arg281: tensor<196x196xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %264 = stablehlo.compare LT, %arg281, %263 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
      %c_0 = stablehlo.constant dense<196> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %266 = stablehlo.add %arg281, %265 : tensor<196x196xi32>
      %267 = stablehlo.select %264, %266, %arg281 : tensor<196x196xi1>, tensor<196x196xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
      %269 = "stablehlo.gather"(%arg280, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[38416,4,1]{0,1,2}"} : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<4x196x196xf32>
      %271 = stablehlo.add %arg279, %270 : tensor<4x196x196xf32>
      mhlo.return %271 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x196xf32>, tensor<4x196xf32>, tensor<196x196xi32>) -> tensor<4x196x196xf32>
    %42 = "mhlo.fusion"(%41) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<4x196x196xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %263 = stablehlo.reduce(%arg279 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<4x196x196xf32>, tensor<f32>) -> tensor<4x196xf32>
      %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1] : (tensor<4x196xf32>) -> tensor<4x196x196xf32>
      %265 = stablehlo.subtract %arg279, %264 : tensor<4x196x196xf32>
      %266 = stablehlo.exponential %265 : tensor<4x196x196xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %267 = stablehlo.reduce(%266 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<4x196x196xf32>, tensor<f32>) -> tensor<4x196xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1] : (tensor<4x196xf32>) -> tensor<4x196x196xf32>
      %269 = stablehlo.divide %266, %268 : tensor<4x196x196xf32>
      mhlo.return %269 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x196xf32>) -> tensor<4x196x196xf32>
    %43 = "mhlo.fusion"(%arg28, %arg27, %arg183, %35, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<256x196xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<256x196xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<256x196xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<256x196xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<256xf32>) -> tensor<256x196xf32>
      %274 = stablehlo.add %272, %273 : tensor<256x196xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x196xf32>) -> tensor<1x4x64x196xf32>
      %276 = stablehlo.slice %275 [0:1, 0:4, 32:64, 0:196] : (tensor<1x4x64x196xf32>) -> tensor<1x4x32x196xf32>
      mhlo.return %276 : tensor<1x4x32x196xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<1x4x32x196xf32>
    %44 = mhlo.bitcast %43 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x32x196xf32>) -> tensor<4x32x196xf32>
    %45 = "mhlo.fusion"(%42, %44) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<4x196x196xf32>, %arg280: tensor<4x32x196xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x196x196xf32>, tensor<4x32x196xf32>) -> tensor<4x196x32xf32>
      %264 = stablehlo.transpose %263, dims = [1, 0, 2] : (tensor<4x196x32xf32>) -> tensor<196x4x32xf32>
      mhlo.return %264 : tensor<196x4x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x196xf32>, tensor<4x32x196xf32>) -> tensor<196x4x32xf32>
    %46 = "mhlo.fusion"(%45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<196x4x32xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<196x4x32xf32>) -> tensor<1x196x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %266 = stablehlo.add %263, %265 : tensor<1x196x128xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %268 = stablehlo.clamp %264, %266, %267 : tensor<1x196x128xf32>
      %269 = stablehlo.multiply %263, %268 : tensor<1x196x128xf32>
      mhlo.return %269 : tensor<1x196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<196x4x32xf32>) -> tensor<1x196x128xf32>
    %47 = mhlo.bitcast %46 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    %48 = "mhlo.fusion"(%47, %arg29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x128xf32>, %arg280: tensor<128x128xf32>):
      %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x128xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<196x128xf32>
      %265 = stablehlo.dot_general %264, %arg280, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
      mhlo.return %265 : tensor<196x128xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
    %49 = "mhlo.fusion"(%arg31, %arg30, %arg185, %48, %arg186, %arg24, %arg23, %arg181, %arg180, %32, %arg18, %arg17, %arg177, %arg176, %26, %arg11, %arg10, %arg172, %10, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<196x128xf32>, %arg283: tensor<128xf32>, %arg284: tensor<128xf32>, %arg285: tensor<128xf32>, %arg286: tensor<128xf32>, %arg287: tensor<128xf32>, %arg288: tensor<196x128xf32>, %arg289: tensor<128xf32>, %arg290: tensor<128xf32>, %arg291: tensor<128xf32>, %arg292: tensor<128xf32>, %arg293: tensor<196x128xf32>, %arg294: tensor<128xf32>, %arg295: tensor<128xf32>, %arg296: tensor<128xf32>, %arg297: tensor<1x128x14x14xf32>, %arg298: tensor<128xf32>):
      %263 = stablehlo.broadcast_in_dim %arg298, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %264 = stablehlo.subtract %arg297, %263 : tensor<1x128x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %266 = stablehlo.add %arg296, %265 : tensor<128xf32>
      %267 = stablehlo.rsqrt %266 : tensor<128xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x128x14x14xf32>
      %270 = stablehlo.broadcast_in_dim %arg295, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1x128x14x14xf32>
      %272 = stablehlo.broadcast_in_dim %arg294, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %273 = stablehlo.add %271, %272 : tensor<1x128x14x14xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %275 = mhlo.bitcast %arg293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %276 = stablehlo.broadcast_in_dim %arg292, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %277 = stablehlo.subtract %275, %276 : tensor<128x196xf32>
      %278 = stablehlo.add %arg291, %265 : tensor<128xf32>
      %279 = stablehlo.rsqrt %278 : tensor<128xf32>
      %280 = stablehlo.broadcast_in_dim %279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %281 = stablehlo.multiply %277, %280 : tensor<128x196xf32>
      %282 = stablehlo.broadcast_in_dim %arg290, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %283 = stablehlo.multiply %281, %282 : tensor<128x196xf32>
      %284 = stablehlo.broadcast_in_dim %arg289, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %285 = stablehlo.add %283, %284 : tensor<128x196xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %287 = stablehlo.add %274, %286 : tensor<1x128x196xf32>
      %288 = mhlo.bitcast %arg288 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %289 = stablehlo.broadcast_in_dim %arg287, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %290 = stablehlo.subtract %288, %289 : tensor<128x196xf32>
      %291 = stablehlo.add %arg286, %265 : tensor<128xf32>
      %292 = stablehlo.rsqrt %291 : tensor<128xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %294 = stablehlo.multiply %290, %293 : tensor<128x196xf32>
      %295 = stablehlo.broadcast_in_dim %arg285, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<128x196xf32>
      %297 = stablehlo.broadcast_in_dim %arg284, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %298 = stablehlo.add %296, %297 : tensor<128x196xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %300 = stablehlo.add %287, %299 : tensor<1x128x196xf32>
      %301 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %302 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %303 = stablehlo.subtract %301, %302 : tensor<128x196xf32>
      %304 = stablehlo.add %arg283, %265 : tensor<128xf32>
      %305 = stablehlo.rsqrt %304 : tensor<128xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %307 = stablehlo.multiply %303, %306 : tensor<128x196xf32>
      %308 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<128x196xf32>
      %310 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %311 = stablehlo.add %309, %310 : tensor<128x196xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %313 = stablehlo.add %300, %312 : tensor<1x128x196xf32>
      mhlo.return %313 : tensor<1x128x196xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x14x14xf32>, tensor<128xf32>) -> tensor<1x128x196xf32>
    %50 = mhlo.bitcast %49 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<1x128x196xf32>) -> tensor<196x128xf32>
    %51 = "mhlo.fusion"(%50, %arg32) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x128xf32>, %arg280: tensor<256x128xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
      mhlo.return %263 : tensor<196x256xf32>
    }) {output_operand_aliasing = []} : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %52 = "mhlo.fusion"(%arg34, %arg33, %51, %arg187, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<196x256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<196x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<256xf32>
      %267 = stablehlo.rsqrt %266 : tensor<256xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<196x256xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<196x256xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %273 = stablehlo.add %271, %272 : tensor<196x256xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %275 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %277 = stablehlo.add %274, %276 : tensor<1x196x256xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %278 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %279 = stablehlo.clamp %275, %277, %278 : tensor<1x196x256xf32>
      %280 = stablehlo.multiply %274, %279 : tensor<1x196x256xf32>
      mhlo.return %280 : tensor<1x196x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x196x256xf32>
    %53 = mhlo.bitcast %52 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x256xf32>) -> tensor<196x256xf32>
    %54 = "mhlo.fusion"(%53, %arg35) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x256xf32>, %arg280: tensor<128x256xf32>):
      %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x256xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<196x256xf32>
      %265 = stablehlo.dot_general %264, %arg280, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
      mhlo.return %265 : tensor<196x128xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
    %55 = "mhlo.fusion"(%arg37, %arg36, %arg189, %54, %arg31, %arg30, %arg185, %48, %arg190, %arg186, %arg24, %arg23, %arg181, %arg180, %32, %arg18, %arg17, %arg177, %arg176, %26, %arg11, %arg10, %arg172, %10, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<196x128xf32>, %arg283: tensor<128xf32>, %arg284: tensor<128xf32>, %arg285: tensor<128xf32>, %arg286: tensor<196x128xf32>, %arg287: tensor<128xf32>, %arg288: tensor<128xf32>, %arg289: tensor<128xf32>, %arg290: tensor<128xf32>, %arg291: tensor<128xf32>, %arg292: tensor<128xf32>, %arg293: tensor<196x128xf32>, %arg294: tensor<128xf32>, %arg295: tensor<128xf32>, %arg296: tensor<128xf32>, %arg297: tensor<128xf32>, %arg298: tensor<196x128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<128xf32>, %arg301: tensor<128xf32>, %arg302: tensor<1x128x14x14xf32>, %arg303: tensor<128xf32>):
      %263 = stablehlo.broadcast_in_dim %arg303, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %264 = stablehlo.subtract %arg302, %263 : tensor<1x128x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %266 = stablehlo.add %arg301, %265 : tensor<128xf32>
      %267 = stablehlo.rsqrt %266 : tensor<128xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x128x14x14xf32>
      %270 = stablehlo.broadcast_in_dim %arg300, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1x128x14x14xf32>
      %272 = stablehlo.broadcast_in_dim %arg299, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %273 = stablehlo.add %271, %272 : tensor<1x128x14x14xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %275 = mhlo.bitcast %arg298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %276 = stablehlo.broadcast_in_dim %arg297, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %277 = stablehlo.subtract %275, %276 : tensor<128x196xf32>
      %278 = stablehlo.add %arg296, %265 : tensor<128xf32>
      %279 = stablehlo.rsqrt %278 : tensor<128xf32>
      %280 = stablehlo.broadcast_in_dim %279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %281 = stablehlo.multiply %277, %280 : tensor<128x196xf32>
      %282 = stablehlo.broadcast_in_dim %arg295, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %283 = stablehlo.multiply %281, %282 : tensor<128x196xf32>
      %284 = stablehlo.broadcast_in_dim %arg294, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %285 = stablehlo.add %283, %284 : tensor<128x196xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %287 = stablehlo.add %274, %286 : tensor<1x128x196xf32>
      %288 = mhlo.bitcast %arg293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %289 = stablehlo.broadcast_in_dim %arg292, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %290 = stablehlo.subtract %288, %289 : tensor<128x196xf32>
      %291 = stablehlo.add %arg291, %265 : tensor<128xf32>
      %292 = stablehlo.rsqrt %291 : tensor<128xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %294 = stablehlo.multiply %290, %293 : tensor<128x196xf32>
      %295 = stablehlo.broadcast_in_dim %arg290, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<128x196xf32>
      %297 = stablehlo.broadcast_in_dim %arg289, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %298 = stablehlo.add %296, %297 : tensor<128x196xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %300 = stablehlo.add %287, %299 : tensor<1x128x196xf32>
      %301 = mhlo.bitcast %arg286 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %302 = stablehlo.broadcast_in_dim %arg285, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %303 = stablehlo.subtract %301, %302 : tensor<128x196xf32>
      %304 = stablehlo.add %arg288, %265 : tensor<128xf32>
      %305 = stablehlo.rsqrt %304 : tensor<128xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %307 = stablehlo.multiply %303, %306 : tensor<128x196xf32>
      %308 = stablehlo.broadcast_in_dim %arg284, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<128x196xf32>
      %310 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %311 = stablehlo.add %309, %310 : tensor<128x196xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %313 = stablehlo.add %300, %312 : tensor<1x128x196xf32>
      %314 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %315 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %316 = stablehlo.subtract %314, %315 : tensor<128x196xf32>
      %317 = stablehlo.add %arg287, %265 : tensor<128xf32>
      %318 = stablehlo.rsqrt %317 : tensor<128xf32>
      %319 = stablehlo.broadcast_in_dim %318, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %320 = stablehlo.multiply %316, %319 : tensor<128x196xf32>
      %321 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<128x196xf32>
      %323 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %324 = stablehlo.add %322, %323 : tensor<128x196xf32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %326 = stablehlo.add %313, %325 : tensor<1x128x196xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x196xf32>) -> tensor<1x128x14x14xf32>
      %328 = stablehlo.slice %327 [0:1, 0:128, 0:14:2, 0:14] : (tensor<1x128x14x14xf32>) -> tensor<1x128x7x14xf32>
      %329 = stablehlo.slice %328 [0:1, 0:128, 0:7, 0:14:2] : (tensor<1x128x7x14xf32>) -> tensor<1x128x7x7xf32>
      mhlo.return %329 : tensor<1x128x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x14x14xf32>, tensor<128xf32>) -> tensor<1x128x7x7xf32>
    %56 = mhlo.bitcast %55 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[49,128]{0,1}"} : (tensor<1x128x7x7xf32>) -> tensor<49x128xf32>
    %57 = "mhlo.fusion"(%56, %arg42) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x128xf32>, %arg280: tensor<128x128xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,49]{0,1}"} : (tensor<128x128xf32>, tensor<49x128xf32>) -> tensor<128x49xf32>
      mhlo.return %263 : tensor<128x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,49]{0,1}"} : (tensor<49x128xf32>, tensor<128x128xf32>) -> tensor<128x49xf32>
    %58 = "mhlo.fusion"(%arg194, %57, %arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128x49xf32>, %arg281: tensor<128xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<128x49xf32>) -> tensor<49x128xf32>
      %264 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x128xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %267 = stablehlo.add %arg281, %266 : tensor<128xf32>
      %268 = stablehlo.rsqrt %267 : tensor<128xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x128xf32>
      mhlo.return %270 : tensor<49x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128x49xf32>, tensor<128xf32>) -> tensor<49x128xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x128xf32>) -> tensor<49x8x16xf32>
    %60 = mhlo.bitcast %arg43 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<8x16xf32>
    %61 = mhlo.bitcast %arg44 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<8x16xf32>
    %62 = "mhlo.fusion"(%arg37, %arg36, %arg189, %54, %arg31, %arg30, %arg185, %48, %arg190, %arg186, %arg24, %arg23, %arg181, %arg180, %32, %arg18, %arg17, %arg177, %arg176, %26, %arg11, %arg10, %arg172, %10, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<196x128xf32>, %arg283: tensor<128xf32>, %arg284: tensor<128xf32>, %arg285: tensor<128xf32>, %arg286: tensor<196x128xf32>, %arg287: tensor<128xf32>, %arg288: tensor<128xf32>, %arg289: tensor<128xf32>, %arg290: tensor<128xf32>, %arg291: tensor<128xf32>, %arg292: tensor<128xf32>, %arg293: tensor<196x128xf32>, %arg294: tensor<128xf32>, %arg295: tensor<128xf32>, %arg296: tensor<128xf32>, %arg297: tensor<128xf32>, %arg298: tensor<196x128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<128xf32>, %arg301: tensor<128xf32>, %arg302: tensor<1x128x14x14xf32>, %arg303: tensor<128xf32>):
      %263 = stablehlo.broadcast_in_dim %arg303, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %264 = stablehlo.subtract %arg302, %263 : tensor<1x128x14x14xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %266 = stablehlo.add %arg301, %265 : tensor<128xf32>
      %267 = stablehlo.rsqrt %266 : tensor<128xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1x128x14x14xf32>
      %270 = stablehlo.broadcast_in_dim %arg300, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1x128x14x14xf32>
      %272 = stablehlo.broadcast_in_dim %arg299, dims = [1] : (tensor<128xf32>) -> tensor<1x128x14x14xf32>
      %273 = stablehlo.add %271, %272 : tensor<1x128x14x14xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %275 = mhlo.bitcast %arg298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %276 = stablehlo.broadcast_in_dim %arg297, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %277 = stablehlo.subtract %275, %276 : tensor<128x196xf32>
      %278 = stablehlo.add %arg296, %265 : tensor<128xf32>
      %279 = stablehlo.rsqrt %278 : tensor<128xf32>
      %280 = stablehlo.broadcast_in_dim %279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %281 = stablehlo.multiply %277, %280 : tensor<128x196xf32>
      %282 = stablehlo.broadcast_in_dim %arg295, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %283 = stablehlo.multiply %281, %282 : tensor<128x196xf32>
      %284 = stablehlo.broadcast_in_dim %arg294, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %285 = stablehlo.add %283, %284 : tensor<128x196xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %287 = stablehlo.add %274, %286 : tensor<1x128x196xf32>
      %288 = mhlo.bitcast %arg293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %289 = stablehlo.broadcast_in_dim %arg292, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %290 = stablehlo.subtract %288, %289 : tensor<128x196xf32>
      %291 = stablehlo.add %arg291, %265 : tensor<128xf32>
      %292 = stablehlo.rsqrt %291 : tensor<128xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %294 = stablehlo.multiply %290, %293 : tensor<128x196xf32>
      %295 = stablehlo.broadcast_in_dim %arg290, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<128x196xf32>
      %297 = stablehlo.broadcast_in_dim %arg289, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %298 = stablehlo.add %296, %297 : tensor<128x196xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %300 = stablehlo.add %287, %299 : tensor<1x128x196xf32>
      %301 = mhlo.bitcast %arg286 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %302 = stablehlo.broadcast_in_dim %arg285, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %303 = stablehlo.subtract %301, %302 : tensor<128x196xf32>
      %304 = stablehlo.add %arg288, %265 : tensor<128xf32>
      %305 = stablehlo.rsqrt %304 : tensor<128xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %307 = stablehlo.multiply %303, %306 : tensor<128x196xf32>
      %308 = stablehlo.broadcast_in_dim %arg284, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<128x196xf32>
      %310 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %311 = stablehlo.add %309, %310 : tensor<128x196xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %313 = stablehlo.add %300, %312 : tensor<1x128x196xf32>
      %314 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<128x196xf32>
      %315 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %316 = stablehlo.subtract %314, %315 : tensor<128x196xf32>
      %317 = stablehlo.add %arg287, %265 : tensor<128xf32>
      %318 = stablehlo.rsqrt %317 : tensor<128xf32>
      %319 = stablehlo.broadcast_in_dim %318, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %320 = stablehlo.multiply %316, %319 : tensor<128x196xf32>
      %321 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<128x196xf32>
      %323 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<128xf32>) -> tensor<128x196xf32>
      %324 = stablehlo.add %322, %323 : tensor<128x196xf32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x196xf32>) -> tensor<1x128x196xf32>
      %326 = stablehlo.add %313, %325 : tensor<1x128x196xf32>
      mhlo.return %326 : tensor<1x128x196xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x14x14xf32>, tensor<128xf32>) -> tensor<1x128x196xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[196,128]{0,1}"} : (tensor<1x128x196xf32>) -> tensor<196x128xf32>
    %64 = "mhlo.fusion"(%63, %arg39) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<196x128xf32>, %arg280: tensor<640x128xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,640]{0,1}"} : (tensor<196x128xf32>, tensor<640x128xf32>) -> tensor<196x640xf32>
      mhlo.return %263 : tensor<196x640xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,640]{0,1}"} : (tensor<196x128xf32>, tensor<640x128xf32>) -> tensor<196x640xf32>
    %65 = "mhlo.fusion"(%64, %arg41, %arg40, %arg192, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<196x640xf32>, %arg280: tensor<640xf32>, %arg281: tensor<640xf32>, %arg282: tensor<640xf32>, %arg283: tensor<640xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<640x196xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<640x196xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<640xf32>
      %268 = stablehlo.rsqrt %267 : tensor<640xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<640x196xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<640x196xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
      %274 = stablehlo.add %272, %273 : tensor<640x196xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x196xf32>) -> tensor<1x8x80x196xf32>
      %276 = stablehlo.slice %275 [0:1, 0:8, 0:16, 0:196] : (tensor<1x8x80x196xf32>) -> tensor<1x8x16x196xf32>
      mhlo.return %276 : tensor<1x8x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<196x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>) -> tensor<1x8x16x196xf32>
    %66 = mhlo.bitcast %65 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x196xf32>) -> tensor<8x16x196xf32>
    %67 = "mhlo.fusion"(%59, %60, %61, %66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x8x16xf32>, %arg280: tensor<8x16xf32>, %arg281: tensor<8x16xf32>, %arg282: tensor<8x16x196xf32>):
      %263 = stablehlo.broadcast_in_dim %arg280, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<49x8x16xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<49x8x16xf32>
      %265 = stablehlo.broadcast_in_dim %arg281, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<49x8x16xf32>
      %266 = stablehlo.add %264, %265 : tensor<49x8x16xf32>
      %267 = stablehlo.transpose %266, dims = [1, 0, 2] : (tensor<49x8x16xf32>) -> tensor<8x49x16xf32>
      %268 = stablehlo.dot_general %arg282, %267, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,196,49]{1,2,0}"} : (tensor<8x16x196xf32>, tensor<8x49x16xf32>) -> tensor<8x196x49xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %269 = stablehlo.broadcast_in_dim %cst, dims = [] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,196,49]{1,2,0}"} : (tensor<f32>) -> tensor<8x196x49xf32>
      %270 = stablehlo.multiply %268, %269 {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,196,49]{1,2,0}"} : tensor<8x196x49xf32>
      mhlo.return %270 : tensor<8x196x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,196,49]{1,2,0}"} : (tensor<49x8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16x196xf32>) -> tensor<8x196x49xf32>
    %68 = "mhlo.fusion"(%67, %arg38, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x196x49xf32>, %arg280: tensor<8x196xf32>, %arg281: tensor<49x196xi32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<8x196x49xf32>) -> tensor<8x49x196xf32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %264 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
      %265 = stablehlo.compare LT, %arg281, %264 : (tensor<49x196xi32>, tensor<49x196xi32>) -> tensor<49x196xi1>
      %c_0 = stablehlo.constant dense<196> : tensor<i32>
      %266 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
      %267 = stablehlo.add %arg281, %266 : tensor<49x196xi32>
      %268 = stablehlo.select %265, %267, %arg281 : tensor<49x196xi1>, tensor<49x196xi32>
      %269 = mhlo.bitcast %268 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[9604,1]{0,1}"} : (tensor<49x196xi32>) -> tensor<9604x1xi32>
      %270 = "stablehlo.gather"(%arg280, %269) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[9604,8,1]{0,1,2}"} : (tensor<8x196xf32>, tensor<9604x1xi32>) -> tensor<9604x8x1xf32>
      %271 = mhlo.bitcast %270 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<9604x8x1xf32>) -> tensor<8x49x196xf32>
      %272 = stablehlo.add %263, %271 : tensor<8x49x196xf32>
      mhlo.return %272 : tensor<8x49x196xf32>
    }) {output_operand_aliasing = []} : (tensor<8x196x49xf32>, tensor<8x196xf32>, tensor<49x196xi32>) -> tensor<8x49x196xf32>
    %69 = "mhlo.fusion"(%68) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x49x196xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %263 = stablehlo.reduce(%arg279 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x49x196xf32>, tensor<f32>) -> tensor<8x49xf32>
      %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1] : (tensor<8x49xf32>) -> tensor<8x49x196xf32>
      %265 = stablehlo.subtract %arg279, %264 : tensor<8x49x196xf32>
      %266 = stablehlo.exponential %265 : tensor<8x49x196xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %267 = stablehlo.reduce(%266 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x49x196xf32>, tensor<f32>) -> tensor<8x49xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1] : (tensor<8x49xf32>) -> tensor<8x49x196xf32>
      %269 = stablehlo.divide %266, %268 : tensor<8x49x196xf32>
      mhlo.return %269 : tensor<8x49x196xf32>
    }) {output_operand_aliasing = []} : (tensor<8x49x196xf32>) -> tensor<8x49x196xf32>
    %70 = "mhlo.fusion"(%64, %arg41, %arg40, %arg192, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<196x640xf32>, %arg280: tensor<640xf32>, %arg281: tensor<640xf32>, %arg282: tensor<640xf32>, %arg283: tensor<640xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<640x196xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<640x196xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<640xf32>
      %268 = stablehlo.rsqrt %267 : tensor<640xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<640x196xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<640x196xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<640xf32>) -> tensor<640x196xf32>
      %274 = stablehlo.add %272, %273 : tensor<640x196xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x196xf32>) -> tensor<1x8x80x196xf32>
      %276 = stablehlo.slice %275 [0:1, 0:8, 16:80, 0:196] : (tensor<1x8x80x196xf32>) -> tensor<1x8x64x196xf32>
      mhlo.return %276 : tensor<1x8x64x196xf32>
    }) {output_operand_aliasing = []} : (tensor<196x640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<640xf32>) -> tensor<1x8x64x196xf32>
    %71 = mhlo.bitcast %70 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x64x196xf32>) -> tensor<8x64x196xf32>
    %72 = "mhlo.fusion"(%69, %71) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x49x196xf32>, %arg280: tensor<8x64x196xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,64,49]{1,2,0}"} : (tensor<8x64x196xf32>, tensor<8x49x196xf32>) -> tensor<8x64x49xf32>
      %264 = stablehlo.transpose %263, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[49,64,8]{1,2,0}"} : (tensor<8x64x49xf32>) -> tensor<49x64x8xf32>
      mhlo.return %264 : tensor<49x64x8xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[49,64,8]{1,2,0}"} : (tensor<8x49x196xf32>, tensor<8x64x196xf32>) -> tensor<49x64x8xf32>
    %73 = "mhlo.fusion"(%72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x64x8xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<49x64x8xf32>) -> tensor<1x49x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %266 = stablehlo.add %263, %265 : tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %268 = stablehlo.clamp %264, %266, %267 : tensor<1x49x512xf32>
      %269 = stablehlo.multiply %263, %268 : tensor<1x49x512xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %271 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x512xf32>
      mhlo.return %272 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<49x64x8xf32>) -> tensor<49x512xf32>
    %74 = stablehlo.custom_call @__cublas$gemm(%arg45, %73) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22131072\22,\22rhs_stride\22:\2225088\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[256,49]{0,1}, s8[624640]{0})"} : (tensor<256x512xf32>, tensor<49x512xf32>) -> tuple<tensor<256x49xf32>, tensor<624640xi8>>
    %75 = stablehlo.get_tuple_element %74[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tuple<tensor<256x49xf32>, tensor<624640xi8>>) -> tensor<256x49xf32>
    %76 = "mhlo.fusion"(%75, %arg196, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x49xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg281, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x256xf32>
      mhlo.return %270 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x49xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %77 = "mhlo.fusion"(%76, %arg46, %arg47, %arg48) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<512x256xf32>):
      %263 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<49x256xf32>
      %265 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %266 = stablehlo.add %264, %265 : tensor<49x256xf32>
      %267 = stablehlo.dot_general %arg282, %266, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<512x256xf32>, tensor<49x256xf32>) -> tensor<512x49xf32>
      mhlo.return %267 : tensor<512x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256xf32>) -> tensor<512x49xf32>
    %78 = "mhlo.fusion"(%arg50, %arg49, %arg198, %77, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x49xf32>, %arg283: tensor<512xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x49xf32>) -> tensor<49x512xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x512xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<512xf32>
      %268 = stablehlo.rsqrt %267 : tensor<512xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x512xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x512xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x512xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x49x512xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x49x512xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<49x512xf32>
      mhlo.return %284 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x49xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %79 = stablehlo.custom_call @__cublas$gemm(%arg51, %78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22131072\22,\22rhs_stride\22:\2225088\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[256,49]{0,1}, s8[624640]{0})"} : (tensor<256x512xf32>, tensor<49x512xf32>) -> tuple<tensor<256x49xf32>, tensor<624640xi8>>
    %80 = stablehlo.get_tuple_element %79[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tuple<tensor<256x49xf32>, tensor<624640xi8>>) -> tensor<256x49xf32>
    %81 = "mhlo.fusion"(%76, %80, %arg53, %arg52, %arg200, %arg47, %arg46, %arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256x49xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>, %arg285: tensor<256xf32>, %arg286: tensor<256xf32>):
      %263 = stablehlo.broadcast_in_dim %arg285, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<49x256xf32>
      %265 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %266 = stablehlo.add %264, %265 : tensor<49x256xf32>
      %267 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
      %268 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %269 = stablehlo.subtract %267, %268 : tensor<49x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %270 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %271 = stablehlo.add %arg286, %270 : tensor<256xf32>
      %272 = stablehlo.rsqrt %271 : tensor<256xf32>
      %273 = stablehlo.broadcast_in_dim %272, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %274 = stablehlo.multiply %269, %273 : tensor<49x256xf32>
      %275 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %276 = stablehlo.multiply %274, %275 : tensor<49x256xf32>
      %277 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %278 = stablehlo.add %276, %277 : tensor<49x256xf32>
      %279 = stablehlo.add %266, %278 : tensor<49x256xf32>
      mhlo.return %279 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256x49xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %82 = "mhlo.fusion"(%81, %arg55) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<384x256xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<384x256xf32>, tensor<49x256xf32>) -> tensor<384x49xf32>
      mhlo.return %263 : tensor<384x49xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<384x49xf32>
    %83:2 = "mhlo.fusion"(%arg57, %arg56, %82, %arg203, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384x49xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<384x49xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<384xf32>
      %267 = stablehlo.rsqrt %266 : tensor<384xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<384x49xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<384x49xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %273 = stablehlo.add %271, %272 : tensor<384x49xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
      %275 = stablehlo.slice %274 [0:1, 0:6, 16:32, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
      %276 = stablehlo.slice %274 [0:1, 0:6, 0:16, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
      mhlo.return %275, %276 : tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384x49xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>)
    %84 = stablehlo.tuple %83#0, %83#1 {xla_shape = "(f32[1,6,16,49]{3,2,1,0}, f32[1,6,16,49]{3,2,1,0})"} : tuple<tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>>
    %85 = mhlo.bitcast %83#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    %86 = mhlo.bitcast %83#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    %87 = "mhlo.fusion"(%85, %86) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<6x16x49xf32>, %arg280: tensor<6x16x49xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<6x49x49xf32>
      mhlo.return %265 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %88 = "mhlo.fusion"(%87, %arg54, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x49xf32>, %arg280: tensor<6x49xf32>, %arg281: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %264 = stablehlo.compare LT, %arg281, %263 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_0 = stablehlo.constant dense<49> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %266 = stablehlo.add %arg281, %265 : tensor<49x49xi32>
      %267 = stablehlo.select %264, %266, %arg281 : tensor<49x49xi1>, tensor<49x49xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %269 = "stablehlo.gather"(%arg280, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[2401,6,1]{0,1,2}"} : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<6x49x49xf32>
      %271 = stablehlo.add %arg279, %270 : tensor<6x49x49xf32>
      mhlo.return %271 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x49xf32>, tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<6x49x49xf32>
    %89 = "mhlo.fusion"(%88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<6x49x49xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %263 = stablehlo.reduce(%arg279 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
      %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
      %265 = stablehlo.subtract %arg279, %264 : tensor<6x49x49xf32>
      %266 = stablehlo.exponential %265 : tensor<6x49x49xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %267 = stablehlo.reduce(%266 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
      %269 = stablehlo.divide %266, %268 : tensor<6x49x49xf32>
      mhlo.return %269 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x49xf32>) -> tensor<6x49x49xf32>
    %90 = "mhlo.fusion"(%arg57, %arg56, %82, %arg203, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384x49xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<384x49xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<384xf32>
      %267 = stablehlo.rsqrt %266 : tensor<384xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<384x49xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<384x49xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %273 = stablehlo.add %271, %272 : tensor<384x49xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
      %275 = stablehlo.slice %274 [0:1, 0:6, 32:64, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x32x49xf32>
      mhlo.return %275 : tensor<1x6x32x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384x49xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x6x32x49xf32>
    %91 = mhlo.bitcast %90 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x32x49xf32>) -> tensor<6x32x49xf32>
    %92 = stablehlo.custom_call @__cublas$gemm(%89, %91) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\222\22],\22rhs_contracting_dimensions\22:[\222\22],\22lhs_batch_dimensions\22:[\220\22],\22rhs_batch_dimensions\22:[\220\22]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\222401\22,\22rhs_stride\22:\221568\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[6,49,32]{2,1,0}, s8[95256]{0})"} : (tensor<6x49x49xf32>, tensor<6x32x49xf32>) -> tuple<tensor<6x49x32xf32>, tensor<95256xi8>>
    %93 = stablehlo.get_tuple_element %92[0] : (tuple<tensor<6x49x32xf32>, tensor<95256xi8>>) -> tensor<6x49x32xf32>
    %94 = "mhlo.fusion"(%93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %263 = stablehlo.transpose %arg279, dims = [1, 0, 2] : (tensor<6x49x32xf32>) -> tensor<49x6x32xf32>
      %264 = mhlo.bitcast %263 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %267 = stablehlo.add %264, %266 : tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %268 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %269 = stablehlo.clamp %265, %267, %268 : tensor<1x49x192xf32>
      %270 = stablehlo.multiply %264, %269 : tensor<1x49x192xf32>
      mhlo.return %270 : tensor<1x49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<1x49x192xf32>
    %95 = mhlo.bitcast %94 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
    %96 = "mhlo.fusion"(%95, %arg58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x192xf32>, %arg280: tensor<256x192xf32>):
      %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x192xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<49x192xf32>
      %265 = stablehlo.dot_general %arg280, %264, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<256x192xf32>, tensor<49x192xf32>) -> tensor<256x49xf32>
      mhlo.return %265 : tensor<256x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<256x49xf32>
    %97 = "mhlo.fusion"(%81, %96, %arg60, %arg59, %arg205, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256x49xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x256xf32>
      %271 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x256xf32>
      %273 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x256xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<49x256xf32>
      mhlo.return %275 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256x49xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %98 = "mhlo.fusion"(%97, %arg61) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<512x256xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<512x256xf32>, tensor<49x256xf32>) -> tensor<512x49xf32>
      mhlo.return %263 : tensor<512x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<512x49xf32>
    %99 = "mhlo.fusion"(%arg63, %arg62, %arg207, %98, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x49xf32>, %arg283: tensor<512xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x49xf32>) -> tensor<49x512xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x512xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<512xf32>
      %268 = stablehlo.rsqrt %267 : tensor<512xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x512xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x512xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x512xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x49x512xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x49x512xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<49x512xf32>
      mhlo.return %284 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x49xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %100 = stablehlo.custom_call @__cublas$gemm(%arg64, %99) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22131072\22,\22rhs_stride\22:\2225088\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[256,49]{0,1}, s8[624640]{0})"} : (tensor<256x512xf32>, tensor<49x512xf32>) -> tuple<tensor<256x49xf32>, tensor<624640xi8>>
    %101 = stablehlo.get_tuple_element %100[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tuple<tensor<256x49xf32>, tensor<624640xi8>>) -> tensor<256x49xf32>
    %102 = "mhlo.fusion"(%97, %101, %arg66, %arg65, %arg209, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256x49xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x256xf32>
      %271 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x256xf32>
      %273 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x256xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<49x256xf32>
      mhlo.return %275 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256x49xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %103 = "mhlo.fusion"(%102, %arg68) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<384x256xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<384x256xf32>, tensor<49x256xf32>) -> tensor<384x49xf32>
      mhlo.return %263 : tensor<384x49xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<384x49xf32>
    %104:2 = "mhlo.fusion"(%arg70, %arg69, %103, %arg212, %arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384x49xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<384x49xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<384xf32>
      %267 = stablehlo.rsqrt %266 : tensor<384xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<384x49xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<384x49xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %273 = stablehlo.add %271, %272 : tensor<384x49xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
      %275 = stablehlo.slice %274 [0:1, 0:6, 16:32, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
      %276 = stablehlo.slice %274 [0:1, 0:6, 0:16, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
      mhlo.return %275, %276 : tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384x49xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>)
    %105 = stablehlo.tuple %104#0, %104#1 {xla_shape = "(f32[1,6,16,49]{3,2,1,0}, f32[1,6,16,49]{3,2,1,0})"} : tuple<tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>>
    %106 = mhlo.bitcast %104#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    %107 = mhlo.bitcast %104#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    %108 = "mhlo.fusion"(%106, %107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<6x16x49xf32>, %arg280: tensor<6x16x49xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<6x49x49xf32>
      mhlo.return %265 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %109 = "mhlo.fusion"(%108, %arg67, %arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x49xf32>, %arg280: tensor<6x49xf32>, %arg281: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %264 = stablehlo.compare LT, %arg281, %263 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_0 = stablehlo.constant dense<49> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %266 = stablehlo.add %arg281, %265 : tensor<49x49xi32>
      %267 = stablehlo.select %264, %266, %arg281 : tensor<49x49xi1>, tensor<49x49xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %269 = "stablehlo.gather"(%arg280, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[2401,6,1]{0,1,2}"} : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<6x49x49xf32>
      %271 = stablehlo.add %arg279, %270 : tensor<6x49x49xf32>
      mhlo.return %271 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x49xf32>, tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<6x49x49xf32>
    %110 = "mhlo.fusion"(%109) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<6x49x49xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %263 = stablehlo.reduce(%arg279 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
      %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
      %265 = stablehlo.subtract %arg279, %264 : tensor<6x49x49xf32>
      %266 = stablehlo.exponential %265 : tensor<6x49x49xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %267 = stablehlo.reduce(%266 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
      %269 = stablehlo.divide %266, %268 : tensor<6x49x49xf32>
      mhlo.return %269 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x49xf32>) -> tensor<6x49x49xf32>
    %111 = "mhlo.fusion"(%arg70, %arg69, %103, %arg212, %arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384x49xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<384x49xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<384xf32>
      %267 = stablehlo.rsqrt %266 : tensor<384xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<384x49xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<384x49xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %273 = stablehlo.add %271, %272 : tensor<384x49xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
      %275 = stablehlo.slice %274 [0:1, 0:6, 32:64, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x32x49xf32>
      mhlo.return %275 : tensor<1x6x32x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384x49xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x6x32x49xf32>
    %112 = mhlo.bitcast %111 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x32x49xf32>) -> tensor<6x32x49xf32>
    %113 = stablehlo.custom_call @__cublas$gemm(%110, %112) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\222\22],\22rhs_contracting_dimensions\22:[\222\22],\22lhs_batch_dimensions\22:[\220\22],\22rhs_batch_dimensions\22:[\220\22]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\222401\22,\22rhs_stride\22:\221568\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[6,49,32]{2,1,0}, s8[95256]{0})"} : (tensor<6x49x49xf32>, tensor<6x32x49xf32>) -> tuple<tensor<6x49x32xf32>, tensor<95256xi8>>
    %114 = stablehlo.get_tuple_element %113[0] : (tuple<tensor<6x49x32xf32>, tensor<95256xi8>>) -> tensor<6x49x32xf32>
    %115 = "mhlo.fusion"(%114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %263 = stablehlo.transpose %arg279, dims = [1, 0, 2] : (tensor<6x49x32xf32>) -> tensor<49x6x32xf32>
      %264 = mhlo.bitcast %263 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %267 = stablehlo.add %264, %266 : tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %268 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %269 = stablehlo.clamp %265, %267, %268 : tensor<1x49x192xf32>
      %270 = stablehlo.multiply %264, %269 : tensor<1x49x192xf32>
      mhlo.return %270 : tensor<1x49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<1x49x192xf32>
    %116 = mhlo.bitcast %115 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
    %117 = "mhlo.fusion"(%116, %arg71) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x192xf32>, %arg280: tensor<256x192xf32>):
      %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x192xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<49x192xf32>
      %265 = stablehlo.dot_general %arg280, %264, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<256x192xf32>, tensor<49x192xf32>) -> tensor<256x49xf32>
      mhlo.return %265 : tensor<256x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<256x49xf32>
    %118 = "mhlo.fusion"(%102, %117, %arg73, %arg72, %arg214, %arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256x49xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x256xf32>
      %271 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x256xf32>
      %273 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x256xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<49x256xf32>
      mhlo.return %275 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256x49xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %119 = "mhlo.fusion"(%118, %arg74) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<512x256xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<512x256xf32>, tensor<49x256xf32>) -> tensor<512x49xf32>
      mhlo.return %263 : tensor<512x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<512x49xf32>
    %120 = "mhlo.fusion"(%arg76, %arg75, %arg216, %119, %arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x49xf32>, %arg283: tensor<512xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x49xf32>) -> tensor<49x512xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x512xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<512xf32>
      %268 = stablehlo.rsqrt %267 : tensor<512xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x512xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x512xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x512xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x49x512xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x49x512xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<49x512xf32>
      mhlo.return %284 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x49xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %121 = stablehlo.custom_call @__cublas$gemm(%arg77, %120) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22131072\22,\22rhs_stride\22:\2225088\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[256,49]{0,1}, s8[624640]{0})"} : (tensor<256x512xf32>, tensor<49x512xf32>) -> tuple<tensor<256x49xf32>, tensor<624640xi8>>
    %122 = stablehlo.get_tuple_element %121[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tuple<tensor<256x49xf32>, tensor<624640xi8>>) -> tensor<256x49xf32>
    %123 = "mhlo.fusion"(%118, %122, %arg79, %arg78, %arg218, %arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256x49xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x256xf32>
      %271 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x256xf32>
      %273 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x256xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<49x256xf32>
      mhlo.return %275 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256x49xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %124 = "mhlo.fusion"(%123, %arg81) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<384x256xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<384x256xf32>, tensor<49x256xf32>) -> tensor<384x49xf32>
      mhlo.return %263 : tensor<384x49xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<384x49xf32>
    %125:2 = "mhlo.fusion"(%arg83, %arg82, %124, %arg221, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384x49xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<384x49xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<384xf32>
      %267 = stablehlo.rsqrt %266 : tensor<384xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<384x49xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<384x49xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %273 = stablehlo.add %271, %272 : tensor<384x49xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
      %275 = stablehlo.slice %274 [0:1, 0:6, 16:32, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
      %276 = stablehlo.slice %274 [0:1, 0:6, 0:16, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x16x49xf32>
      mhlo.return %275, %276 : tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384x49xf32>, tensor<384xf32>, tensor<384xf32>) -> (tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>)
    %126 = stablehlo.tuple %125#0, %125#1 {xla_shape = "(f32[1,6,16,49]{3,2,1,0}, f32[1,6,16,49]{3,2,1,0})"} : tuple<tensor<1x6x16x49xf32>, tensor<1x6x16x49xf32>>
    %127 = mhlo.bitcast %125#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    %128 = mhlo.bitcast %125#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    %129 = "mhlo.fusion"(%127, %128) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<6x16x49xf32>, %arg280: tensor<6x16x49xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<6x49x49xf32>
      mhlo.return %265 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x16x49xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %130 = "mhlo.fusion"(%129, %arg80, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x49xf32>, %arg280: tensor<6x49xf32>, %arg281: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %264 = stablehlo.compare LT, %arg281, %263 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_0 = stablehlo.constant dense<49> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %266 = stablehlo.add %arg281, %265 : tensor<49x49xi32>
      %267 = stablehlo.select %264, %266, %arg281 : tensor<49x49xi1>, tensor<49x49xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %269 = "stablehlo.gather"(%arg280, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[2401,6,1]{0,1,2}"} : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<6x49x49xf32>
      %271 = stablehlo.add %arg279, %270 : tensor<6x49x49xf32>
      mhlo.return %271 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x49xf32>, tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<6x49x49xf32>
    %131 = "mhlo.fusion"(%130) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<6x49x49xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %263 = stablehlo.reduce(%arg279 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
      %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
      %265 = stablehlo.subtract %arg279, %264 : tensor<6x49x49xf32>
      %266 = stablehlo.exponential %265 : tensor<6x49x49xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %267 = stablehlo.reduce(%266 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<6x49x49xf32>, tensor<f32>) -> tensor<6x49xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1] : (tensor<6x49xf32>) -> tensor<6x49x49xf32>
      %269 = stablehlo.divide %266, %268 : tensor<6x49x49xf32>
      mhlo.return %269 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x49xf32>) -> tensor<6x49x49xf32>
    %132 = "mhlo.fusion"(%arg83, %arg82, %124, %arg221, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384x49xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %264 = stablehlo.subtract %arg281, %263 : tensor<384x49xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<384xf32>
      %267 = stablehlo.rsqrt %266 : tensor<384xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<384x49xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<384x49xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<384xf32>) -> tensor<384x49xf32>
      %273 = stablehlo.add %271, %272 : tensor<384x49xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x49xf32>) -> tensor<1x6x64x49xf32>
      %275 = stablehlo.slice %274 [0:1, 0:6, 32:64, 0:49] : (tensor<1x6x64x49xf32>) -> tensor<1x6x32x49xf32>
      mhlo.return %275 : tensor<1x6x32x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384x49xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x6x32x49xf32>
    %133 = mhlo.bitcast %132 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x32x49xf32>) -> tensor<6x32x49xf32>
    %134 = stablehlo.custom_call @__cublas$gemm(%131, %133) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\222\22],\22rhs_contracting_dimensions\22:[\222\22],\22lhs_batch_dimensions\22:[\220\22],\22rhs_batch_dimensions\22:[\220\22]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\222401\22,\22rhs_stride\22:\221568\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[6,49,32]{2,1,0}, s8[95256]{0})"} : (tensor<6x49x49xf32>, tensor<6x32x49xf32>) -> tuple<tensor<6x49x32xf32>, tensor<95256xi8>>
    %135 = stablehlo.get_tuple_element %134[0] : (tuple<tensor<6x49x32xf32>, tensor<95256xi8>>) -> tensor<6x49x32xf32>
    %136 = "mhlo.fusion"(%135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %263 = stablehlo.transpose %arg279, dims = [1, 0, 2] : (tensor<6x49x32xf32>) -> tensor<49x6x32xf32>
      %264 = mhlo.bitcast %263 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %267 = stablehlo.add %264, %266 : tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %268 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %269 = stablehlo.clamp %265, %267, %268 : tensor<1x49x192xf32>
      %270 = stablehlo.multiply %264, %269 : tensor<1x49x192xf32>
      mhlo.return %270 : tensor<1x49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<1x49x192xf32>
    %137 = mhlo.bitcast %136 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
    %138 = "mhlo.fusion"(%137, %arg84) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x192xf32>, %arg280: tensor<256x192xf32>):
      %cst = stablehlo.constant dense<0.166666672> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x192xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<49x192xf32>
      %265 = stablehlo.dot_general %arg280, %264, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<256x192xf32>, tensor<49x192xf32>) -> tensor<256x49xf32>
      mhlo.return %265 : tensor<256x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<256x49xf32>
    %139 = "mhlo.fusion"(%123, %138, %arg86, %arg85, %arg223, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256x49xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x256xf32>
      %271 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x256xf32>
      %273 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x256xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<49x256xf32>
      mhlo.return %275 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256x49xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %140 = "mhlo.fusion"(%139, %arg87) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<512x256xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<512x256xf32>, tensor<49x256xf32>) -> tensor<512x49xf32>
      mhlo.return %263 : tensor<512x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,49]{0,1}"} : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<512x49xf32>
    %141 = "mhlo.fusion"(%arg89, %arg88, %arg225, %140, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x49xf32>, %arg283: tensor<512xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x49xf32>) -> tensor<49x512xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x512xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<512xf32>
      %268 = stablehlo.rsqrt %267 : tensor<512xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x512xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x512xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x512xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x49x512xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x49x512xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<49x512xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<49x512xf32>
      mhlo.return %284 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x49xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %142 = stablehlo.custom_call @__cublas$gemm(%arg90, %141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22131072\22,\22rhs_stride\22:\2225088\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[256,49]{0,1}, s8[624640]{0})"} : (tensor<256x512xf32>, tensor<49x512xf32>) -> tuple<tensor<256x49xf32>, tensor<624640xi8>>
    %143 = stablehlo.get_tuple_element %142[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,49]{0,1}"} : (tuple<tensor<256x49xf32>, tensor<624640xi8>>) -> tensor<256x49xf32>
    %144 = "mhlo.fusion"(%139, %143, %arg92, %arg91, %arg227, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256x49xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x49xf32>) -> tensor<49x256xf32>
      %264 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<49x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<49x256xf32>
      %271 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<49x256xf32>
      %273 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %274 = stablehlo.add %272, %273 : tensor<49x256xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<49x256xf32>
      mhlo.return %275 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256x49xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %145 = "mhlo.fusion"(%144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x256xf32>) -> tensor<1x7x7x256xf32>
      %264 = stablehlo.slice %263 [0:1, 0:7:2, 0:7, 0:256] : (tensor<1x7x7x256xf32>) -> tensor<1x4x7x256xf32>
      %265 = stablehlo.slice %264 [0:1, 0:4, 0:7:2, 0:256] : (tensor<1x4x7x256xf32>) -> tensor<1x4x4x256xf32>
      mhlo.return %265 : tensor<1x4x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>) -> tensor<1x4x4x256xf32>
    %146 = mhlo.bitcast %145 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x4x256xf32>) -> tensor<16x256xf32>
    %147 = stablehlo.custom_call @__cublas$gemm(%arg97, %146) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\2265536\22,\22rhs_stride\22:\224096\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[256,16]{0,1}, s8[278528]{0})"} : (tensor<256x256xf32>, tensor<16x256xf32>) -> tuple<tensor<256x16xf32>, tensor<278528xi8>>
    %148 = stablehlo.get_tuple_element %147[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,16]{0,1}"} : (tuple<tensor<256x16xf32>, tensor<278528xi8>>) -> tensor<256x16xf32>
    %149 = "mhlo.fusion"(%arg232, %148, %arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256x16xf32>, %arg281: tensor<256xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x16xf32>) -> tensor<16x256xf32>
      %264 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x256xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %267 = stablehlo.add %arg281, %266 : tensor<256xf32>
      %268 = stablehlo.rsqrt %267 : tensor<256xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x256xf32>
      mhlo.return %270 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256x16xf32>, tensor<256xf32>) -> tensor<16x256xf32>
    %150 = mhlo.bitcast %149 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x256xf32>) -> tensor<16x16x16xf32>
    %151 = mhlo.bitcast %arg98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<16x16xf32>
    %152 = mhlo.bitcast %arg99 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<16x16xf32>
    %153 = "mhlo.fusion"(%144, %arg94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<1280x256xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x256xf32>, tensor<49x256xf32>) -> tensor<1280x49xf32>
      mhlo.return %263 : tensor<1280x49xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<1280x256xf32>) -> tensor<1280x49xf32>
    %154 = "mhlo.fusion"(%153, %arg96, %arg95, %arg230, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1280x49xf32>, %arg280: tensor<1280xf32>, %arg281: tensor<1280xf32>, %arg282: tensor<1280xf32>, %arg283: tensor<1280xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
      %264 = stablehlo.subtract %arg279, %263 : tensor<1280x49xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<1280xf32>
      %267 = stablehlo.rsqrt %266 : tensor<1280xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1280x49xf32>
      %270 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1280x49xf32>
      %272 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
      %273 = stablehlo.add %271, %272 : tensor<1280x49xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1280x49xf32>) -> tensor<1x16x80x49xf32>
      %275 = stablehlo.slice %274 [0:1, 0:16, 0:16, 0:49] : (tensor<1x16x80x49xf32>) -> tensor<1x16x16x49xf32>
      mhlo.return %275 : tensor<1x16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x49xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x16x16x49xf32>
    %155 = mhlo.bitcast %154 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x49xf32>) -> tensor<16x16x49xf32>
    %156 = "mhlo.fusion"(%150, %151, %152, %155) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<16x16x16xf32>, %arg280: tensor<16x16xf32>, %arg281: tensor<16x16xf32>, %arg282: tensor<16x16x49xf32>):
      %263 = stablehlo.broadcast_in_dim %arg280, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<16x16x16xf32>
      %264 = stablehlo.multiply %arg279, %263 : tensor<16x16x16xf32>
      %265 = stablehlo.broadcast_in_dim %arg281, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<16x16x16xf32>
      %266 = stablehlo.add %264, %265 : tensor<16x16x16xf32>
      %267 = stablehlo.transpose %266, dims = [1, 0, 2] : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
      %268 = stablehlo.dot_general %267, %arg282, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16x16xf32>, tensor<16x16x49xf32>) -> tensor<16x16x49xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %269 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x16x49xf32>
      %270 = stablehlo.multiply %268, %269 : tensor<16x16x49xf32>
      mhlo.return %270 : tensor<16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16x49xf32>) -> tensor<16x16x49xf32>
    %157 = "mhlo.fusion"(%156, %arg93, %arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x16x49xf32>, %arg280: tensor<16x49xf32>, %arg281: tensor<16x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
      %264 = stablehlo.compare LT, %arg281, %263 : (tensor<16x49xi32>, tensor<16x49xi32>) -> tensor<16x49xi1>
      %c_0 = stablehlo.constant dense<49> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
      %266 = stablehlo.add %arg281, %265 : tensor<16x49xi32>
      %267 = stablehlo.select %264, %266, %arg281 : tensor<16x49xi1>, tensor<16x49xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[784,1]{0,1}"} : (tensor<16x49xi32>) -> tensor<784x1xi32>
      %269 = "stablehlo.gather"(%arg280, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[784,16,1]{0,1,2}"} : (tensor<16x49xf32>, tensor<784x1xi32>) -> tensor<784x16x1xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<784x16x1xf32>) -> tensor<16x16x49xf32>
      %271 = stablehlo.add %arg279, %270 : tensor<16x16x49xf32>
      mhlo.return %271 : tensor<16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x49xf32>, tensor<16x49xf32>, tensor<16x49xi32>) -> tensor<16x16x49xf32>
    %158 = "mhlo.fusion"(%157) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<16x16x49xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %263 = stablehlo.reduce(%arg279 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<16x16x49xf32>, tensor<f32>) -> tensor<16x16xf32>
      %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1] : (tensor<16x16xf32>) -> tensor<16x16x49xf32>
      %265 = stablehlo.subtract %arg279, %264 : tensor<16x16x49xf32>
      %266 = stablehlo.exponential %265 : tensor<16x16x49xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %267 = stablehlo.reduce(%266 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x16x49xf32>, tensor<f32>) -> tensor<16x16xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1] : (tensor<16x16xf32>) -> tensor<16x16x49xf32>
      %269 = stablehlo.divide %266, %268 : tensor<16x16x49xf32>
      mhlo.return %269 : tensor<16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x49xf32>) -> tensor<16x16x49xf32>
    %159 = "mhlo.fusion"(%153, %arg96, %arg95, %arg230, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1280x49xf32>, %arg280: tensor<1280xf32>, %arg281: tensor<1280xf32>, %arg282: tensor<1280xf32>, %arg283: tensor<1280xf32>):
      %263 = stablehlo.broadcast_in_dim %arg282, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
      %264 = stablehlo.subtract %arg279, %263 : tensor<1280x49xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280xf32>
      %266 = stablehlo.add %arg283, %265 : tensor<1280xf32>
      %267 = stablehlo.rsqrt %266 : tensor<1280xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<1280x49xf32>
      %270 = stablehlo.broadcast_in_dim %arg281, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<1280x49xf32>
      %272 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<1280xf32>) -> tensor<1280x49xf32>
      %273 = stablehlo.add %271, %272 : tensor<1280x49xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1280x49xf32>) -> tensor<1x16x80x49xf32>
      %275 = stablehlo.slice %274 [0:1, 0:16, 16:80, 0:49] : (tensor<1x16x80x49xf32>) -> tensor<1x16x64x49xf32>
      mhlo.return %275 : tensor<1x16x64x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x49xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x16x64x49xf32>
    %160 = mhlo.bitcast %159 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x64x49xf32>) -> tensor<16x64x49xf32>
    %161 = "mhlo.fusion"(%158, %160) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<16x16x49xf32>, %arg280: tensor<16x64x49xf32>):
      %263 = stablehlo.dot_general %arg280, %arg279, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,64,16]{1,2,0}"} : (tensor<16x64x49xf32>, tensor<16x16x49xf32>) -> tensor<16x64x16xf32>
      %264 = stablehlo.transpose %263, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,64,16]{1,2,0}"} : (tensor<16x64x16xf32>) -> tensor<16x64x16xf32>
      mhlo.return %264 : tensor<16x64x16xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,64,16]{1,2,0}"} : (tensor<16x16x49xf32>, tensor<16x64x49xf32>) -> tensor<16x64x16xf32>
    %162 = "mhlo.fusion"(%161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x64x16xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<16x64x16xf32>) -> tensor<1x16x1024xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %266 = stablehlo.add %263, %265 : tensor<1x16x1024xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %268 = stablehlo.clamp %264, %266, %267 : tensor<1x16x1024xf32>
      %269 = stablehlo.multiply %263, %268 : tensor<1x16x1024xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x1024xf32>) -> tensor<16x1024xf32>
      %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %271 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x1024xf32>
      mhlo.return %272 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x64x16xf32>) -> tensor<16x1024xf32>
    %163 = stablehlo.custom_call @__cublas$gemm(%arg100, %162) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22393216\22,\22rhs_stride\22:\2216384\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[1638400]{0})"} : (tensor<384x1024xf32>, tensor<16x1024xf32>) -> tuple<tensor<384x16xf32>, tensor<1638400xi8>>
    %164 = stablehlo.get_tuple_element %163[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<1638400xi8>>) -> tensor<384x16xf32>
    %165 = "mhlo.fusion"(%arg102, %arg101, %arg234, %164, %arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384x16xf32>, %arg283: tensor<384xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      mhlo.return %274 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %166 = stablehlo.custom_call @__cublas$gemm(%arg103, %165) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[768,16]{0,1}, s8[1204224]{0})"} : (tensor<768x384xf32>, tensor<16x384xf32>) -> tuple<tensor<768x16xf32>, tensor<1204224xi8>>
    %167 = stablehlo.get_tuple_element %166[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,16]{0,1}"} : (tuple<tensor<768x16xf32>, tensor<1204224xi8>>) -> tensor<768x16xf32>
    %168 = "mhlo.fusion"(%arg105, %arg104, %arg236, %167, %arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<768x16xf32>, %arg283: tensor<768xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x768xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<768xf32>
      %268 = stablehlo.rsqrt %267 : tensor<768xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x768xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x768xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x768xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x16x768xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x16x768xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<16x768xf32>
      mhlo.return %284 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x16xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %169 = stablehlo.custom_call @__cublas$gemm(%arg106, %168) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\2212288\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[1228800]{0})"} : (tensor<384x768xf32>, tensor<16x768xf32>) -> tuple<tensor<384x16xf32>, tensor<1228800xi8>>
    %170 = stablehlo.get_tuple_element %169[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<1228800xi8>>) -> tensor<384x16xf32>
    %171 = "mhlo.fusion"(%165, %arg108, %arg107, %arg238, %170, %arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384x16xf32>, %arg284: tensor<384xf32>):
      %263 = mhlo.bitcast %arg283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<16x384xf32>
      mhlo.return %275 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %172 = stablehlo.custom_call @__cublas$gemm(%arg110, %171) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22196608\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[512,16]{1,0}, s8[811008]{0})"} : (tensor<512x384xf32>, tensor<16x384xf32>) -> tuple<tensor<512x16xf32>, tensor<811008xi8>>
    %173 = stablehlo.get_tuple_element %172[0] : (tuple<tensor<512x16xf32>, tensor<811008xi8>>) -> tensor<512x16xf32>
    %174:2 = "mhlo.fusion"(%arg112, %arg111, %arg242, %173, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x16xf32>, %arg283: tensor<512xf32>):
      %263 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %264 = stablehlo.subtract %arg282, %263 : tensor<512x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<512xf32>
      %267 = stablehlo.rsqrt %266 : tensor<512xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<512x16xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<512x16xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %273 = stablehlo.add %271, %272 : tensor<512x16xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
      %275 = stablehlo.slice %274 [0:1, 0:8, 16:32, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
      %276 = stablehlo.slice %274 [0:1, 0:8, 0:16, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
      mhlo.return %275, %276 : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>)
    %175 = stablehlo.tuple %174#0, %174#1 {xla_shape = "(f32[1,8,16,16]{3,2,1,0}, f32[1,8,16,16]{3,2,1,0})"} : tuple<tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>>
    %176 = mhlo.bitcast %174#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    %177 = mhlo.bitcast %174#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    %178 = "mhlo.fusion"(%176, %177) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<8x16x16xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<8x16x16xf32>
      mhlo.return %265 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %179 = "mhlo.fusion"(%arg109, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %264 = stablehlo.compare LT, %arg280, %263 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_0 = stablehlo.constant dense<16> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %266 = stablehlo.add %arg280, %265 : tensor<16x16xi32>
      %267 = stablehlo.select %264, %266, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %269 = "stablehlo.gather"(%arg279, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %269 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %180 = "mhlo.fusion"(%178, %179) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<256x8x1xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
      %264 = stablehlo.add %arg279, %263 : tensor<8x16x16xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %265 = stablehlo.reduce(%264 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
      %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
      %267 = stablehlo.subtract %264, %266 : tensor<8x16x16xf32>
      %268 = stablehlo.exponential %267 : tensor<8x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %269 = stablehlo.reduce(%268 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
      %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
      %271 = stablehlo.divide %268, %270 : tensor<8x16x16xf32>
      mhlo.return %271 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
    %181 = "mhlo.fusion"(%arg112, %arg111, %arg242, %173, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x16xf32>, %arg283: tensor<512xf32>):
      %263 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %264 = stablehlo.subtract %arg282, %263 : tensor<512x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<512xf32>
      %267 = stablehlo.rsqrt %266 : tensor<512xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<512x16xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<512x16xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %273 = stablehlo.add %271, %272 : tensor<512x16xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
      %275 = stablehlo.slice %274 [0:1, 0:8, 32:64, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x32x16xf32>
      mhlo.return %275 : tensor<1x8x32x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512xf32>) -> tensor<1x8x32x16xf32>
    %182 = mhlo.bitcast %181 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x32x16xf32>) -> tensor<8x32x16xf32>
    %183 = "mhlo.fusion"(%180, %182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<8x32x16xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<8x16x32xf32>
      %264 = stablehlo.transpose %263, dims = [1, 0, 2] : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
      mhlo.return %264 : tensor<16x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<16x8x32xf32>
    %184 = "mhlo.fusion"(%183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x8x32xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %266 = stablehlo.add %263, %265 : tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %268 = stablehlo.clamp %264, %266, %267 : tensor<1x16x256xf32>
      %269 = stablehlo.multiply %263, %268 : tensor<1x16x256xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %271 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x256xf32>
      mhlo.return %272 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x32xf32>) -> tensor<16x256xf32>
    %185 = stablehlo.custom_call @__cublas$gemm(%arg113, %184) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\2298304\22,\22rhs_stride\22:\224096\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[409600]{0})"} : (tensor<384x256xf32>, tensor<16x256xf32>) -> tuple<tensor<384x16xf32>, tensor<409600xi8>>
    %186 = stablehlo.get_tuple_element %185[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<409600xi8>>) -> tensor<384x16xf32>
    %187 = "mhlo.fusion"(%171, %arg115, %arg114, %arg243, %186, %arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384x16xf32>, %arg284: tensor<384xf32>):
      %263 = mhlo.bitcast %arg283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<16x384xf32>
      mhlo.return %275 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %188 = stablehlo.custom_call @__cublas$gemm(%arg116, %187) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[768,16]{0,1}, s8[1204224]{0})"} : (tensor<768x384xf32>, tensor<16x384xf32>) -> tuple<tensor<768x16xf32>, tensor<1204224xi8>>
    %189 = stablehlo.get_tuple_element %188[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,16]{0,1}"} : (tuple<tensor<768x16xf32>, tensor<1204224xi8>>) -> tensor<768x16xf32>
    %190 = "mhlo.fusion"(%arg118, %arg117, %arg245, %189, %arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<768x16xf32>, %arg283: tensor<768xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x768xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<768xf32>
      %268 = stablehlo.rsqrt %267 : tensor<768xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x768xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x768xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x768xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x16x768xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x16x768xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<16x768xf32>
      mhlo.return %284 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x16xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %191 = stablehlo.custom_call @__cublas$gemm(%arg119, %190) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\2212288\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[1228800]{0})"} : (tensor<384x768xf32>, tensor<16x768xf32>) -> tuple<tensor<384x16xf32>, tensor<1228800xi8>>
    %192 = stablehlo.get_tuple_element %191[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<1228800xi8>>) -> tensor<384x16xf32>
    %193 = "mhlo.fusion"(%187, %arg121, %arg120, %arg247, %192, %arg248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384x16xf32>, %arg284: tensor<384xf32>):
      %263 = mhlo.bitcast %arg283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<16x384xf32>
      mhlo.return %275 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %194 = stablehlo.custom_call @__cublas$gemm(%arg123, %193) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22196608\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[512,16]{1,0}, s8[811008]{0})"} : (tensor<512x384xf32>, tensor<16x384xf32>) -> tuple<tensor<512x16xf32>, tensor<811008xi8>>
    %195 = stablehlo.get_tuple_element %194[0] : (tuple<tensor<512x16xf32>, tensor<811008xi8>>) -> tensor<512x16xf32>
    %196:2 = "mhlo.fusion"(%arg125, %arg124, %arg251, %195, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x16xf32>, %arg283: tensor<512xf32>):
      %263 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %264 = stablehlo.subtract %arg282, %263 : tensor<512x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<512xf32>
      %267 = stablehlo.rsqrt %266 : tensor<512xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<512x16xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<512x16xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %273 = stablehlo.add %271, %272 : tensor<512x16xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
      %275 = stablehlo.slice %274 [0:1, 0:8, 16:32, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
      %276 = stablehlo.slice %274 [0:1, 0:8, 0:16, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
      mhlo.return %275, %276 : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>)
    %197 = stablehlo.tuple %196#0, %196#1 {xla_shape = "(f32[1,8,16,16]{3,2,1,0}, f32[1,8,16,16]{3,2,1,0})"} : tuple<tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>>
    %198 = mhlo.bitcast %196#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    %199 = mhlo.bitcast %196#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    %200 = "mhlo.fusion"(%198, %199) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<8x16x16xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<8x16x16xf32>
      mhlo.return %265 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %201 = "mhlo.fusion"(%arg122, %arg249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %264 = stablehlo.compare LT, %arg280, %263 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_0 = stablehlo.constant dense<16> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %266 = stablehlo.add %arg280, %265 : tensor<16x16xi32>
      %267 = stablehlo.select %264, %266, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %269 = "stablehlo.gather"(%arg279, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %269 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %202 = "mhlo.fusion"(%200, %201) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<256x8x1xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
      %264 = stablehlo.add %arg279, %263 : tensor<8x16x16xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %265 = stablehlo.reduce(%264 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
      %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
      %267 = stablehlo.subtract %264, %266 : tensor<8x16x16xf32>
      %268 = stablehlo.exponential %267 : tensor<8x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %269 = stablehlo.reduce(%268 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
      %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
      %271 = stablehlo.divide %268, %270 : tensor<8x16x16xf32>
      mhlo.return %271 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
    %203 = "mhlo.fusion"(%arg125, %arg124, %arg251, %195, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x16xf32>, %arg283: tensor<512xf32>):
      %263 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %264 = stablehlo.subtract %arg282, %263 : tensor<512x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<512xf32>
      %267 = stablehlo.rsqrt %266 : tensor<512xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<512x16xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<512x16xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %273 = stablehlo.add %271, %272 : tensor<512x16xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
      %275 = stablehlo.slice %274 [0:1, 0:8, 32:64, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x32x16xf32>
      mhlo.return %275 : tensor<1x8x32x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512xf32>) -> tensor<1x8x32x16xf32>
    %204 = mhlo.bitcast %203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x32x16xf32>) -> tensor<8x32x16xf32>
    %205 = "mhlo.fusion"(%202, %204) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<8x32x16xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<8x16x32xf32>
      %264 = stablehlo.transpose %263, dims = [1, 0, 2] : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
      mhlo.return %264 : tensor<16x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<16x8x32xf32>
    %206 = "mhlo.fusion"(%205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x8x32xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %266 = stablehlo.add %263, %265 : tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %268 = stablehlo.clamp %264, %266, %267 : tensor<1x16x256xf32>
      %269 = stablehlo.multiply %263, %268 : tensor<1x16x256xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %271 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x256xf32>
      mhlo.return %272 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x32xf32>) -> tensor<16x256xf32>
    %207 = stablehlo.custom_call @__cublas$gemm(%arg126, %206) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\2298304\22,\22rhs_stride\22:\224096\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[409600]{0})"} : (tensor<384x256xf32>, tensor<16x256xf32>) -> tuple<tensor<384x16xf32>, tensor<409600xi8>>
    %208 = stablehlo.get_tuple_element %207[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<409600xi8>>) -> tensor<384x16xf32>
    %209 = "mhlo.fusion"(%193, %arg128, %arg127, %arg252, %208, %arg253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384x16xf32>, %arg284: tensor<384xf32>):
      %263 = mhlo.bitcast %arg283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<16x384xf32>
      mhlo.return %275 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %210 = stablehlo.custom_call @__cublas$gemm(%arg129, %209) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[768,16]{0,1}, s8[1204224]{0})"} : (tensor<768x384xf32>, tensor<16x384xf32>) -> tuple<tensor<768x16xf32>, tensor<1204224xi8>>
    %211 = stablehlo.get_tuple_element %210[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,16]{0,1}"} : (tuple<tensor<768x16xf32>, tensor<1204224xi8>>) -> tensor<768x16xf32>
    %212 = "mhlo.fusion"(%arg131, %arg130, %arg254, %211, %arg255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<768x16xf32>, %arg283: tensor<768xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x768xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<768xf32>
      %268 = stablehlo.rsqrt %267 : tensor<768xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x768xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x768xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x768xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x16x768xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x16x768xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<16x768xf32>
      mhlo.return %284 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x16xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %213 = stablehlo.custom_call @__cublas$gemm(%arg132, %212) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\2212288\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[1228800]{0})"} : (tensor<384x768xf32>, tensor<16x768xf32>) -> tuple<tensor<384x16xf32>, tensor<1228800xi8>>
    %214 = stablehlo.get_tuple_element %213[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<1228800xi8>>) -> tensor<384x16xf32>
    %215 = "mhlo.fusion"(%209, %arg134, %arg133, %arg256, %214, %arg257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384x16xf32>, %arg284: tensor<384xf32>):
      %263 = mhlo.bitcast %arg283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<16x384xf32>
      mhlo.return %275 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %216 = stablehlo.custom_call @__cublas$gemm(%arg136, %215) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22196608\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[512,16]{1,0}, s8[811008]{0})"} : (tensor<512x384xf32>, tensor<16x384xf32>) -> tuple<tensor<512x16xf32>, tensor<811008xi8>>
    %217 = stablehlo.get_tuple_element %216[0] : (tuple<tensor<512x16xf32>, tensor<811008xi8>>) -> tensor<512x16xf32>
    %218:2 = "mhlo.fusion"(%arg138, %arg137, %arg260, %217, %arg259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x16xf32>, %arg283: tensor<512xf32>):
      %263 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %264 = stablehlo.subtract %arg282, %263 : tensor<512x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<512xf32>
      %267 = stablehlo.rsqrt %266 : tensor<512xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<512x16xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<512x16xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %273 = stablehlo.add %271, %272 : tensor<512x16xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
      %275 = stablehlo.slice %274 [0:1, 0:8, 16:32, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
      %276 = stablehlo.slice %274 [0:1, 0:8, 0:16, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
      mhlo.return %275, %276 : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>)
    %219 = stablehlo.tuple %218#0, %218#1 {xla_shape = "(f32[1,8,16,16]{3,2,1,0}, f32[1,8,16,16]{3,2,1,0})"} : tuple<tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>>
    %220 = mhlo.bitcast %218#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    %221 = mhlo.bitcast %218#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    %222 = "mhlo.fusion"(%220, %221) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<8x16x16xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<8x16x16xf32>
      mhlo.return %265 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %223 = "mhlo.fusion"(%arg135, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %264 = stablehlo.compare LT, %arg280, %263 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_0 = stablehlo.constant dense<16> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %266 = stablehlo.add %arg280, %265 : tensor<16x16xi32>
      %267 = stablehlo.select %264, %266, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %269 = "stablehlo.gather"(%arg279, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %269 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %224 = "mhlo.fusion"(%222, %223) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<256x8x1xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
      %264 = stablehlo.add %arg279, %263 : tensor<8x16x16xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %265 = stablehlo.reduce(%264 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
      %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
      %267 = stablehlo.subtract %264, %266 : tensor<8x16x16xf32>
      %268 = stablehlo.exponential %267 : tensor<8x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %269 = stablehlo.reduce(%268 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
      %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
      %271 = stablehlo.divide %268, %270 : tensor<8x16x16xf32>
      mhlo.return %271 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
    %225 = "mhlo.fusion"(%arg138, %arg137, %arg260, %217, %arg259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x16xf32>, %arg283: tensor<512xf32>):
      %263 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %264 = stablehlo.subtract %arg282, %263 : tensor<512x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<512xf32>
      %267 = stablehlo.rsqrt %266 : tensor<512xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<512x16xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<512x16xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %273 = stablehlo.add %271, %272 : tensor<512x16xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
      %275 = stablehlo.slice %274 [0:1, 0:8, 32:64, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x32x16xf32>
      mhlo.return %275 : tensor<1x8x32x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512xf32>) -> tensor<1x8x32x16xf32>
    %226 = mhlo.bitcast %225 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x32x16xf32>) -> tensor<8x32x16xf32>
    %227 = "mhlo.fusion"(%224, %226) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<8x32x16xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<8x16x32xf32>
      %264 = stablehlo.transpose %263, dims = [1, 0, 2] : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
      mhlo.return %264 : tensor<16x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<16x8x32xf32>
    %228 = "mhlo.fusion"(%227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x8x32xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %266 = stablehlo.add %263, %265 : tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %268 = stablehlo.clamp %264, %266, %267 : tensor<1x16x256xf32>
      %269 = stablehlo.multiply %263, %268 : tensor<1x16x256xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %271 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x256xf32>
      mhlo.return %272 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x32xf32>) -> tensor<16x256xf32>
    %229 = stablehlo.custom_call @__cublas$gemm(%arg139, %228) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\2298304\22,\22rhs_stride\22:\224096\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[409600]{0})"} : (tensor<384x256xf32>, tensor<16x256xf32>) -> tuple<tensor<384x16xf32>, tensor<409600xi8>>
    %230 = stablehlo.get_tuple_element %229[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<409600xi8>>) -> tensor<384x16xf32>
    %231 = "mhlo.fusion"(%215, %arg141, %arg140, %arg261, %230, %arg262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384x16xf32>, %arg284: tensor<384xf32>):
      %263 = mhlo.bitcast %arg283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<16x384xf32>
      mhlo.return %275 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %232 = stablehlo.custom_call @__cublas$gemm(%arg142, %231) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[768,16]{0,1}, s8[1204224]{0})"} : (tensor<768x384xf32>, tensor<16x384xf32>) -> tuple<tensor<768x16xf32>, tensor<1204224xi8>>
    %233 = stablehlo.get_tuple_element %232[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,16]{0,1}"} : (tuple<tensor<768x16xf32>, tensor<1204224xi8>>) -> tensor<768x16xf32>
    %234 = "mhlo.fusion"(%arg144, %arg143, %arg263, %233, %arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<768x16xf32>, %arg283: tensor<768xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x768xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<768xf32>
      %268 = stablehlo.rsqrt %267 : tensor<768xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x768xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x768xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x768xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x16x768xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x16x768xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<16x768xf32>
      mhlo.return %284 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x16xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %235 = stablehlo.custom_call @__cublas$gemm(%arg145, %234) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\2212288\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[1228800]{0})"} : (tensor<384x768xf32>, tensor<16x768xf32>) -> tuple<tensor<384x16xf32>, tensor<1228800xi8>>
    %236 = stablehlo.get_tuple_element %235[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<1228800xi8>>) -> tensor<384x16xf32>
    %237 = "mhlo.fusion"(%231, %arg147, %arg146, %arg265, %236, %arg266) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384x16xf32>, %arg284: tensor<384xf32>):
      %263 = mhlo.bitcast %arg283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<16x384xf32>
      mhlo.return %275 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %238 = stablehlo.custom_call @__cublas$gemm(%arg149, %237) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22196608\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[512,16]{1,0}, s8[811008]{0})"} : (tensor<512x384xf32>, tensor<16x384xf32>) -> tuple<tensor<512x16xf32>, tensor<811008xi8>>
    %239 = stablehlo.get_tuple_element %238[0] : (tuple<tensor<512x16xf32>, tensor<811008xi8>>) -> tensor<512x16xf32>
    %240:2 = "mhlo.fusion"(%arg151, %arg150, %arg269, %239, %arg268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x16xf32>, %arg283: tensor<512xf32>):
      %263 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %264 = stablehlo.subtract %arg282, %263 : tensor<512x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<512xf32>
      %267 = stablehlo.rsqrt %266 : tensor<512xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<512x16xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<512x16xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %273 = stablehlo.add %271, %272 : tensor<512x16xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
      %275 = stablehlo.slice %274 [0:1, 0:8, 16:32, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
      %276 = stablehlo.slice %274 [0:1, 0:8, 0:16, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x16x16xf32>
      mhlo.return %275, %276 : tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512xf32>) -> (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>)
    %241 = stablehlo.tuple %240#0, %240#1 {xla_shape = "(f32[1,8,16,16]{3,2,1,0}, f32[1,8,16,16]{3,2,1,0})"} : tuple<tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>>
    %242 = mhlo.bitcast %240#1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    %243 = mhlo.bitcast %240#0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    %244 = "mhlo.fusion"(%242, %243) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<8x16x16xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
      %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<8x16x16xf32>
      mhlo.return %265 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %245 = "mhlo.fusion"(%arg148, %arg267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %264 = stablehlo.compare LT, %arg280, %263 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_0 = stablehlo.constant dense<16> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %266 = stablehlo.add %arg280, %265 : tensor<16x16xi32>
      %267 = stablehlo.select %264, %266, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %269 = "stablehlo.gather"(%arg279, %268) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> {result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %269 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1, 2]> : tensor<3xindex>, xla_shape = "f32[256,8,1]{0,1,2}"} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %246 = "mhlo.fusion"(%244, %245) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<256x8x1xf32>):
      %263 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
      %264 = stablehlo.add %arg279, %263 : tensor<8x16x16xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %265 = stablehlo.reduce(%264 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
      %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
      %267 = stablehlo.subtract %264, %266 : tensor<8x16x16xf32>
      %268 = stablehlo.exponential %267 : tensor<8x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %269 = stablehlo.reduce(%268 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x16x16xf32>, tensor<f32>) -> tensor<8x16xf32>
      %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1] : (tensor<8x16xf32>) -> tensor<8x16x16xf32>
      %271 = stablehlo.divide %268, %270 : tensor<8x16x16xf32>
      mhlo.return %271 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<256x8x1xf32>) -> tensor<8x16x16xf32>
    %247 = "mhlo.fusion"(%arg151, %arg150, %arg269, %239, %arg268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<512x16xf32>, %arg283: tensor<512xf32>):
      %263 = stablehlo.broadcast_in_dim %arg283, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %264 = stablehlo.subtract %arg282, %263 : tensor<512x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %266 = stablehlo.add %arg281, %265 : tensor<512xf32>
      %267 = stablehlo.rsqrt %266 : tensor<512xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %269 = stablehlo.multiply %264, %268 : tensor<512x16xf32>
      %270 = stablehlo.broadcast_in_dim %arg280, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %271 = stablehlo.multiply %269, %270 : tensor<512x16xf32>
      %272 = stablehlo.broadcast_in_dim %arg279, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
      %273 = stablehlo.add %271, %272 : tensor<512x16xf32>
      %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16xf32>) -> tensor<1x8x64x16xf32>
      %275 = stablehlo.slice %274 [0:1, 0:8, 32:64, 0:16] : (tensor<1x8x64x16xf32>) -> tensor<1x8x32x16xf32>
      mhlo.return %275 : tensor<1x8x32x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512xf32>) -> tensor<1x8x32x16xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x32x16xf32>) -> tensor<8x32x16xf32>
    %249 = "mhlo.fusion"(%246, %248) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<8x16x16xf32>, %arg280: tensor<8x32x16xf32>):
      %263 = stablehlo.dot_general %arg279, %arg280, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<8x16x32xf32>
      %264 = stablehlo.transpose %263, dims = [1, 0, 2] : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
      mhlo.return %264 : tensor<16x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16xf32>, tensor<8x32x16xf32>) -> tensor<16x8x32xf32>
    %250 = "mhlo.fusion"(%249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x8x32xf32>):
      %263 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %265 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %266 = stablehlo.add %263, %265 : tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %268 = stablehlo.clamp %264, %266, %267 : tensor<1x16x256xf32>
      %269 = stablehlo.multiply %263, %268 : tensor<1x16x256xf32>
      %270 = mhlo.bitcast %269 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %271 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x256xf32>
      mhlo.return %272 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x32xf32>) -> tensor<16x256xf32>
    %251 = stablehlo.custom_call @__cublas$gemm(%arg152, %250) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\2298304\22,\22rhs_stride\22:\224096\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[409600]{0})"} : (tensor<384x256xf32>, tensor<16x256xf32>) -> tuple<tensor<384x16xf32>, tensor<409600xi8>>
    %252 = stablehlo.get_tuple_element %251[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<409600xi8>>) -> tensor<384x16xf32>
    %253 = "mhlo.fusion"(%237, %arg154, %arg153, %arg270, %252, %arg271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<384x16xf32>, %arg284: tensor<384xf32>):
      %263 = mhlo.bitcast %arg283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg284, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg279, %274 : tensor<16x384xf32>
      mhlo.return %275 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %254 = stablehlo.custom_call @__cublas$gemm(%arg155, %253) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\226144\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[768,16]{0,1}, s8[1204224]{0})"} : (tensor<768x384xf32>, tensor<16x384xf32>) -> tuple<tensor<768x16xf32>, tensor<1204224xi8>>
    %255 = stablehlo.get_tuple_element %254[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,16]{0,1}"} : (tuple<tensor<768x16xf32>, tensor<1204224xi8>>) -> tensor<768x16xf32>
    %256 = "mhlo.fusion"(%arg157, %arg156, %arg272, %255, %arg273) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<768x16xf32>, %arg283: tensor<768xf32>):
      %263 = mhlo.bitcast %arg282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<768x16xf32>) -> tensor<16x768xf32>
      %264 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x768xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %267 = stablehlo.add %arg283, %266 : tensor<768xf32>
      %268 = stablehlo.rsqrt %267 : tensor<768xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x768xf32>
      %271 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x768xf32>
      %273 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x768xf32>
      %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %278 = stablehlo.add %275, %277 : tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %280 = stablehlo.clamp %276, %278, %279 : tensor<1x16x768xf32>
      %281 = stablehlo.multiply %275, %280 : tensor<1x16x768xf32>
      %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      %cst_3 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
      %284 = stablehlo.multiply %282, %283 : tensor<16x768xf32>
      mhlo.return %284 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x16xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %257 = stablehlo.custom_call @__cublas$gemm(%arg158, %256) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22294912\22,\22rhs_stride\22:\2212288\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[384,16]{0,1}, s8[1228800]{0})"} : (tensor<384x768xf32>, tensor<16x768xf32>) -> tuple<tensor<384x16xf32>, tensor<1228800xi8>>
    %258 = stablehlo.get_tuple_element %257[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,16]{0,1}"} : (tuple<tensor<384x16xf32>, tensor<1228800xi8>>) -> tensor<384x16xf32>
    %259 = "mhlo.fusion"(%arg162, %arg161, %arg277, %arg276, %253, %arg160, %arg159, %arg275, %arg274, %258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>, %arg285: tensor<384xf32>, %arg286: tensor<384xf32>, %arg287: tensor<384xf32>, %arg288: tensor<384x16xf32>):
      %263 = mhlo.bitcast %arg288 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<384x16xf32>) -> tensor<16x384xf32>
      %264 = stablehlo.broadcast_in_dim %arg287, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %265 = stablehlo.subtract %263, %264 : tensor<16x384xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %267 = stablehlo.add %arg286, %266 : tensor<384xf32>
      %268 = stablehlo.rsqrt %267 : tensor<384xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %270 = stablehlo.multiply %265, %269 : tensor<16x384xf32>
      %271 = stablehlo.broadcast_in_dim %arg285, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %272 = stablehlo.multiply %270, %271 : tensor<16x384xf32>
      %273 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %274 = stablehlo.add %272, %273 : tensor<16x384xf32>
      %275 = stablehlo.add %arg283, %274 : tensor<16x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %276 = stablehlo.reduce(%275 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<16x384xf32>, tensor<f32>) -> tensor<384xf32>
      %cst_1 = stablehlo.constant dense<6.250000e-02> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %278 = stablehlo.multiply %276, %277 : tensor<384xf32>
      %279 = stablehlo.subtract %278, %arg282 : tensor<384xf32>
      %280 = stablehlo.add %arg281, %266 : tensor<384xf32>
      %281 = stablehlo.rsqrt %280 : tensor<384xf32>
      %282 = stablehlo.multiply %279, %281 : tensor<384xf32>
      %283 = stablehlo.multiply %282, %arg280 : tensor<384xf32>
      %284 = stablehlo.add %283, %arg279 : tensor<384xf32>
      mhlo.return %284 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x16xf32>) -> tensor<384xf32>
    %260 = "mhlo.fusion"(%arg163, %259) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg279: tensor<1000x384xf32>, %arg280: tensor<384xf32>):
      %263 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<1000x384xf32>
      %264 = stablehlo.multiply %263, %arg279 : tensor<1000x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %265 = stablehlo.reduce(%264 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1000x384xf32>, tensor<f32>) -> tensor<1000xf32>
      mhlo.return %265 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x384xf32>, tensor<384xf32>) -> tensor<1000xf32>
    %261 = "mhlo.fusion"(%arg164, %260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1000xf32>, %arg280: tensor<1000xf32>):
      %263 = stablehlo.add %arg279, %arg280 : tensor<1000xf32>
      mhlo.return %263 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
    %262 = mhlo.bitcast %261 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %262 : tensor<1x1000xf32>
  }
}
