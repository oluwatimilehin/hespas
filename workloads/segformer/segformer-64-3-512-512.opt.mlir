module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.68_computation(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>) -> tensor<1048576x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1048576x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<1048576x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    return %4 : tensor<1048576x32xf32>
  }
  func.func private @gemm_fusion_dot_computation(%arg0: tensor<64x32x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x64xf32>) -> tensor<16384x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<64x32x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<64x32x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x32x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<16384x32xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x64xf32>) -> tensor<16384x64xf32>
    return %6 : tensor<16384x64xf32>
  }
  func.func private @gemm_fusion_dot_general.71_computation(%arg0: tensor<32xf32>, %arg1: tensor<64x16384x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x256x32xf32>) -> tensor<64x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<64x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
    %3 = stablehlo.add %2, %arg3 : tensor<64x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x256x32xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.dot_general %1, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x32xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16384x256xf32>
    return %7 : tensor<64x16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.72_computation(%arg0: tensor<64x16384x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x256x32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<64x256x32xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x256xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x32xf32>
    return %2 : tensor<64x16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.73_computation(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<1048576x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    return %0 : tensor<1048576x32xf32>
  }
  func.func private @gemm_fusion_dot_general.74_computation(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x32xf32>) -> tensor<1048576x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1048576x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<1048576x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<128x32xf32>) -> tensor<1048576x128xf32>
    return %4 : tensor<1048576x128xf32>
  }
  func.func private @gemm_fusion_dot_general.75_computation(%arg0: tensor<64x128x16384xf32>, %arg1: tensor<32x128xf32>) -> tensor<1048576x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x128xf32>) -> tensor<1048576x128xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x128xf32>, tensor<32x128xf32>) -> tensor<1048576x32xf32>
    return %2 : tensor<1048576x32xf32>
  }
  func.func private @gemm_fusion_dot_general.76_computation(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>) -> tensor<1048576x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1048576x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<1048576x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    return %4 : tensor<1048576x32xf32>
  }
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<64x32x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x64xf32>) -> tensor<16384x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<64x32x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<64x32x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x32x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<16384x32xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x64xf32>) -> tensor<16384x64xf32>
    return %6 : tensor<16384x64xf32>
  }
  func.func private @gemm_fusion_dot_general.79_computation(%arg0: tensor<32xf32>, %arg1: tensor<64x16384x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x256x32xf32>) -> tensor<64x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<64x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
    %3 = stablehlo.add %2, %arg3 : tensor<64x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x256x32xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.dot_general %1, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x32xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16384x256xf32>
    return %7 : tensor<64x16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.80_computation(%arg0: tensor<64x16384x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x256x32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<64x256x32xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x256xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x32xf32>
    return %2 : tensor<64x16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.81_computation(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<1048576x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    return %0 : tensor<1048576x32xf32>
  }
  func.func private @gemm_fusion_dot_general.82_computation(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x32xf32>) -> tensor<1048576x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1048576x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<1048576x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<128x32xf32>) -> tensor<1048576x128xf32>
    return %4 : tensor<1048576x128xf32>
  }
  func.func private @gemm_fusion_dot_general.83_computation(%arg0: tensor<64x128x16384xf32>, %arg1: tensor<32x128xf32>) -> tensor<1048576x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x128xf32>) -> tensor<1048576x128xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x128xf32>, tensor<32x128xf32>) -> tensor<1048576x32xf32>
    return %2 : tensor<1048576x32xf32>
  }
  func.func private @gemm_fusion_dot_general.84_computation(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>) -> tensor<262144x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<262144x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<262144x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    return %4 : tensor<262144x64xf32>
  }
  func.func private @gemm_fusion_dot.2_computation(%arg0: tensor<64x64x256xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<16384x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x64x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x64xf32>) -> tensor<16384x64xf32>
    %6 = stablehlo.concatenate %arg3, %arg4, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %8 = stablehlo.dot %5, %7, precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x128xf32>) -> tensor<16384x128xf32>
    return %8 : tensor<16384x128xf32>
  }
  func.func private @gemm_fusion_dot_general.87_computation(%arg0: tensor<128x4096x32xf32>, %arg1: tensor<128x32x256xf32>) -> tensor<128x4096x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x32xf32>, tensor<128x32x256xf32>) -> tensor<128x4096x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<128x4096x256xf32>
    return %2 : tensor<128x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.88_computation(%arg0: tensor<128x4096x256xf32>, %arg1: tensor<128x256x32xf32>) -> tensor<128x4096x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x256xf32>, tensor<128x256x32xf32>) -> tensor<128x4096x32xf32>
    return %0 : tensor<128x4096x32xf32>
  }
  func.func private @gemm_fusion_dot_general.89_computation(%arg0: tensor<262144x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<262144x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    return %0 : tensor<262144x64xf32>
  }
  func.func private @gemm_fusion_dot_general.90_computation(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<262144x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<262144x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<262144x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<256x64xf32>) -> tensor<262144x256xf32>
    return %4 : tensor<262144x256xf32>
  }
  func.func private @gemm_fusion_dot_general.91_computation(%arg0: tensor<64x256x4096xf32>, %arg1: tensor<64x256xf32>) -> tensor<262144x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x256xf32>) -> tensor<262144x256xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x256xf32>, tensor<64x256xf32>) -> tensor<262144x64xf32>
    return %2 : tensor<262144x64xf32>
  }
  func.func private @gemm_fusion_dot_general.92_computation(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>) -> tensor<262144x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<262144x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<262144x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    return %4 : tensor<262144x64xf32>
  }
  func.func private @gemm_fusion_dot.3_computation(%arg0: tensor<64x64x256xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<16384x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x64x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x64xf32>) -> tensor<16384x64xf32>
    %6 = stablehlo.concatenate %arg3, %arg4, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %8 = stablehlo.dot %5, %7, precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x128xf32>) -> tensor<16384x128xf32>
    return %8 : tensor<16384x128xf32>
  }
  func.func private @gemm_fusion_dot_general.95_computation(%arg0: tensor<128x4096x32xf32>, %arg1: tensor<128x32x256xf32>) -> tensor<128x4096x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x32xf32>, tensor<128x32x256xf32>) -> tensor<128x4096x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<128x4096x256xf32>
    return %2 : tensor<128x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.96_computation(%arg0: tensor<128x4096x256xf32>, %arg1: tensor<128x256x32xf32>) -> tensor<128x4096x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x256xf32>, tensor<128x256x32xf32>) -> tensor<128x4096x32xf32>
    return %0 : tensor<128x4096x32xf32>
  }
  func.func private @gemm_fusion_dot_general.97_computation(%arg0: tensor<262144x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<262144x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    return %0 : tensor<262144x64xf32>
  }
  func.func private @gemm_fusion_dot_general.98_computation(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<262144x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<262144x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<262144x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<256x64xf32>) -> tensor<262144x256xf32>
    return %4 : tensor<262144x256xf32>
  }
  func.func private @gemm_fusion_dot_general.99_computation(%arg0: tensor<64x256x4096xf32>, %arg1: tensor<64x256xf32>) -> tensor<262144x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x256xf32>) -> tensor<262144x256xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x256xf32>, tensor<64x256xf32>) -> tensor<262144x64xf32>
    return %2 : tensor<262144x64xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    return %4 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot.4_computation(%arg0: tensor<64x160x256xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x320xf32>) -> tensor<16384x320xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<64x160x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x160x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<64x160x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x160x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x160xf32>) -> tensor<16384x160xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x320xf32>) -> tensor<16384x320xf32>
    return %6 : tensor<16384x320xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<320x1024x32xf32>, %arg1: tensor<320x32x256xf32>) -> tensor<320x1024x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x32xf32>, tensor<320x32x256xf32>) -> tensor<320x1024x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<320x1024x256xf32>
    return %2 : tensor<320x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<320x1024x256xf32>, %arg1: tensor<320x256x32xf32>) -> tensor<320x1024x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x256xf32>, tensor<320x256x32xf32>) -> tensor<320x1024x32xf32>
    return %0 : tensor<320x1024x32xf32>
  }
  func.func private @gemm_fusion_dot_general.105_computation(%arg0: tensor<65536x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    return %0 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot_general.106_computation(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<640x160xf32>) -> tensor<65536x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<640x160xf32>) -> tensor<65536x640xf32>
    return %4 : tensor<65536x640xf32>
  }
  func.func private @gemm_fusion_dot_general.107_computation(%arg0: tensor<64x640x1024xf32>, %arg1: tensor<160x640xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x640xf32>) -> tensor<65536x640xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x640xf32>, tensor<160x640xf32>) -> tensor<65536x160xf32>
    return %2 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    return %4 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot.5_computation(%arg0: tensor<64x160x256xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x320xf32>) -> tensor<16384x320xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<64x160x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x160x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<64x160x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x160x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x160xf32>) -> tensor<16384x160xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x320xf32>) -> tensor<16384x320xf32>
    return %6 : tensor<16384x320xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<320x1024x32xf32>, %arg1: tensor<320x32x256xf32>) -> tensor<320x1024x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x32xf32>, tensor<320x32x256xf32>) -> tensor<320x1024x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<320x1024x256xf32>
    return %2 : tensor<320x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<320x1024x256xf32>, %arg1: tensor<320x256x32xf32>) -> tensor<320x1024x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x256xf32>, tensor<320x256x32xf32>) -> tensor<320x1024x32xf32>
    return %0 : tensor<320x1024x32xf32>
  }
  func.func private @gemm_fusion_dot_general.113_computation(%arg0: tensor<65536x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    return %0 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot_general.114_computation(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<640x160xf32>) -> tensor<65536x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<640x160xf32>) -> tensor<65536x640xf32>
    return %4 : tensor<65536x640xf32>
  }
  func.func private @gemm_fusion_dot_general.115_computation(%arg0: tensor<64x640x1024xf32>, %arg1: tensor<160x640xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x640xf32>) -> tensor<65536x640xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x640xf32>, tensor<160x640xf32>) -> tensor<65536x160xf32>
    return %2 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot.7_computation(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>, %arg4: tensor<256x256xf32>, %arg5: tensor<256x256xf32>) -> tensor<16384x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16384x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<16384x256xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<768x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x768xf32>) -> tensor<16384x768xf32>
    return %6 : tensor<16384x768xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<512x256x32xf32>, %arg1: tensor<512x32x256xf32>) -> tensor<512x256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x32xf32>, tensor<512x32x256xf32>) -> tensor<512x256x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x256x256xf32>
    return %2 : tensor<512x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<512x256x256xf32>, %arg1: tensor<512x256x32xf32>) -> tensor<512x256x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x256xf32>, tensor<512x256x32xf32>) -> tensor<512x256x32xf32>
    return %0 : tensor<512x256x32xf32>
  }
  func.func private @gemm_fusion_dot_general.121_computation(%arg0: tensor<16384x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<16384x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    return %0 : tensor<16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.122_computation(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256xf32>) -> tensor<16384x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16384x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<16384x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<1024x256xf32>) -> tensor<16384x1024xf32>
    return %4 : tensor<16384x1024xf32>
  }
  func.func private @gemm_fusion_dot_general.123_computation(%arg0: tensor<64x1024x256xf32>, %arg1: tensor<256x1024xf32>) -> tensor<16384x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<16384x1024xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x1024xf32>, tensor<256x1024xf32>) -> tensor<16384x256xf32>
    return %2 : tensor<16384x256xf32>
  }
  func.func private @gemm_fusion_dot.9_computation(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>, %arg4: tensor<256x256xf32>, %arg5: tensor<256x256xf32>) -> tensor<16384x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16384x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<16384x256xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<768x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x768xf32>) -> tensor<16384x768xf32>
    return %6 : tensor<16384x768xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<512x256x32xf32>, %arg1: tensor<512x32x256xf32>) -> tensor<512x256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x32xf32>, tensor<512x32x256xf32>) -> tensor<512x256x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x256x256xf32>
    return %2 : tensor<512x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<512x256x256xf32>, %arg1: tensor<512x256x32xf32>) -> tensor<512x256x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x256xf32>, tensor<512x256x32xf32>) -> tensor<512x256x32xf32>
    return %0 : tensor<512x256x32xf32>
  }
  func.func private @gemm_fusion_dot_general.129_computation(%arg0: tensor<16384x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<16384x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    return %0 : tensor<16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.130_computation(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256xf32>) -> tensor<16384x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16384x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<16384x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<1024x256xf32>) -> tensor<16384x1024xf32>
    return %4 : tensor<16384x1024xf32>
  }
  func.func private @gemm_fusion_dot_general.131_computation(%arg0: tensor<64x1024x256xf32>, %arg1: tensor<256x1024xf32>) -> tensor<16384x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<16384x1024xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x1024xf32>, tensor<256x1024xf32>) -> tensor<16384x256xf32>
    return %2 : tensor<16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<64x256x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<64x256x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x256x256xf32>, tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    return %6 : tensor<64x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<256x160xf32>) -> tensor<64x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x160xf32>) -> tensor<160x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<64x160x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1024x160xf32>, tensor<64x160x256xf32>) -> tensor<64x1024x256xf32>
    return %6 : tensor<64x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<64x4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x4096x64xf32>, tensor<64x64x256xf32>) -> tensor<64x4096x256xf32>
    return %6 : tensor<64x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<256x32xf32>) -> tensor<64x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x32xf32>) -> tensor<32x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<64x32x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x32xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
    return %6 : tensor<64x16384x256xf32>
  }
  func.func private @region_0.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.60(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x256xf32>, %arg3: tensor<16384x256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<16384x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<64x256x256xf32>
    %7 = stablehlo.add %2, %6 : tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<64x256x256xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<64x256x256xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %17 = stablehlo.add %15, %16 : tensor<64x256xf32>
    %18 = stablehlo.rsqrt %17 : tensor<64x256xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<64x256x256xf32>
    return %20 : tensor<64x256x256xf32>
  }
  func.func private @region_0.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.62(%arg0: tensor<64x256x256xf32>, %arg1: tensor<16384x256xf32>, %arg2: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x256x256xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x256x256xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x256xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x256x256xf32>
    return %16 : tensor<64x256x256xf32>
  }
  func.func private @region_0.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.64(%arg0: tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<512x256x256xf32>, tensor<f32>) -> tensor<512x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x256xf32>) -> tensor<64x8x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<64x8x256x256xf32>
    %5 = stablehlo.exponential %4 : tensor<64x8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<64x8x256x256xf32>
    return %8 : tensor<64x8x256x256xf32>
  }
  func.func private @region_0.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.66(%arg0: tensor<64x256x256xf32>) -> tensor<64x256x256xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x256x256xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x256x256xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x256x256xf32>
    return %12 : tensor<64x256x256xf32>
  }
  func.func private @region_0.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.68(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x16x16xf32>, %arg3: tensor<16384x256xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<64x256x256xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x256x256xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256xf32>
    %17 = stablehlo.rsqrt %16 : tensor<64x256xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<64x256x256xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x256x256xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %23 = stablehlo.add %21, %22 : tensor<64x256x256xf32>
    %24 = stablehlo.add %2, %23 : tensor<64x256x256xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x256xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<64x256x256xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<64x256x256xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<64x256xf32>
    %32 = stablehlo.add %31, %15 : tensor<64x256xf32>
    %33 = stablehlo.rsqrt %32 : tensor<64x256xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<64x256x256xf32>
    return %35 : tensor<64x256x256xf32>
  }
  func.func private @region_0.1.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.70(%arg0: tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<512x256x256xf32>, tensor<f32>) -> tensor<512x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x256xf32>) -> tensor<64x8x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<64x8x256x256xf32>
    %5 = stablehlo.exponential %4 : tensor<64x8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<64x8x256x256xf32>
    return %8 : tensor<64x8x256x256xf32>
  }
  func.func private @region_0.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.72(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x16x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<64x256x256xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<64x256x256xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x256xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<64x256x256xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<64x256x256xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %20 = stablehlo.add %18, %19 : tensor<64x256x256xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<64x256xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<64x256x256xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<64x256x256xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<64x256xf32>
    %28 = stablehlo.add %27, %12 : tensor<64x256xf32>
    %29 = stablehlo.rsqrt %28 : tensor<64x256xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<64x256x256xf32>
    return %31 : tensor<64x256x256xf32>
  }
  func.func private @region_0.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.77(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<65536x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x1024x160xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x1024x160xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x1024xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x1024xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x1024x160xf32>
    return %16 : tensor<64x1024x160xf32>
  }
  func.func private @region_0.1.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.79(%arg0: tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<320x1024x256xf32>, tensor<f32>) -> tensor<320x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<320x1024xf32>) -> tensor<64x5x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<64x5x1024x256xf32>
    %5 = stablehlo.exponential %4 : tensor<64x5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<64x5x1024x256xf32>
    return %8 : tensor<64x5x1024x256xf32>
  }
  func.func private @region_0.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.81(%arg0: tensor<64x1024x160xf32>) -> tensor<64x1024x160xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x1024x160xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x1024x160xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x1024xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x1024x160xf32>
    return %12 : tensor<64x1024x160xf32>
  }
  func.func private @region_0.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.83(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x160x32x32xf32>, %arg3: tensor<65536x160xf32>, %arg4: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x32x32xf32>, tensor<f32>) -> tensor<64x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x32xf32>) -> tensor<64x1024xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<64x1024x160xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x1024x160xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x1024xf32>
    %17 = stablehlo.rsqrt %16 : tensor<64x1024xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<64x1024x160xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x1024x160xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %23 = stablehlo.add %21, %22 : tensor<64x1024x160xf32>
    %24 = stablehlo.add %2, %23 : tensor<64x1024x160xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x1024xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<64x1024x160xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<64x1024x160xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<64x1024xf32>
    %32 = stablehlo.add %31, %15 : tensor<64x1024xf32>
    %33 = stablehlo.rsqrt %32 : tensor<64x1024xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<64x1024x160xf32>
    return %35 : tensor<64x1024x160xf32>
  }
  func.func private @region_0.1.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.85(%arg0: tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<320x1024x256xf32>, tensor<f32>) -> tensor<320x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<320x1024xf32>) -> tensor<64x5x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<64x5x1024x256xf32>
    %5 = stablehlo.exponential %4 : tensor<64x5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<64x5x1024x256xf32>
    return %8 : tensor<64x5x1024x256xf32>
  }
  func.func private @region_0.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.87(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x160x32x32xf32>) -> tensor<64x1024x160xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x32x32xf32>, tensor<f32>) -> tensor<64x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x32x32xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x32x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x32xf32>) -> tensor<64x1024xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<64x1024x160xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<64x1024x160xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x1024xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x1024xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<64x1024x160xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<64x1024x160xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %20 = stablehlo.add %18, %19 : tensor<64x1024x160xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<64x1024xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<64x1024x160xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<64x1024x160xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<64x1024xf32>
    %28 = stablehlo.add %27, %12 : tensor<64x1024xf32>
    %29 = stablehlo.rsqrt %28 : tensor<64x1024xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<64x1024x160xf32>
    return %31 : tensor<64x1024x160xf32>
  }
  func.func private @region_0.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.92(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<262144x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x4096x64xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x4096x64xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x4096xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x4096xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x4096x64xf32>
    return %16 : tensor<64x4096x64xf32>
  }
  func.func private @region_0.1.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.94(%arg0: tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x4096x256xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x4096xf32>) -> tensor<64x2x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<64x2x4096x256xf32>
    %5 = stablehlo.exponential %4 : tensor<64x2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<64x2x4096x256xf32>
    return %8 : tensor<64x2x4096x256xf32>
  }
  func.func private @region_0.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.96(%arg0: tensor<64x4096x64xf32>) -> tensor<64x4096x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x4096x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x4096x64xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x4096xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x4096xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x4096x64xf32>
    return %12 : tensor<64x4096x64xf32>
  }
  func.func private @region_0.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.98(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64x64x64xf32>, %arg3: tensor<262144x64xf32>, %arg4: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x64x64xf32>, tensor<f32>) -> tensor<64x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x64xf32>) -> tensor<64x4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<64x4096x64xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x4096x64xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x4096xf32>
    %17 = stablehlo.rsqrt %16 : tensor<64x4096xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<64x4096x64xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x4096x64xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %23 = stablehlo.add %21, %22 : tensor<64x4096x64xf32>
    %24 = stablehlo.add %2, %23 : tensor<64x4096x64xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x4096xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<64x4096x64xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<64x4096x64xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<64x4096xf32>
    %32 = stablehlo.add %31, %15 : tensor<64x4096xf32>
    %33 = stablehlo.rsqrt %32 : tensor<64x4096xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<64x4096x64xf32>
    return %35 : tensor<64x4096x64xf32>
  }
  func.func private @region_0.1.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.100(%arg0: tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x4096x256xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x4096xf32>) -> tensor<64x2x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<64x2x4096x256xf32>
    %5 = stablehlo.exponential %4 : tensor<64x2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<64x2x4096x256xf32>
    return %8 : tensor<64x2x4096x256xf32>
  }
  func.func private @region_0.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.102(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64x64x64xf32>) -> tensor<64x4096x64xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x64x64xf32>, tensor<f32>) -> tensor<64x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x64x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x64xf32>) -> tensor<64x4096xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<64x4096x64xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<64x4096x64xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x4096xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x4096xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<64x4096x64xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<64x4096x64xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %20 = stablehlo.add %18, %19 : tensor<64x4096x64xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<64x4096xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<64x4096x64xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<64x4096x64xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<64x4096xf32>
    %28 = stablehlo.add %27, %12 : tensor<64x4096xf32>
    %29 = stablehlo.rsqrt %28 : tensor<64x4096xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<64x4096x64xf32>
    return %31 : tensor<64x4096x64xf32>
  }
  func.func private @region_0.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.107(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<1048576x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x16384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x16384x32xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x16384x32xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x16384xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x16384xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x16384x32xf32>
    return %16 : tensor<64x16384x32xf32>
  }
  func.func private @region_0.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.109(%arg0: tensor<64x16384x256xf32>) -> tensor<64x16384x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<64x16384x256xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<64x16384x256xf32>
    %3 = stablehlo.exponential %2 : tensor<64x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x256xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<64x16384x256xf32>
    return %6 : tensor<64x16384x256xf32>
  }
  func.func private @region_0.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.111(%arg0: tensor<64x16384x32xf32>) -> tensor<64x16384x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x16384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x16384x32xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x16384x32xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x16384xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x16384xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x16384x32xf32>
    return %12 : tensor<64x16384x32xf32>
  }
  func.func private @region_0.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.113(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x32x128x128xf32>, %arg3: tensor<1048576x32xf32>, %arg4: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x128x128xf32>, tensor<f32>) -> tensor<64x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x128xf32>) -> tensor<64x16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<64x16384x32xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x16384x32xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x16384xf32>
    %17 = stablehlo.rsqrt %16 : tensor<64x16384xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<64x16384x32xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x16384x32xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %23 = stablehlo.add %21, %22 : tensor<64x16384x32xf32>
    %24 = stablehlo.add %2, %23 : tensor<64x16384x32xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x16384xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<64x16384x32xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<64x16384x32xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<64x16384xf32>
    %32 = stablehlo.add %31, %15 : tensor<64x16384xf32>
    %33 = stablehlo.rsqrt %32 : tensor<64x16384xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<64x16384x32xf32>
    return %35 : tensor<64x16384x32xf32>
  }
  func.func private @region_0.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.115(%arg0: tensor<64x16384x256xf32>) -> tensor<64x16384x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<64x16384x256xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<64x16384x256xf32>
    %3 = stablehlo.exponential %2 : tensor<64x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x256xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<64x16384x256xf32>
    return %6 : tensor<64x16384x256xf32>
  }
  func.func private @region_0.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.117(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x32x128x128xf32>) -> tensor<64x16384x32xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x128x128xf32>, tensor<f32>) -> tensor<64x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x128x128xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x128xf32>) -> tensor<64x16384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<64x16384x32xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<64x16384x32xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x16384xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x16384xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<64x16384x32xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<64x16384x32xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %20 = stablehlo.add %18, %19 : tensor<64x16384x32xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<64x16384xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<64x16384x32xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<64x16384x32xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<64x16384xf32>
    %28 = stablehlo.add %27, %12 : tensor<64x16384xf32>
    %29 = stablehlo.rsqrt %28 : tensor<64x16384xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<64x16384x32xf32>
    return %31 : tensor<64x16384x32xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<512x256x32xf32>) -> tensor<64x256x8x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x32xf32>) -> tensor<64x8x256x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x8x256x32xf32>) -> tensor<64x256x8x32xf32>
    return %1 : tensor<64x256x8x32xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<512x256x32xf32>) -> tensor<64x256x8x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x32xf32>) -> tensor<64x8x256x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x8x256x32xf32>) -> tensor<64x256x8x32xf32>
    return %1 : tensor<64x256x8x32xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<320x1024x32xf32>) -> tensor<64x1024x5x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x32xf32>) -> tensor<64x5x1024x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x5x1024x32xf32>) -> tensor<64x1024x5x32xf32>
    return %1 : tensor<64x1024x5x32xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<320x1024x32xf32>) -> tensor<64x1024x5x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x32xf32>) -> tensor<64x5x1024x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x5x1024x32xf32>) -> tensor<64x1024x5x32xf32>
    return %1 : tensor<64x1024x5x32xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<128x4096x32xf32>) -> tensor<64x4096x2x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x32xf32>) -> tensor<64x2x4096x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x2x4096x32xf32>) -> tensor<64x4096x2x32xf32>
    return %1 : tensor<64x4096x2x32xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<128x4096x32xf32>) -> tensor<64x4096x2x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x32xf32>) -> tensor<64x2x4096x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x2x4096x32xf32>) -> tensor<64x4096x2x32xf32>
    return %1 : tensor<64x4096x2x32xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<64x1024x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<64x1024x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x16x16xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x1024x16x16xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x1024x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
    %5 = stablehlo.negate %2 : tensor<64x1024x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x1024x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
    %9 = stablehlo.abs %8 : tensor<64x1024x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<64x1024x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x1024x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x1024x256xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<64x1024x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x1024x256xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<64x1024x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %22 = stablehlo.add %20, %21 : tensor<64x1024x256xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<64x1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x1024x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x1024x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %28 = stablehlo.add %26, %27 : tensor<64x1024x256xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<64x1024x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<64x1024x256xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<64x1024x256xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<64x1024x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %37 = stablehlo.negate %13 : tensor<64x1024x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
    %40 = stablehlo.exponential %37 : tensor<64x1024x256xf32>
    %41 = stablehlo.divide %10, %9 : tensor<64x1024x256xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<64x1024x256xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %45 = stablehlo.divide %10, %13 : tensor<64x1024x256xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<64x1024x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %48 = stablehlo.add %46, %47 : tensor<64x1024x256xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<64x1024x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %51 = stablehlo.add %49, %50 : tensor<64x1024x256xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<64x1024x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x1024x256xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<64x1024x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %57 = stablehlo.add %55, %56 : tensor<64x1024x256xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<64x1024x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x1024x256xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<64x1024x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x1024x256xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<64x1024x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %66 = stablehlo.add %64, %65 : tensor<64x1024x256xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<64x1024x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %69 = stablehlo.add %67, %68 : tensor<64x1024x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<64x1024x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %73 = stablehlo.add %71, %72 : tensor<64x1024x256xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<64x1024x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %76 = stablehlo.add %74, %75 : tensor<64x1024x256xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<64x1024x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %79 = stablehlo.add %77, %78 : tensor<64x1024x256xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<64x1024x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %82 = stablehlo.add %80, %81 : tensor<64x1024x256xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<64x1024x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %85 = stablehlo.add %83, %84 : tensor<64x1024x256xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<64x1024x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %88 = stablehlo.add %86, %87 : tensor<64x1024x256xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<64x1024x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %91 = stablehlo.add %89, %90 : tensor<64x1024x256xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<64x1024x256xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<64x1024x256xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<64x1024x256xf32>
    return %98 : tensor<64x1024x256xf32>
  }
  func.func private @fused_transpose.6(%arg0: tensor<16384x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<64x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<16384x1024xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x1024xf32>) -> tensor<64x256x1024xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<64x256x1024xf32>) -> tensor<64x1024x256xf32>
    return %3 : tensor<64x1024x256xf32>
  }
  func.func private @fused_transpose.7(%arg0: tensor<16384x768xf32>, %arg1: tensor<256xf32>) -> tensor<64x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 512:768] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
    return %4 : tensor<64x8x256x32xf32>
  }
  func.func private @fused_transpose.8(%arg0: tensor<16384x768xf32>, %arg1: tensor<256xf32>) -> tensor<64x8x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 256:512] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<64x256x8x32xf32>) -> tensor<64x8x32x256xf32>
    return %4 : tensor<64x8x32x256xf32>
  }
  func.func private @fused_transpose.9(%arg0: tensor<16384x768xf32>, %arg1: tensor<256xf32>) -> tensor<64x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 0:256] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
    return %4 : tensor<64x8x256x32xf32>
  }
  func.func private @fused_computation.69(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<64x256x16x16xf32>, %arg5: tensor<16384x256xf32>, %arg6: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<16384x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x16x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<64x256x256xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<64x256x256xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x256xf32>
    %20 = stablehlo.rsqrt %19 : tensor<64x256xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<64x256x256xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<64x256x256xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %26 = stablehlo.add %24, %25 : tensor<64x256x256xf32>
    %27 = stablehlo.add %5, %26 : tensor<64x256x256xf32>
    %28 = stablehlo.add %2, %27 : tensor<64x256x256xf32>
    return %28 : tensor<64x256x256xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<64x1024x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<64x1024x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x16x16xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x1024x16x16xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x1024x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
    %5 = stablehlo.negate %2 : tensor<64x1024x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x1024x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
    %9 = stablehlo.abs %8 : tensor<64x1024x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<64x1024x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x1024x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x1024x256xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<64x1024x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x1024x256xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<64x1024x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %22 = stablehlo.add %20, %21 : tensor<64x1024x256xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<64x1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x1024x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x1024x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %28 = stablehlo.add %26, %27 : tensor<64x1024x256xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<64x1024x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<64x1024x256xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<64x1024x256xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<64x1024x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %37 = stablehlo.negate %13 : tensor<64x1024x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
    %40 = stablehlo.exponential %37 : tensor<64x1024x256xf32>
    %41 = stablehlo.divide %10, %9 : tensor<64x1024x256xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<64x1024x256xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %45 = stablehlo.divide %10, %13 : tensor<64x1024x256xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<64x1024x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %48 = stablehlo.add %46, %47 : tensor<64x1024x256xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<64x1024x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %51 = stablehlo.add %49, %50 : tensor<64x1024x256xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<64x1024x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x1024x256xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<64x1024x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %57 = stablehlo.add %55, %56 : tensor<64x1024x256xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<64x1024x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x1024x256xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<64x1024x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x1024x256xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<64x1024x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %66 = stablehlo.add %64, %65 : tensor<64x1024x256xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<64x1024x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %69 = stablehlo.add %67, %68 : tensor<64x1024x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<64x1024x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %73 = stablehlo.add %71, %72 : tensor<64x1024x256xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<64x1024x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %76 = stablehlo.add %74, %75 : tensor<64x1024x256xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<64x1024x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %79 = stablehlo.add %77, %78 : tensor<64x1024x256xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<64x1024x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %82 = stablehlo.add %80, %81 : tensor<64x1024x256xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<64x1024x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %85 = stablehlo.add %83, %84 : tensor<64x1024x256xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<64x1024x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %88 = stablehlo.add %86, %87 : tensor<64x1024x256xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<64x1024x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
    %91 = stablehlo.add %89, %90 : tensor<64x1024x256xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<64x1024x256xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<64x1024x256xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<64x1024x256xf32>
    return %98 : tensor<64x1024x256xf32>
  }
  func.func private @fused_transpose.10(%arg0: tensor<16384x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<64x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<16384x1024xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x1024xf32>) -> tensor<64x256x1024xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<64x256x1024xf32>) -> tensor<64x1024x256xf32>
    return %3 : tensor<64x1024x256xf32>
  }
  func.func private @fused_transpose.11(%arg0: tensor<16384x768xf32>, %arg1: tensor<256xf32>) -> tensor<64x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 512:768] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
    return %4 : tensor<64x8x256x32xf32>
  }
  func.func private @fused_transpose.12(%arg0: tensor<16384x768xf32>, %arg1: tensor<256xf32>) -> tensor<64x8x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 256:512] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<64x256x8x32xf32>) -> tensor<64x8x32x256xf32>
    return %4 : tensor<64x8x32x256xf32>
  }
  func.func private @fused_transpose.13(%arg0: tensor<16384x768xf32>, %arg1: tensor<256xf32>) -> tensor<64x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 0:256] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
    return %4 : tensor<64x8x256x32xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<64x640x32x32xf32>, %arg1: tensor<640xf32>) -> tensor<64x640x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x640x32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<64x640x32x32xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x640x32x32xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x640x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
    %5 = stablehlo.negate %2 : tensor<64x640x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x640x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x640x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
    %9 = stablehlo.abs %8 : tensor<64x640x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<64x640x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x640x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x640x1024xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<64x640x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x640x1024xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<64x640x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<64x640x1024xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<64x640x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x640x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x640x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<64x640x1024xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<64x640x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<64x640x1024xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<64x640x1024xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<64x640x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %37 = stablehlo.negate %13 : tensor<64x640x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
    %40 = stablehlo.exponential %37 : tensor<64x640x1024xf32>
    %41 = stablehlo.divide %10, %9 : tensor<64x640x1024xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<64x640x1024xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %45 = stablehlo.divide %10, %13 : tensor<64x640x1024xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<64x640x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %48 = stablehlo.add %46, %47 : tensor<64x640x1024xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<64x640x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %51 = stablehlo.add %49, %50 : tensor<64x640x1024xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<64x640x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x640x1024xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<64x640x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %57 = stablehlo.add %55, %56 : tensor<64x640x1024xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<64x640x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x640x1024xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<64x640x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x640x1024xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<64x640x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %66 = stablehlo.add %64, %65 : tensor<64x640x1024xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<64x640x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %69 = stablehlo.add %67, %68 : tensor<64x640x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<64x640x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %73 = stablehlo.add %71, %72 : tensor<64x640x1024xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<64x640x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<64x640x1024xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<64x640x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %79 = stablehlo.add %77, %78 : tensor<64x640x1024xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<64x640x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %82 = stablehlo.add %80, %81 : tensor<64x640x1024xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<64x640x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %85 = stablehlo.add %83, %84 : tensor<64x640x1024xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<64x640x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %88 = stablehlo.add %86, %87 : tensor<64x640x1024xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<64x640x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %91 = stablehlo.add %89, %90 : tensor<64x640x1024xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<64x640x1024xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<64x640x1024xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<64x640x1024xf32>
    return %98 : tensor<64x640x1024xf32>
  }
  func.func private @fused_transpose.14(%arg0: tensor<65536x640xf32>, %arg1: tensor<640xf32>) -> tensor<64x640x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<65536x640xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x640xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x640xf32>) -> tensor<64x1024x640xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<64x1024x640xf32>) -> tensor<64x640x1024xf32>
    return %3 : tensor<64x640x1024xf32>
  }
  func.func private @fused_transpose.15(%arg0: tensor<16384x320xf32>, %arg1: tensor<160xf32>) -> tensor<64x5x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 160:320] : (tensor<16384x320xf32>) -> tensor<16384x160xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x160xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x256x5x32xf32>) -> tensor<64x5x256x32xf32>
    return %4 : tensor<64x5x256x32xf32>
  }
  func.func private @fused_transpose.16(%arg0: tensor<16384x320xf32>, %arg1: tensor<160xf32>) -> tensor<64x5x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 0:160] : (tensor<16384x320xf32>) -> tensor<16384x160xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x160xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<64x256x5x32xf32>) -> tensor<64x5x32x256xf32>
    return %4 : tensor<64x5x32x256xf32>
  }
  func.func private @fused_transpose.17(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>) -> tensor<64x5x1024x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<64x1024x5x32xf32>) -> tensor<64x5x1024x32xf32>
    return %3 : tensor<64x5x1024x32xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160xf32>, %arg4: tensor<64x160x32x32xf32>, %arg5: tensor<65536x160xf32>, %arg6: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<65536x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x32x32xf32>, tensor<f32>) -> tensor<64x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x32x32xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x32x32xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x32xf32>) -> tensor<64x1024xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<64x1024x160xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<64x1024x160xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<64x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x1024xf32>
    %20 = stablehlo.rsqrt %19 : tensor<64x1024xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<64x1024x160xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<64x1024x160xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %26 = stablehlo.add %24, %25 : tensor<64x1024x160xf32>
    %27 = stablehlo.add %5, %26 : tensor<64x1024x160xf32>
    %28 = stablehlo.add %2, %27 : tensor<64x1024x160xf32>
    return %28 : tensor<64x1024x160xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<64x640x32x32xf32>, %arg1: tensor<640xf32>) -> tensor<64x640x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x640x32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<64x640x32x32xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x640x32x32xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x640x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
    %5 = stablehlo.negate %2 : tensor<64x640x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x640x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x640x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
    %9 = stablehlo.abs %8 : tensor<64x640x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<64x640x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x640x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x640x1024xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<64x640x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x640x1024xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<64x640x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<64x640x1024xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<64x640x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x640x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x640x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<64x640x1024xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<64x640x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<64x640x1024xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<64x640x1024xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<64x640x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %37 = stablehlo.negate %13 : tensor<64x640x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
    %40 = stablehlo.exponential %37 : tensor<64x640x1024xf32>
    %41 = stablehlo.divide %10, %9 : tensor<64x640x1024xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<64x640x1024xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %45 = stablehlo.divide %10, %13 : tensor<64x640x1024xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<64x640x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %48 = stablehlo.add %46, %47 : tensor<64x640x1024xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<64x640x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %51 = stablehlo.add %49, %50 : tensor<64x640x1024xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<64x640x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x640x1024xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<64x640x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %57 = stablehlo.add %55, %56 : tensor<64x640x1024xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<64x640x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x640x1024xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<64x640x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x640x1024xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<64x640x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %66 = stablehlo.add %64, %65 : tensor<64x640x1024xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<64x640x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %69 = stablehlo.add %67, %68 : tensor<64x640x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<64x640x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %73 = stablehlo.add %71, %72 : tensor<64x640x1024xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<64x640x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<64x640x1024xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<64x640x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %79 = stablehlo.add %77, %78 : tensor<64x640x1024xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<64x640x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %82 = stablehlo.add %80, %81 : tensor<64x640x1024xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<64x640x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %85 = stablehlo.add %83, %84 : tensor<64x640x1024xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<64x640x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %88 = stablehlo.add %86, %87 : tensor<64x640x1024xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<64x640x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
    %91 = stablehlo.add %89, %90 : tensor<64x640x1024xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<64x640x1024xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<64x640x1024xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<64x640x1024xf32>
    return %98 : tensor<64x640x1024xf32>
  }
  func.func private @fused_transpose.18(%arg0: tensor<65536x640xf32>, %arg1: tensor<640xf32>) -> tensor<64x640x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<65536x640xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x640xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x640xf32>) -> tensor<64x1024x640xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<64x1024x640xf32>) -> tensor<64x640x1024xf32>
    return %3 : tensor<64x640x1024xf32>
  }
  func.func private @fused_transpose.19(%arg0: tensor<16384x320xf32>, %arg1: tensor<160xf32>) -> tensor<64x5x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 160:320] : (tensor<16384x320xf32>) -> tensor<16384x160xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x160xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x256x5x32xf32>) -> tensor<64x5x256x32xf32>
    return %4 : tensor<64x5x256x32xf32>
  }
  func.func private @fused_transpose.20(%arg0: tensor<16384x320xf32>, %arg1: tensor<160xf32>) -> tensor<64x5x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 0:160] : (tensor<16384x320xf32>) -> tensor<16384x160xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x160xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<64x256x5x32xf32>) -> tensor<64x5x32x256xf32>
    return %4 : tensor<64x5x32x256xf32>
  }
  func.func private @fused_transpose.21(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>) -> tensor<64x5x1024x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<64x1024x5x32xf32>) -> tensor<64x5x1024x32xf32>
    return %3 : tensor<64x5x1024x32xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<64x256x64x64xf32>, %arg1: tensor<256xf32>) -> tensor<64x256x4096xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x256x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x256x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
    %5 = stablehlo.negate %2 : tensor<64x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
    %9 = stablehlo.abs %8 : tensor<64x256x4096xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<64x256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x4096xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x4096xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<64x256x4096xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x256x4096xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<64x256x4096xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<64x256x4096xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<64x256x4096xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x256x4096xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %28 = stablehlo.add %26, %27 : tensor<64x256x4096xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<64x256x4096xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %31 = stablehlo.add %29, %30 : tensor<64x256x4096xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<64x256x4096xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<64x256x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %37 = stablehlo.negate %13 : tensor<64x256x4096xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
    %40 = stablehlo.exponential %37 : tensor<64x256x4096xf32>
    %41 = stablehlo.divide %10, %9 : tensor<64x256x4096xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<64x256x4096xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %45 = stablehlo.divide %10, %13 : tensor<64x256x4096xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<64x256x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %48 = stablehlo.add %46, %47 : tensor<64x256x4096xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<64x256x4096xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %51 = stablehlo.add %49, %50 : tensor<64x256x4096xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<64x256x4096xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x256x4096xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<64x256x4096xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %57 = stablehlo.add %55, %56 : tensor<64x256x4096xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<64x256x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x256x4096xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<64x256x4096xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x256x4096xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<64x256x4096xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %66 = stablehlo.add %64, %65 : tensor<64x256x4096xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<64x256x4096xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %69 = stablehlo.add %67, %68 : tensor<64x256x4096xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<64x256x4096xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %73 = stablehlo.add %71, %72 : tensor<64x256x4096xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<64x256x4096xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %76 = stablehlo.add %74, %75 : tensor<64x256x4096xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<64x256x4096xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %79 = stablehlo.add %77, %78 : tensor<64x256x4096xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<64x256x4096xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %82 = stablehlo.add %80, %81 : tensor<64x256x4096xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<64x256x4096xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %85 = stablehlo.add %83, %84 : tensor<64x256x4096xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<64x256x4096xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %88 = stablehlo.add %86, %87 : tensor<64x256x4096xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<64x256x4096xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %91 = stablehlo.add %89, %90 : tensor<64x256x4096xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<64x256x4096xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<64x256x4096xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<64x256x4096xf32>
    return %98 : tensor<64x256x4096xf32>
  }
  func.func private @fused_transpose.22(%arg0: tensor<262144x256xf32>, %arg1: tensor<256xf32>) -> tensor<64x256x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<262144x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<64x4096x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
    return %3 : tensor<64x256x4096xf32>
  }
  func.func private @fused_transpose.23(%arg0: tensor<16384x128xf32>, %arg1: tensor<64xf32>) -> tensor<64x2x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 64:128] : (tensor<16384x128xf32>) -> tensor<16384x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x256x2x32xf32>) -> tensor<64x2x256x32xf32>
    return %4 : tensor<64x2x256x32xf32>
  }
  func.func private @fused_transpose.24(%arg0: tensor<16384x128xf32>, %arg1: tensor<64xf32>) -> tensor<64x2x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 0:64] : (tensor<16384x128xf32>) -> tensor<16384x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<64x256x2x32xf32>) -> tensor<64x2x32x256xf32>
    return %4 : tensor<64x2x32x256xf32>
  }
  func.func private @fused_transpose.25(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x2x4096x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<64x4096x2x32xf32>) -> tensor<64x2x4096x32xf32>
    return %3 : tensor<64x2x4096x32xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64x64x64x64xf32>, %arg5: tensor<262144x64xf32>, %arg6: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<262144x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x64x64xf32>, tensor<f32>) -> tensor<64x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x64x64xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x64xf32>) -> tensor<64x4096xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<64x4096x64xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<64x4096x64xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<64x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x4096xf32>
    %20 = stablehlo.rsqrt %19 : tensor<64x4096xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<64x4096x64xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<64x4096x64xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %26 = stablehlo.add %24, %25 : tensor<64x4096x64xf32>
    %27 = stablehlo.add %5, %26 : tensor<64x4096x64xf32>
    %28 = stablehlo.add %2, %27 : tensor<64x4096x64xf32>
    return %28 : tensor<64x4096x64xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<64x256x64x64xf32>, %arg1: tensor<256xf32>) -> tensor<64x256x4096xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x256x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x256x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
    %5 = stablehlo.negate %2 : tensor<64x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
    %9 = stablehlo.abs %8 : tensor<64x256x4096xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<64x256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x4096xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x4096xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<64x256x4096xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x256x4096xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<64x256x4096xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<64x256x4096xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<64x256x4096xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x256x4096xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %28 = stablehlo.add %26, %27 : tensor<64x256x4096xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<64x256x4096xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %31 = stablehlo.add %29, %30 : tensor<64x256x4096xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<64x256x4096xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<64x256x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %37 = stablehlo.negate %13 : tensor<64x256x4096xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
    %40 = stablehlo.exponential %37 : tensor<64x256x4096xf32>
    %41 = stablehlo.divide %10, %9 : tensor<64x256x4096xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<64x256x4096xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %45 = stablehlo.divide %10, %13 : tensor<64x256x4096xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<64x256x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %48 = stablehlo.add %46, %47 : tensor<64x256x4096xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<64x256x4096xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %51 = stablehlo.add %49, %50 : tensor<64x256x4096xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<64x256x4096xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x256x4096xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<64x256x4096xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %57 = stablehlo.add %55, %56 : tensor<64x256x4096xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<64x256x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x256x4096xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<64x256x4096xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x256x4096xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<64x256x4096xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %66 = stablehlo.add %64, %65 : tensor<64x256x4096xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<64x256x4096xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %69 = stablehlo.add %67, %68 : tensor<64x256x4096xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<64x256x4096xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %73 = stablehlo.add %71, %72 : tensor<64x256x4096xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<64x256x4096xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %76 = stablehlo.add %74, %75 : tensor<64x256x4096xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<64x256x4096xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %79 = stablehlo.add %77, %78 : tensor<64x256x4096xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<64x256x4096xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %82 = stablehlo.add %80, %81 : tensor<64x256x4096xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<64x256x4096xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %85 = stablehlo.add %83, %84 : tensor<64x256x4096xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<64x256x4096xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %88 = stablehlo.add %86, %87 : tensor<64x256x4096xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<64x256x4096xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
    %91 = stablehlo.add %89, %90 : tensor<64x256x4096xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<64x256x4096xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<64x256x4096xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<64x256x4096xf32>
    return %98 : tensor<64x256x4096xf32>
  }
  func.func private @fused_transpose.26(%arg0: tensor<262144x256xf32>, %arg1: tensor<256xf32>) -> tensor<64x256x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<262144x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<64x4096x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
    return %3 : tensor<64x256x4096xf32>
  }
  func.func private @fused_transpose.27(%arg0: tensor<16384x128xf32>, %arg1: tensor<64xf32>) -> tensor<64x2x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 64:128] : (tensor<16384x128xf32>) -> tensor<16384x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x256x2x32xf32>) -> tensor<64x2x256x32xf32>
    return %4 : tensor<64x2x256x32xf32>
  }
  func.func private @fused_transpose.28(%arg0: tensor<16384x128xf32>, %arg1: tensor<64xf32>) -> tensor<64x2x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
    %1 = stablehlo.slice %arg0 [0:16384, 0:64] : (tensor<16384x128xf32>) -> tensor<16384x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<16384x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<64x256x2x32xf32>) -> tensor<64x2x32x256xf32>
    return %4 : tensor<64x2x32x256xf32>
  }
  func.func private @fused_transpose.29(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x2x4096x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<64x4096x2x32xf32>) -> tensor<64x2x4096x32xf32>
    return %3 : tensor<64x2x4096x32xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<64x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<64x128x16384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x128x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
    %5 = stablehlo.negate %2 : tensor<64x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
    %9 = stablehlo.abs %8 : tensor<64x128x16384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<64x128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x128x16384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x128x16384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<64x128x16384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x128x16384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<64x128x16384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<64x128x16384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<64x128x16384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x128x16384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %28 = stablehlo.add %26, %27 : tensor<64x128x16384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<64x128x16384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %31 = stablehlo.add %29, %30 : tensor<64x128x16384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<64x128x16384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<64x128x16384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %37 = stablehlo.negate %13 : tensor<64x128x16384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
    %40 = stablehlo.exponential %37 : tensor<64x128x16384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<64x128x16384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<64x128x16384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<64x128x16384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<64x128x16384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %48 = stablehlo.add %46, %47 : tensor<64x128x16384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<64x128x16384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %51 = stablehlo.add %49, %50 : tensor<64x128x16384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<64x128x16384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x128x16384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<64x128x16384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %57 = stablehlo.add %55, %56 : tensor<64x128x16384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<64x128x16384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x128x16384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<64x128x16384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x128x16384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<64x128x16384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %66 = stablehlo.add %64, %65 : tensor<64x128x16384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<64x128x16384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %69 = stablehlo.add %67, %68 : tensor<64x128x16384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<64x128x16384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %73 = stablehlo.add %71, %72 : tensor<64x128x16384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<64x128x16384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %76 = stablehlo.add %74, %75 : tensor<64x128x16384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<64x128x16384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %79 = stablehlo.add %77, %78 : tensor<64x128x16384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<64x128x16384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %82 = stablehlo.add %80, %81 : tensor<64x128x16384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<64x128x16384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %85 = stablehlo.add %83, %84 : tensor<64x128x16384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<64x128x16384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %88 = stablehlo.add %86, %87 : tensor<64x128x16384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<64x128x16384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %91 = stablehlo.add %89, %90 : tensor<64x128x16384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<64x128x16384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<64x128x16384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<64x128x16384xf32>
    return %98 : tensor<64x128x16384xf32>
  }
  func.func private @fused_transpose.30(%arg0: tensor<1048576x128xf32>, %arg1: tensor<128xf32>) -> tensor<64x128x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1048576x128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x128xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x128xf32>) -> tensor<64x16384x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<64x16384x128xf32>) -> tensor<64x128x16384xf32>
    return %3 : tensor<64x128x16384xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<64x32x128x128xf32>, %arg5: tensor<1048576x32xf32>, %arg6: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<1048576x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x128x128xf32>, tensor<f32>) -> tensor<64x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x128x128xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x128xf32>) -> tensor<64x16384xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<64x16384x32xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<64x16384x32xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x16384xf32>
    %20 = stablehlo.rsqrt %19 : tensor<64x16384xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<64x16384x32xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<64x16384x32xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %26 = stablehlo.add %24, %25 : tensor<64x16384x32xf32>
    %27 = stablehlo.add %5, %26 : tensor<64x16384x32xf32>
    %28 = stablehlo.add %2, %27 : tensor<64x16384x32xf32>
    return %28 : tensor<64x16384x32xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<64x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<64x128x16384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x128x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
    %5 = stablehlo.negate %2 : tensor<64x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
    %9 = stablehlo.abs %8 : tensor<64x128x16384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<64x128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x128x16384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x128x16384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<64x128x16384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x128x16384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<64x128x16384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<64x128x16384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<64x128x16384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<64x128x16384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %28 = stablehlo.add %26, %27 : tensor<64x128x16384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<64x128x16384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %31 = stablehlo.add %29, %30 : tensor<64x128x16384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<64x128x16384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<64x128x16384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %37 = stablehlo.negate %13 : tensor<64x128x16384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
    %40 = stablehlo.exponential %37 : tensor<64x128x16384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<64x128x16384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<64x128x16384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<64x128x16384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<64x128x16384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %48 = stablehlo.add %46, %47 : tensor<64x128x16384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<64x128x16384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %51 = stablehlo.add %49, %50 : tensor<64x128x16384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<64x128x16384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %54 = stablehlo.add %52, %53 : tensor<64x128x16384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<64x128x16384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %57 = stablehlo.add %55, %56 : tensor<64x128x16384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<64x128x16384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %60 = stablehlo.add %58, %59 : tensor<64x128x16384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<64x128x16384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %63 = stablehlo.add %61, %62 : tensor<64x128x16384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<64x128x16384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %66 = stablehlo.add %64, %65 : tensor<64x128x16384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<64x128x16384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %69 = stablehlo.add %67, %68 : tensor<64x128x16384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<64x128x16384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %73 = stablehlo.add %71, %72 : tensor<64x128x16384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<64x128x16384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %76 = stablehlo.add %74, %75 : tensor<64x128x16384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<64x128x16384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %79 = stablehlo.add %77, %78 : tensor<64x128x16384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<64x128x16384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %82 = stablehlo.add %80, %81 : tensor<64x128x16384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<64x128x16384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %85 = stablehlo.add %83, %84 : tensor<64x128x16384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<64x128x16384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %88 = stablehlo.add %86, %87 : tensor<64x128x16384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<64x128x16384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
    %91 = stablehlo.add %89, %90 : tensor<64x128x16384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<64x128x16384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<64x128x16384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<64x128x16384xf32>
    return %98 : tensor<64x128x16384xf32>
  }
  func.func private @fused_transpose.31(%arg0: tensor<1048576x128xf32>, %arg1: tensor<128xf32>) -> tensor<64x128x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1048576x128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x128xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x128xf32>) -> tensor<64x16384x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<64x16384x128xf32>) -> tensor<64x128x16384xf32>
    return %3 : tensor<64x128x16384xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<128x1xf32>, %arg1: tensor<128xf32>, %arg2: tensor<64x256x256xf32>, %arg3: tensor<128x1xf32>, %arg4: tensor<128xf32>, %arg5: tensor<64x1024x256xf32>, %arg6: tensor<128x1xf32>, %arg7: tensor<128x1xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<64x16384x256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<64x4096x256xf32>, %arg13: tensor<256xf32>) -> tensor<64x1024x128x128xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5 = stablehlo.add %3, %4 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %9 = stablehlo.add %7, %8 : tensor<128xf32>
    %10 = stablehlo.maximum %1, %9 : tensor<128xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %12 = stablehlo.convert %11 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %14 = stablehlo.compare LT, %12, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_3 = stablehlo.constant dense<16> : tensor<i32>
    %15 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %16 = stablehlo.add %12, %15 : tensor<128x1xi32>
    %17 = stablehlo.select %14, %16, %12 : tensor<128x1xi1>, tensor<128x1xi32>
    %18 = mhlo.bitcast %17 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %20 = stablehlo.convert %10 : (tensor<128xf32>) -> tensor<128xi32>
    %21 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %22 = stablehlo.compare LT, %20, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %23 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %24 = stablehlo.add %20, %23 : tensor<128xi32>
    %25 = stablehlo.select %22, %24, %20 : tensor<128xi1>, tensor<128xi32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %27 = stablehlo.concatenate %19, %26, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %28 = mhlo.bitcast %27 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %29 = "stablehlo.gather"(%0, %28) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %30 = mhlo.bitcast %29 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %c_4 = stablehlo.constant dense<15> : tensor<i32>
    %31 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %c_5 = stablehlo.constant dense<1> : tensor<i32>
    %32 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %33 = stablehlo.add %20, %32 : tensor<128xi32>
    %34 = stablehlo.minimum %31, %33 : tensor<128xi32>
    %35 = stablehlo.compare LT, %34, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %36 = stablehlo.add %34, %23 : tensor<128xi32>
    %37 = stablehlo.select %35, %36, %34 : tensor<128xi1>, tensor<128xi32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %39 = stablehlo.concatenate %19, %38, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %40 = mhlo.bitcast %39 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %41 = "stablehlo.gather"(%0, %40) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %43 = stablehlo.subtract %42, %30 : tensor<64x256x128x128xf32>
    %44 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<64x256x128x128xf32>
    %46 = stablehlo.add %30, %45 : tensor<64x256x128x128xf32>
    %47 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %48 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %49 = stablehlo.add %12, %48 : tensor<128x1xi32>
    %50 = stablehlo.minimum %47, %49 : tensor<128x1xi32>
    %51 = stablehlo.compare LT, %50, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %52 = stablehlo.add %50, %15 : tensor<128x1xi32>
    %53 = stablehlo.select %51, %52, %50 : tensor<128x1xi1>, tensor<128x1xi32>
    %54 = mhlo.bitcast %53 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %56 = stablehlo.concatenate %55, %26, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %57 = mhlo.bitcast %56 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %58 = "stablehlo.gather"(%0, %57) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %60 = stablehlo.concatenate %55, %38, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %62 = "stablehlo.gather"(%0, %61) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %64 = stablehlo.subtract %63, %59 : tensor<64x256x128x128xf32>
    %65 = stablehlo.multiply %64, %44 : tensor<64x256x128x128xf32>
    %66 = stablehlo.add %59, %65 : tensor<64x256x128x128xf32>
    %67 = stablehlo.subtract %66, %46 : tensor<64x256x128x128xf32>
    %68 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %70 = stablehlo.multiply %67, %69 : tensor<64x256x128x128xf32>
    %71 = stablehlo.add %46, %70 : tensor<64x256x128x128xf32>
    %72 = mhlo.bitcast %arg5 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32,32]{1,3,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x32x32xf32>
    %cst_6 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %74 = stablehlo.multiply %5, %73 : tensor<128xf32>
    %75 = stablehlo.add %74, %8 : tensor<128xf32>
    %76 = stablehlo.maximum %1, %75 : tensor<128xf32>
    %77 = mhlo.bitcast %76 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %78 = stablehlo.convert %77 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %79 = stablehlo.compare LT, %78, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_7 = stablehlo.constant dense<32> : tensor<i32>
    %80 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %81 = stablehlo.add %78, %80 : tensor<128x1xi32>
    %82 = stablehlo.select %79, %81, %78 : tensor<128x1xi1>, tensor<128x1xi32>
    %83 = mhlo.bitcast %82 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %85 = stablehlo.convert %76 : (tensor<128xf32>) -> tensor<128xi32>
    %86 = stablehlo.compare LT, %85, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %87 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %88 = stablehlo.add %85, %87 : tensor<128xi32>
    %89 = stablehlo.select %86, %88, %85 : tensor<128xi1>, tensor<128xi32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %91 = stablehlo.concatenate %84, %90, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %92 = mhlo.bitcast %91 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %93 = "stablehlo.gather"(%72, %92) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %c_8 = stablehlo.constant dense<31> : tensor<i32>
    %95 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %96 = stablehlo.add %85, %32 : tensor<128xi32>
    %97 = stablehlo.minimum %95, %96 : tensor<128xi32>
    %98 = stablehlo.compare LT, %97, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %99 = stablehlo.add %97, %87 : tensor<128xi32>
    %100 = stablehlo.select %98, %99, %97 : tensor<128xi1>, tensor<128xi32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %102 = stablehlo.concatenate %84, %101, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %103 = mhlo.bitcast %102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %104 = "stablehlo.gather"(%72, %103) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %106 = stablehlo.subtract %105, %94 : tensor<64x256x128x128xf32>
    %107 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %108 = stablehlo.multiply %106, %107 : tensor<64x256x128x128xf32>
    %109 = stablehlo.add %94, %108 : tensor<64x256x128x128xf32>
    %110 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %111 = stablehlo.add %78, %48 : tensor<128x1xi32>
    %112 = stablehlo.minimum %110, %111 : tensor<128x1xi32>
    %113 = stablehlo.compare LT, %112, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %114 = stablehlo.add %112, %80 : tensor<128x1xi32>
    %115 = stablehlo.select %113, %114, %112 : tensor<128x1xi1>, tensor<128x1xi32>
    %116 = mhlo.bitcast %115 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %118 = stablehlo.concatenate %117, %90, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %119 = mhlo.bitcast %118 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %120 = "stablehlo.gather"(%72, %119) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %121 = mhlo.bitcast %120 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %122 = stablehlo.concatenate %117, %101, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %124 = "stablehlo.gather"(%72, %123) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %125 = mhlo.bitcast %124 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %126 = stablehlo.subtract %125, %121 : tensor<64x256x128x128xf32>
    %127 = stablehlo.multiply %126, %107 : tensor<64x256x128x128xf32>
    %128 = stablehlo.add %121, %127 : tensor<64x256x128x128xf32>
    %129 = stablehlo.subtract %128, %109 : tensor<64x256x128x128xf32>
    %130 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %132 = stablehlo.multiply %129, %131 : tensor<64x256x128x128xf32>
    %133 = stablehlo.add %109, %132 : tensor<64x256x128x128xf32>
    %134 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<256xf32>) -> tensor<64x4096x256xf32>
    %135 = stablehlo.add %arg12, %134 : tensor<64x4096x256xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x4096x256xf32>) -> tensor<64x256x64x64xf32>
    %137 = stablehlo.multiply %5, %4 : tensor<128xf32>
    %138 = stablehlo.add %137, %8 : tensor<128xf32>
    %139 = stablehlo.maximum %1, %138 : tensor<128xf32>
    %140 = mhlo.bitcast %139 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %141 = stablehlo.convert %140 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %142 = stablehlo.compare LT, %141, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_9 = stablehlo.constant dense<64> : tensor<i32>
    %143 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %144 = stablehlo.add %141, %143 : tensor<128x1xi32>
    %145 = stablehlo.select %142, %144, %141 : tensor<128x1xi1>, tensor<128x1xi32>
    %146 = mhlo.bitcast %145 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %148 = stablehlo.convert %139 : (tensor<128xf32>) -> tensor<128xi32>
    %149 = stablehlo.compare LT, %148, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %150 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %151 = stablehlo.add %148, %150 : tensor<128xi32>
    %152 = stablehlo.select %149, %151, %148 : tensor<128xi1>, tensor<128xi32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %154 = stablehlo.concatenate %147, %153, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %155 = mhlo.bitcast %154 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %156 = "stablehlo.gather"(%136, %155) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %c_10 = stablehlo.constant dense<63> : tensor<i32>
    %158 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %159 = stablehlo.add %148, %32 : tensor<128xi32>
    %160 = stablehlo.minimum %158, %159 : tensor<128xi32>
    %161 = stablehlo.compare LT, %160, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %162 = stablehlo.add %160, %150 : tensor<128xi32>
    %163 = stablehlo.select %161, %162, %160 : tensor<128xi1>, tensor<128xi32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %165 = stablehlo.concatenate %147, %164, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %166 = mhlo.bitcast %165 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %167 = "stablehlo.gather"(%136, %166) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %168 = mhlo.bitcast %167 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %169 = stablehlo.subtract %168, %157 : tensor<64x256x128x128xf32>
    %170 = stablehlo.broadcast_in_dim %arg8, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %171 = stablehlo.multiply %169, %170 : tensor<64x256x128x128xf32>
    %172 = stablehlo.add %157, %171 : tensor<64x256x128x128xf32>
    %173 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %174 = stablehlo.add %141, %48 : tensor<128x1xi32>
    %175 = stablehlo.minimum %173, %174 : tensor<128x1xi32>
    %176 = stablehlo.compare LT, %175, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %177 = stablehlo.add %175, %143 : tensor<128x1xi32>
    %178 = stablehlo.select %176, %177, %175 : tensor<128x1xi1>, tensor<128x1xi32>
    %179 = mhlo.bitcast %178 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %181 = stablehlo.concatenate %180, %153, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %182 = mhlo.bitcast %181 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %183 = "stablehlo.gather"(%136, %182) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %185 = stablehlo.concatenate %180, %164, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %187 = "stablehlo.gather"(%136, %186) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %189 = stablehlo.subtract %188, %184 : tensor<64x256x128x128xf32>
    %190 = stablehlo.multiply %189, %170 : tensor<64x256x128x128xf32>
    %191 = stablehlo.add %184, %190 : tensor<64x256x128x128xf32>
    %192 = stablehlo.subtract %191, %172 : tensor<64x256x128x128xf32>
    %193 = mhlo.bitcast %arg6 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %195 = stablehlo.multiply %192, %194 : tensor<64x256x128x128xf32>
    %196 = stablehlo.add %172, %195 : tensor<64x256x128x128xf32>
    %197 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<256xf32>) -> tensor<64x16384x256xf32>
    %198 = stablehlo.add %arg10, %197 : tensor<64x16384x256xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,128,128]{1,3,2,0}"} : (tensor<64x16384x256xf32>) -> tensor<64x256x128x128xf32>
    %200 = stablehlo.maximum %1, %3 : tensor<128xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %202 = stablehlo.convert %201 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %203 = stablehlo.compare LT, %202, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_11 = stablehlo.constant dense<128> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %205 = stablehlo.add %202, %204 : tensor<128x1xi32>
    %206 = stablehlo.select %203, %205, %202 : tensor<128x1xi1>, tensor<128x1xi32>
    %207 = mhlo.bitcast %206 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %209 = stablehlo.convert %200 : (tensor<128xf32>) -> tensor<128xi32>
    %210 = stablehlo.compare LT, %209, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %211 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %212 = stablehlo.add %209, %211 : tensor<128xi32>
    %213 = stablehlo.select %210, %212, %209 : tensor<128xi1>, tensor<128xi32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %215 = stablehlo.concatenate %208, %214, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %216 = mhlo.bitcast %215 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %217 = "stablehlo.gather"(%199, %216) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %c_12 = stablehlo.constant dense<127> : tensor<i32>
    %219 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %220 = stablehlo.add %209, %32 : tensor<128xi32>
    %221 = stablehlo.minimum %219, %220 : tensor<128xi32>
    %222 = stablehlo.compare LT, %221, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %223 = stablehlo.add %221, %211 : tensor<128xi32>
    %224 = stablehlo.select %222, %223, %221 : tensor<128xi1>, tensor<128xi32>
    %225 = stablehlo.broadcast_in_dim %224, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %226 = stablehlo.concatenate %208, %225, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %227 = mhlo.bitcast %226 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %228 = "stablehlo.gather"(%199, %227) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %230 = stablehlo.subtract %229, %218 : tensor<64x256x128x128xf32>
    %231 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %232 = stablehlo.multiply %230, %231 : tensor<64x256x128x128xf32>
    %233 = stablehlo.add %218, %232 : tensor<64x256x128x128xf32>
    %234 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %235 = stablehlo.add %202, %48 : tensor<128x1xi32>
    %236 = stablehlo.minimum %234, %235 : tensor<128x1xi32>
    %237 = stablehlo.compare LT, %236, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %238 = stablehlo.add %236, %204 : tensor<128x1xi32>
    %239 = stablehlo.select %237, %238, %236 : tensor<128x1xi1>, tensor<128x1xi32>
    %240 = mhlo.bitcast %239 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %241 = stablehlo.broadcast_in_dim %240, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %242 = stablehlo.concatenate %241, %214, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %243 = mhlo.bitcast %242 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %244 = "stablehlo.gather"(%199, %243) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %246 = stablehlo.concatenate %241, %225, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %247 = mhlo.bitcast %246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %248 = "stablehlo.gather"(%199, %247) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %249 = mhlo.bitcast %248 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
    %250 = stablehlo.subtract %249, %245 : tensor<64x256x128x128xf32>
    %251 = stablehlo.multiply %250, %231 : tensor<64x256x128x128xf32>
    %252 = stablehlo.add %245, %251 : tensor<64x256x128x128xf32>
    %253 = stablehlo.subtract %252, %233 : tensor<64x256x128x128xf32>
    %254 = mhlo.bitcast %arg7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %256 = stablehlo.multiply %253, %255 : tensor<64x256x128x128xf32>
    %257 = stablehlo.add %233, %256 : tensor<64x256x128x128xf32>
    %258 = stablehlo.concatenate %71, %133, %196, %257, dim = 1 : (tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>) -> tensor<64x1024x128x128xf32>
    return %258 : tensor<64x1024x128x128xf32>
  }
  func.func private @fused_maximum(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x128x128xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x128x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x128x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x256x128x128xf32>
    return %9 : tensor<64x256x128x128xf32>
  }
  func.func private @region_0.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.562(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<1048576x32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<64x16384x32xf32>, %arg5: tensor<1048576x32xf32>, %arg6: tensor<32xf32>) -> (tensor<64x32x16384xf32>, tensor<64x16384x32xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<1048576x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<64x16384x32xf32>
    %7 = stablehlo.add %2, %6 : tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x16384xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<64x16384x32xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<64x16384x32xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %17 = stablehlo.add %15, %16 : tensor<64x16384xf32>
    %18 = stablehlo.rsqrt %17 : tensor<64x16384xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<64x16384x32xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x16384x32xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x16384x32xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
    return %25, %20 : tensor<64x32x16384xf32>, tensor<64x16384x32xf32>
  }
  func.func private @fused_transpose.32(%arg0: tensor<32xf32>, %arg1: tensor<64x16384x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x32x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
    return %4 : tensor<64x32x16384xf32>
  }
  func.func private @fused_transpose.33(%arg0: tensor<32xf32>, %arg1: tensor<64x16384x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x32x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
    return %4 : tensor<64x32x16384xf32>
  }
  func.func private @fused_add(%arg0: tensor<64x1024x256xf32>, %arg1: tensor<256xf32>) -> tensor<64x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x1024x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x1024x256xf32>
    return %1 : tensor<64x1024x256xf32>
  }
  func.func private @region_0.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.683(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<262144x64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64x4096x64xf32>, %arg5: tensor<262144x64xf32>, %arg6: tensor<64xf32>) -> (tensor<64x64x4096xf32>, tensor<64x4096x64xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<262144x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<64x4096x64xf32>
    %7 = stablehlo.add %2, %6 : tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x4096xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<64x4096x64xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<64x4096x64xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<64x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %17 = stablehlo.add %15, %16 : tensor<64x4096xf32>
    %18 = stablehlo.rsqrt %17 : tensor<64x4096xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<64x4096x64xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x4096x64xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x4096x64xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
    return %25, %20 : tensor<64x64x4096xf32>, tensor<64x4096x64xf32>
  }
  func.func private @fused_transpose.34(%arg0: tensor<64xf32>, %arg1: tensor<64x4096x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x64x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
    return %4 : tensor<64x64x4096xf32>
  }
  func.func private @fused_transpose.35(%arg0: tensor<64xf32>, %arg1: tensor<64x4096x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x64x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
    return %4 : tensor<64x64x4096xf32>
  }
  func.func private @region_0.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.702(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<65536x160xf32>, %arg3: tensor<160xf32>, %arg4: tensor<64x1024x160xf32>, %arg5: tensor<65536x160xf32>, %arg6: tensor<160xf32>) -> (tensor<64x160x1024xf32>, tensor<64x1024x160xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<65536x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<64x1024x160xf32>
    %7 = stablehlo.add %2, %6 : tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<64x1024xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<64x1024x160xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<64x1024x160xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<64x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %17 = stablehlo.add %15, %16 : tensor<64x1024xf32>
    %18 = stablehlo.rsqrt %17 : tensor<64x1024xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<64x1024x160xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x1024x160xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x1024x160xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
    return %25, %20 : tensor<64x160x1024xf32>, tensor<64x1024x160xf32>
  }
  func.func private @fused_transpose.36(%arg0: tensor<160xf32>, %arg1: tensor<64x1024x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x160x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
    return %4 : tensor<64x160x1024xf32>
  }
  func.func private @fused_transpose.37(%arg0: tensor<160xf32>, %arg1: tensor<64x1024x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x160x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
    return %4 : tensor<64x160x1024xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<64x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x256x256xf32>
    return %1 : tensor<64x256x256xf32>
  }
  func.func private @region_0.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<64x16x16xf32>, %arg1: tensor<64x160x16x16xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<64x160x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x160x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %5 : tensor<64x256xf32>
  }
  func.func private @region_0.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<64x16x16xf32>, %arg1: tensor<64x160x16x16xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<64x160x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x160x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %5 : tensor<64x256xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16xf32>, %arg2: tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<64x160x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x160x256xf32>
    return %5 : tensor<64x160x256xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16xf32>, %arg2: tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<64x160x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x160x256xf32>
    return %5 : tensor<64x160x256xf32>
  }
  func.func private @region_0.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<64x64x16x16xf32>, %arg1: tensor<64x16x16xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<64x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<64x64x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<64x64x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<64x64x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %7 : tensor<64x256xf32>
  }
  func.func private @region_0.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<64x64x16x16xf32>, %arg1: tensor<64x16x16xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<64x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<64x64x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<64x64x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<64x64x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %7 : tensor<64x256xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<64x64x16x16xf32>, %arg1: tensor<64x256xf32>, %arg2: tensor<64x16x16xf32>) -> tensor<64x64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %2 = stablehlo.multiply %arg2, %1 : tensor<64x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<64x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %7 = stablehlo.multiply %arg1, %6 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<64x64x256xf32>
    return %12 : tensor<64x64x256xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<64x64x16x16xf32>, %arg1: tensor<64x256xf32>, %arg2: tensor<64x16x16xf32>) -> tensor<64x64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %2 = stablehlo.multiply %arg2, %1 : tensor<64x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<64x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %7 = stablehlo.multiply %arg1, %6 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<64x64x256xf32>
    return %12 : tensor<64x64x256xf32>
  }
  func.func private @region_0.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<64x32x16x16xf32>, %arg1: tensor<64x16x16xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<64x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<64x32x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<64x32x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<64x32x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %7 : tensor<64x256xf32>
  }
  func.func private @region_0.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<64x32x16x16xf32>, %arg1: tensor<64x16x16xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<64x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<64x32x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<64x32x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<64x32x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %7 : tensor<64x256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<64x32x16x16xf32>, %arg1: tensor<64x256xf32>, %arg2: tensor<64x16x16xf32>) -> tensor<64x32x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %2 = stablehlo.multiply %arg2, %1 : tensor<64x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<64x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %7 = stablehlo.multiply %arg1, %6 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<64x32x256xf32>
    return %12 : tensor<64x32x256xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<64x32x16x16xf32>, %arg1: tensor<64x256xf32>, %arg2: tensor<64x16x16xf32>) -> tensor<64x32x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %2 = stablehlo.multiply %arg2, %1 : tensor<64x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<64x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %7 = stablehlo.multiply %arg1, %6 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<64x32x256xf32>
    return %12 : tensor<64x32x256xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<64x16x16xf32>) -> tensor<64x16x16xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x16x16xf32>
    return %1 : tensor<64x16x16xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<64x16x16xf32>) -> tensor<64x16x16xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x16x16xf32>
    return %1 : tensor<64x16x16xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_rsqrt.1(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_clamp() -> tensor<128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4 = stablehlo.add %2, %3 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %8 = stablehlo.add %6, %7 : tensor<128xf32>
    %9 = stablehlo.maximum %0, %8 : tensor<128xf32>
    %10 = stablehlo.convert %9 : (tensor<128xf32>) -> tensor<128xi32>
    %11 = stablehlo.convert %10 : (tensor<128xi32>) -> tensor<128xf32>
    %12 = stablehlo.subtract %9, %11 : tensor<128xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %14 = stablehlo.clamp %0, %12, %13 : tensor<128xf32>
    return %14 : tensor<128xf32>
  }
  func.func private @fused_clamp.1() -> tensor<128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4 = stablehlo.add %2, %3 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %8 = stablehlo.add %6, %7 : tensor<128xf32>
    %9 = stablehlo.maximum %0, %8 : tensor<128xf32>
    %10 = stablehlo.convert %9 : (tensor<128xf32>) -> tensor<128xi32>
    %11 = stablehlo.convert %10 : (tensor<128xi32>) -> tensor<128xf32>
    %12 = stablehlo.subtract %9, %11 : tensor<128xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %14 = stablehlo.clamp %0, %12, %13 : tensor<128xf32>
    return %14 : tensor<128xf32>
  }
  func.func private @fused_clamp.2() -> tensor<128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4 = stablehlo.add %2, %3 : tensor<128xf32>
    %5 = stablehlo.multiply %4, %3 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128xf32>
    %8 = stablehlo.maximum %0, %7 : tensor<128xf32>
    %9 = stablehlo.convert %8 : (tensor<128xf32>) -> tensor<128xi32>
    %10 = stablehlo.convert %9 : (tensor<128xi32>) -> tensor<128xf32>
    %11 = stablehlo.subtract %8, %10 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %13 = stablehlo.clamp %0, %11, %12 : tensor<128xf32>
    return %13 : tensor<128xf32>
  }
  func.func private @fused_clamp.3() -> tensor<128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %3 = stablehlo.maximum %0, %2 : tensor<128xf32>
    %4 = stablehlo.convert %3 : (tensor<128xf32>) -> tensor<128xi32>
    %5 = stablehlo.convert %4 : (tensor<128xi32>) -> tensor<128xf32>
    %6 = stablehlo.subtract %3, %5 : tensor<128xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %8 = stablehlo.clamp %0, %6, %7 : tensor<128xf32>
    return %8 : tensor<128xf32>
  }
  func.func private @fused_clamp.4() -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5 = stablehlo.add %3, %4 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %9 = stablehlo.add %7, %8 : tensor<128xf32>
    %10 = stablehlo.maximum %1, %9 : tensor<128xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %12 = stablehlo.convert %11 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %13 = stablehlo.convert %12 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %14 = stablehlo.subtract %11, %13 : tensor<128x1xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %16 = stablehlo.clamp %0, %14, %15 : tensor<128x1xf32>
    return %16 : tensor<128x1xf32>
  }
  func.func private @fused_clamp.5() -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5 = stablehlo.add %3, %4 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %9 = stablehlo.add %7, %8 : tensor<128xf32>
    %10 = stablehlo.maximum %1, %9 : tensor<128xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %12 = stablehlo.convert %11 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %13 = stablehlo.convert %12 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %14 = stablehlo.subtract %11, %13 : tensor<128x1xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %16 = stablehlo.clamp %0, %14, %15 : tensor<128x1xf32>
    return %16 : tensor<128x1xf32>
  }
  func.func private @fused_clamp.6() -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5 = stablehlo.add %3, %4 : tensor<128xf32>
    %6 = stablehlo.multiply %5, %4 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %8 = stablehlo.add %6, %7 : tensor<128xf32>
    %9 = stablehlo.maximum %1, %8 : tensor<128xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %11 = stablehlo.convert %10 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %12 = stablehlo.convert %11 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %13 = stablehlo.subtract %10, %12 : tensor<128x1xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %15 = stablehlo.clamp %0, %13, %14 : tensor<128x1xf32>
    return %15 : tensor<128x1xf32>
  }
  func.func private @fused_clamp.7() -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %4 = stablehlo.maximum %1, %3 : tensor<128xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %6 = stablehlo.convert %5 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %7 = stablehlo.convert %6 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %8 = stablehlo.subtract %5, %7 : tensor<128x1xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %10 = stablehlo.clamp %0, %8, %9 : tensor<128x1xf32>
    return %10 : tensor<128x1xf32>
  }
  func.func private @fused_rsqrt.2(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<64x32x16x16xf32>) -> tensor<64x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    return %0 : tensor<64x16x16xf32>
  }
  func.func private @region_0.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<64x160x16x16xf32>) -> tensor<64x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    return %0 : tensor<64x16x16xf32>
  }
  func.func private @region_0.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<64x32x16x16xf32>) -> tensor<64x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    return %0 : tensor<64x16x16xf32>
  }
  func.func private @region_0.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<64x64x16x16xf32>) -> tensor<64x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    return %0 : tensor<64x16x16xf32>
  }
  func.func private @region_0.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<64x64x16x16xf32>) -> tensor<64x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    return %0 : tensor<64x16x16xf32>
  }
  func.func private @region_0.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<64x160x16x16xf32>) -> tensor<64x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
    return %0 : tensor<64x16x16xf32>
  }
  func.func private @wrapped_concatenate_computation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
  func.func private @wrapped_slice_computation(%arg0: tensor<16384x64xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.slice %arg0 [0:16384, 0:32] : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
    return %0 : tensor<16384x32xf32>
  }
  func.func private @wrapped_slice_computation.1(%arg0: tensor<16384x64xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.slice %arg0 [0:16384, 32:64] : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
    return %0 : tensor<16384x32xf32>
  }
  func.func private @wrapped_concatenate_computation.1(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
  func.func private @wrapped_slice_computation.2(%arg0: tensor<16384x64xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.slice %arg0 [0:16384, 0:32] : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
    return %0 : tensor<16384x32xf32>
  }
  func.func private @wrapped_slice_computation.3(%arg0: tensor<16384x64xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.slice %arg0 [0:16384, 32:64] : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
    return %0 : tensor<16384x32xf32>
  }
  func.func private @wrapped_concatenate_computation.2(%arg0: tensor<160x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<320x160xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    return %0 : tensor<320x160xf32>
  }
  func.func private @wrapped_concatenate_computation.3(%arg0: tensor<160x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<320x160xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    return %0 : tensor<320x160xf32>
  }
  func.func @main(%arg0: tensor<32x3x7x7xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<64x32x3x3xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<160x64x3x3xf32>, %arg9: tensor<160xf32>, %arg10: tensor<160xf32>, %arg11: tensor<160xf32>, %arg12: tensor<256x160x3x3xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32x32xf32>, %arg21: tensor<32xf32>, %arg22: tensor<32x32xf32>, %arg23: tensor<32xf32>, %arg24: tensor<32x32x8x8xf32>, %arg25: tensor<32xf32>, %arg26: tensor<32xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32x32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<32xf32>, %arg31: tensor<32xf32>, %arg32: tensor<128x32xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128x1x3x3xf32>, %arg35: tensor<128xf32>, %arg36: tensor<32x128xf32>, %arg37: tensor<32xf32>, %arg38: tensor<32xf32>, %arg39: tensor<32xf32>, %arg40: tensor<32x32xf32>, %arg41: tensor<32xf32>, %arg42: tensor<32x32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<32x32xf32>, %arg45: tensor<32xf32>, %arg46: tensor<32x32x8x8xf32>, %arg47: tensor<32xf32>, %arg48: tensor<32xf32>, %arg49: tensor<32xf32>, %arg50: tensor<32x32xf32>, %arg51: tensor<32xf32>, %arg52: tensor<32xf32>, %arg53: tensor<32xf32>, %arg54: tensor<128x32xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128x1x3x3xf32>, %arg57: tensor<128xf32>, %arg58: tensor<32x128xf32>, %arg59: tensor<32xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64xf32>, %arg62: tensor<64x64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64x64xf32>, %arg65: tensor<64xf32>, %arg66: tensor<64x64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<64x64x4x4xf32>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64x64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<64xf32>, %arg75: tensor<64xf32>, %arg76: tensor<256x64xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x1x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<64x256xf32>, %arg81: tensor<64xf32>, %arg82: tensor<64xf32>, %arg83: tensor<64xf32>, %arg84: tensor<64x64xf32>, %arg85: tensor<64xf32>, %arg86: tensor<64x64xf32>, %arg87: tensor<64xf32>, %arg88: tensor<64x64xf32>, %arg89: tensor<64xf32>, %arg90: tensor<64x64x4x4xf32>, %arg91: tensor<64xf32>, %arg92: tensor<64xf32>, %arg93: tensor<64xf32>, %arg94: tensor<64x64xf32>, %arg95: tensor<64xf32>, %arg96: tensor<64xf32>, %arg97: tensor<64xf32>, %arg98: tensor<256x64xf32>, %arg99: tensor<256xf32>, %arg100: tensor<256x1x3x3xf32>, %arg101: tensor<256xf32>, %arg102: tensor<64x256xf32>, %arg103: tensor<64xf32>, %arg104: tensor<160xf32>, %arg105: tensor<160xf32>, %arg106: tensor<160x160xf32>, %arg107: tensor<160xf32>, %arg108: tensor<160x160xf32>, %arg109: tensor<160xf32>, %arg110: tensor<160x160xf32>, %arg111: tensor<160xf32>, %arg112: tensor<160x160x2x2xf32>, %arg113: tensor<160xf32>, %arg114: tensor<160xf32>, %arg115: tensor<160xf32>, %arg116: tensor<160x160xf32>, %arg117: tensor<160xf32>, %arg118: tensor<160xf32>, %arg119: tensor<160xf32>, %arg120: tensor<640x160xf32>, %arg121: tensor<640xf32>, %arg122: tensor<640x1x3x3xf32>, %arg123: tensor<640xf32>, %arg124: tensor<160x640xf32>, %arg125: tensor<160xf32>, %arg126: tensor<160xf32>, %arg127: tensor<160xf32>, %arg128: tensor<160x160xf32>, %arg129: tensor<160xf32>, %arg130: tensor<160x160xf32>, %arg131: tensor<160xf32>, %arg132: tensor<160x160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<160x160x2x2xf32>, %arg135: tensor<160xf32>, %arg136: tensor<160xf32>, %arg137: tensor<160xf32>, %arg138: tensor<160x160xf32>, %arg139: tensor<160xf32>, %arg140: tensor<160xf32>, %arg141: tensor<160xf32>, %arg142: tensor<640x160xf32>, %arg143: tensor<640xf32>, %arg144: tensor<640x1x3x3xf32>, %arg145: tensor<640xf32>, %arg146: tensor<160x640xf32>, %arg147: tensor<160xf32>, %arg148: tensor<256xf32>, %arg149: tensor<256xf32>, %arg150: tensor<256x256xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256x256xf32>, %arg153: tensor<256xf32>, %arg154: tensor<256x256xf32>, %arg155: tensor<256xf32>, %arg156: tensor<256x256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<1024x256xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<1024x1x3x3xf32>, %arg163: tensor<1024xf32>, %arg164: tensor<256x1024xf32>, %arg165: tensor<256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<256x256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256x256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x256xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256x256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<1024x256xf32>, %arg179: tensor<1024xf32>, %arg180: tensor<1024x1x3x3xf32>, %arg181: tensor<1024xf32>, %arg182: tensor<256x1024xf32>, %arg183: tensor<256xf32>, %arg184: tensor<32xf32>, %arg185: tensor<32xf32>, %arg186: tensor<64xf32>, %arg187: tensor<64xf32>, %arg188: tensor<160xf32>, %arg189: tensor<160xf32>, %arg190: tensor<256xf32>, %arg191: tensor<256xf32>, %arg192: tensor<256x32xf32>, %arg193: tensor<256xf32>, %arg194: tensor<256x64xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256x160xf32>, %arg197: tensor<256xf32>, %arg198: tensor<256x256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256x1024x1x1xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<150x256x1x1xf32>, %arg204: tensor<150xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<64x3x512x512xf32>) -> tensor<64x150x128x128xf32> {
    %0 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %298 = stablehlo.add %arg208, %297 : tensor<256xf32>
      %299 = stablehlo.rsqrt %298 : tensor<256xf32>
      mhlo.return %299 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %299 = stablehlo.iota dim = 0 : tensor<128xi32>
      %300 = stablehlo.convert %299 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %302 = stablehlo.add %300, %301 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %306 = stablehlo.add %304, %305 : tensor<128xf32>
      %307 = stablehlo.maximum %298, %306 : tensor<128xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %309 = stablehlo.convert %308 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %310 = stablehlo.convert %309 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %311 = stablehlo.subtract %308, %310 : tensor<128x1xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %313 = stablehlo.clamp %297, %311, %312 : tensor<128x1xf32>
      mhlo.return %313 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %298 = stablehlo.iota dim = 0 : tensor<128xi32>
      %299 = stablehlo.convert %298 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %301 = stablehlo.add %299, %300 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.add %303, %304 : tensor<128xf32>
      %306 = stablehlo.maximum %297, %305 : tensor<128xf32>
      %307 = stablehlo.convert %306 : (tensor<128xf32>) -> tensor<128xi32>
      %308 = stablehlo.convert %307 : (tensor<128xi32>) -> tensor<128xf32>
      %309 = stablehlo.subtract %306, %308 : tensor<128xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %311 = stablehlo.clamp %297, %309, %310 : tensor<128xf32>
      mhlo.return %311 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %3 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%arg207, %arg0, %arg1) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x3x512x512xf32>, tensor<32x3x7x7xf32>, tensor<32xf32>) -> tuple<tensor<64x32x128x128xf32>, tensor<0xui8>>
    %4 = stablehlo.get_tuple_element %3[0] : (tuple<tensor<64x32x128x128xf32>, tensor<0xui8>>) -> tensor<64x32x128x128xf32>
    %5 = "mhlo.fusion"(%arg3, %arg2, %4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x32x128x128xf32>):
      %297 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1] : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x128x128xf32>, tensor<f32>) -> tensor<64x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<64x128x128xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x128xf32>) -> tensor<64x16384xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %304 = stablehlo.subtract %298, %303 : tensor<64x16384x32xf32>
      %305 = stablehlo.multiply %304, %304 : tensor<64x16384x32xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<64x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %310 = stablehlo.add %308, %309 : tensor<64x16384xf32>
      %311 = stablehlo.rsqrt %310 : tensor<64x16384xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %313 = stablehlo.multiply %304, %312 : tensor<64x16384x32xf32>
      %314 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<64x16384x32xf32>
      %316 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %317 = stablehlo.add %315, %316 : tensor<64x16384x32xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %319 = stablehlo.multiply %318, %307 : tensor<64x16384xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<64x16384x32xf32>
      %322 = stablehlo.multiply %321, %321 : tensor<64x16384x32xf32>
      %323 = stablehlo.reduce(%322 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %324 = stablehlo.multiply %323, %307 : tensor<64x16384xf32>
      %325 = stablehlo.add %324, %309 : tensor<64x16384xf32>
      %326 = stablehlo.rsqrt %325 : tensor<64x16384xf32>
      %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %328 = stablehlo.multiply %321, %327 : tensor<64x16384x32xf32>
      mhlo.return %328 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x32x128x128xf32>) -> tensor<64x16384x32xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    %7 = "mhlo.fusion"(%6, %arg16, %arg17, %arg18) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<1048576x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %300 = stablehlo.add %298, %299 : tensor<1048576x32xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
      mhlo.return %301 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %9 = "mhlo.fusion"(%arg17, %5, %arg16) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<64x16384x32xf32>, %arg210: tensor<32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %298 = stablehlo.multiply %arg209, %297 : tensor<64x16384x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x16384x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
      mhlo.return %301 : tensor<64x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<64x16384x32xf32>, tensor<32xf32>) -> tensor<64x32x16384xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x16384xf32>) -> tensor<64x32x128x128xf32>
    %11 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%10, %arg24, %arg25) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,32,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x32x128x128xf32>, tensor<32x32x8x8xf32>, tensor<32xf32>) -> tuple<tensor<64x32x16x16xf32>, tensor<0xui8>>
    %12 = stablehlo.get_tuple_element %11[0] : (tuple<tensor<64x32x16x16xf32>, tensor<0xui8>>) -> tensor<64x32x16x16xf32>
    %13 = "mhlo.fusion"(%12) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x32x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      mhlo.return %297 : tensor<64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x16x16xf32>) -> tensor<64x16x16xf32>
    %14 = "mhlo.fusion"(%12, %13) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x32x16x16xf32>, %arg209: tensor<64x16x16xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %299 = stablehlo.multiply %arg209, %298 : tensor<64x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
      %302 = stablehlo.subtract %297, %301 : tensor<64x32x256xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<64x32x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %304 = stablehlo.reduce(%303 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<64x32x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %304 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x16x16xf32>, tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %15 = "mhlo.fusion"(%12, %14, %13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x16x16xf32>, %arg209: tensor<64x256xf32>, %arg210: tensor<64x16x16xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %299 = stablehlo.multiply %arg210, %298 : tensor<64x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
      %302 = stablehlo.subtract %297, %301 : tensor<64x32x256xf32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %304 = stablehlo.multiply %arg209, %303 : tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<64x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<64x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
      %309 = stablehlo.multiply %302, %308 : tensor<64x32x256xf32>
      mhlo.return %309 : tensor<64x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x16x16xf32>, tensor<64x256xf32>, tensor<64x16x16xf32>) -> tensor<64x32x256xf32>
    %16 = "mhlo.fusion"(%arg20, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32xf32>, %arg209: tensor<32x32xf32>):
      %297 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
      mhlo.return %297 : tensor<64x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,64]{0,1}"} : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %18 = "mhlo.fusion"(%15, %arg26, %arg27, %17) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x32x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<64x32x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x32x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<64x32x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x32x256xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<16384x32xf32>
      %303 = stablehlo.dot %302, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x64xf32>) -> tensor<16384x64xf32>
      mhlo.return %303 : tensor<16384x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x256xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x64xf32>) -> tensor<16384x64xf32>
    %19 = "mhlo.fusion"(%18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x64xf32>):
      %297 = stablehlo.slice %arg208 [0:16384, 0:32] : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
      mhlo.return %297 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<64x256x32xf32>
    %21 = "mhlo.fusion"(%arg19, %8, %arg21, %20) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<64x16384x32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<64x256x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %298 = stablehlo.add %297, %arg209 : tensor<64x16384x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
      %300 = stablehlo.add %299, %arg211 : tensor<64x256x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x256x32xf32>) -> tensor<64x32x256xf32>
      %302 = stablehlo.dot_general %298, %301, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x32xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x16384x256xf32>
      mhlo.return %304 : tensor<64x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<64x16384x32xf32>, tensor<32xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x256xf32>
    %22 = "mhlo.fusion"(%21) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<64x16384x256xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x256xf32>
      %299 = stablehlo.subtract %arg208, %298 : tensor<64x16384x256xf32>
      %300 = stablehlo.exponential %299 : tensor<64x16384x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%300 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x256xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x256xf32>
      %303 = stablehlo.divide %300, %302 : tensor<64x16384x256xf32>
      mhlo.return %303 : tensor<64x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x256xf32>) -> tensor<64x16384x256xf32>
    %23 = "mhlo.fusion"(%18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x64xf32>):
      %297 = stablehlo.slice %arg208 [0:16384, 32:64] : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
      mhlo.return %297 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
    %24 = mhlo.bitcast %23 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<64x256x32xf32>
    %25 = "mhlo.fusion"(%22, %arg23, %24) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x256x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
      %298 = stablehlo.add %297, %arg210 : tensor<64x256x32xf32>
      %299 = stablehlo.dot_general %arg208, %298, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x256xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x32xf32>
      mhlo.return %299 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x256xf32>, tensor<32xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x32xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    %27 = "mhlo.fusion"(%26, %arg28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32x32xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
      mhlo.return %297 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    %28 = "mhlo.fusion"(%arg3, %arg2, %4, %27, %arg29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x32x128x128xf32>, %arg211: tensor<1048576x32xf32>, %arg212: tensor<32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %298 = stablehlo.add %297, %arg211 : tensor<1048576x32xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %300 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x128x128xf32>, tensor<f32>) -> tensor<64x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x128x128xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x128xf32>) -> tensor<64x16384xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %307 = stablehlo.subtract %301, %306 : tensor<64x16384x32xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<64x16384x32xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x16384xf32>
      %314 = stablehlo.rsqrt %313 : tensor<64x16384xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %316 = stablehlo.multiply %307, %315 : tensor<64x16384x32xf32>
      %317 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<64x16384x32xf32>
      %319 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %320 = stablehlo.add %318, %319 : tensor<64x16384x32xf32>
      %321 = stablehlo.add %299, %320 : tensor<64x16384x32xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x16384xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %325 = stablehlo.subtract %321, %324 : tensor<64x16384x32xf32>
      %326 = stablehlo.multiply %325, %325 : tensor<64x16384x32xf32>
      %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %328 = stablehlo.multiply %327, %310 : tensor<64x16384xf32>
      %329 = stablehlo.add %328, %312 : tensor<64x16384xf32>
      %330 = stablehlo.rsqrt %329 : tensor<64x16384xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %332 = stablehlo.multiply %325, %331 : tensor<64x16384x32xf32>
      mhlo.return %332 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x32x128x128xf32>, tensor<1048576x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %29 = mhlo.bitcast %28 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    %30 = "mhlo.fusion"(%29, %arg30, %arg31, %arg32) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<1048576x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %300 = stablehlo.add %298, %299 : tensor<1048576x32xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<128x32xf32>) -> tensor<1048576x128xf32>
      mhlo.return %301 : tensor<1048576x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32xf32>) -> tensor<1048576x128xf32>
    %31 = "mhlo.fusion"(%30, %arg33) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1048576x128xf32>, %arg209: tensor<128xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<1048576x128xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<1048576x128xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x128xf32>) -> tensor<64x16384x128xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] : (tensor<64x16384x128xf32>) -> tensor<64x128x16384xf32>
      mhlo.return %300 : tensor<64x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x128xf32>, tensor<128xf32>) -> tensor<64x128x16384xf32>
    %32 = mhlo.bitcast %31 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x16384xf32>) -> tensor<64x128x128x128xf32>
    %33 = stablehlo.custom_call @__cudnn$convForward(%32, %arg34) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<64x128x128x128xf32>, tensor<0xui8>>
    %34 = stablehlo.get_tuple_element %33[0] : (tuple<tensor<64x128x128x128xf32>, tensor<0xui8>>) -> tensor<64x128x128x128xf32>
    %35 = "mhlo.fusion"(%34, %arg35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %298 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
      %299 = stablehlo.add %arg208, %298 : tensor<64x128x128x128xf32>
      %300 = stablehlo.multiply %297, %299 : tensor<64x128x128x128xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
      %302 = stablehlo.negate %299 : tensor<64x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x128x128x128xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
      %306 = stablehlo.abs %305 : tensor<64x128x16384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %308 = stablehlo.compare LT, %306, %307 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %310 = stablehlo.multiply %305, %305 : tensor<64x128x16384xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x128x16384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x128x16384xf32>
      %314 = stablehlo.multiply %313, %310 : tensor<64x128x16384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x128x16384xf32>
      %317 = stablehlo.multiply %316, %310 : tensor<64x128x16384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %319 = stablehlo.add %317, %318 : tensor<64x128x16384xf32>
      %320 = stablehlo.multiply %319, %310 : tensor<64x128x16384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %322 = stablehlo.add %320, %321 : tensor<64x128x16384xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x128x16384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %325 = stablehlo.add %323, %324 : tensor<64x128x16384xf32>
      %326 = stablehlo.multiply %325, %310 : tensor<64x128x16384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %328 = stablehlo.add %326, %327 : tensor<64x128x16384xf32>
      %329 = stablehlo.multiply %305, %328 : tensor<64x128x16384xf32>
      %330 = stablehlo.subtract %307, %329 : tensor<64x128x16384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %332 = stablehlo.compare LT, %305, %331 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %334 = stablehlo.negate %310 : tensor<64x128x16384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %336 = stablehlo.compare LT, %334, %335 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
      %337 = stablehlo.exponential %334 : tensor<64x128x16384xf32>
      %338 = stablehlo.divide %307, %306 : tensor<64x128x16384xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x128x16384xf32>
      %340 = stablehlo.compare LT, %306, %333 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %342 = stablehlo.divide %307, %310 : tensor<64x128x16384xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<64x128x16384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %345 = stablehlo.add %343, %344 : tensor<64x128x16384xf32>
      %346 = stablehlo.multiply %345, %342 : tensor<64x128x16384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %348 = stablehlo.add %346, %347 : tensor<64x128x16384xf32>
      %349 = stablehlo.multiply %348, %342 : tensor<64x128x16384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %351 = stablehlo.add %349, %350 : tensor<64x128x16384xf32>
      %352 = stablehlo.multiply %351, %342 : tensor<64x128x16384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %354 = stablehlo.add %352, %353 : tensor<64x128x16384xf32>
      %355 = stablehlo.multiply %354, %342 : tensor<64x128x16384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %357 = stablehlo.add %355, %356 : tensor<64x128x16384xf32>
      %358 = stablehlo.multiply %357, %342 : tensor<64x128x16384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %360 = stablehlo.add %358, %359 : tensor<64x128x16384xf32>
      %361 = stablehlo.multiply %360, %342 : tensor<64x128x16384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %363 = stablehlo.add %361, %362 : tensor<64x128x16384xf32>
      %364 = stablehlo.multiply %363, %342 : tensor<64x128x16384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %366 = stablehlo.add %364, %365 : tensor<64x128x16384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %368 = stablehlo.multiply %367, %342 : tensor<64x128x16384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %370 = stablehlo.add %368, %369 : tensor<64x128x16384xf32>
      %371 = stablehlo.multiply %370, %342 : tensor<64x128x16384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %372 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %373 = stablehlo.add %371, %372 : tensor<64x128x16384xf32>
      %374 = stablehlo.multiply %373, %342 : tensor<64x128x16384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %376 = stablehlo.add %374, %375 : tensor<64x128x16384xf32>
      %377 = stablehlo.multiply %376, %342 : tensor<64x128x16384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x128x16384xf32>
      %380 = stablehlo.multiply %379, %342 : tensor<64x128x16384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %382 = stablehlo.add %380, %381 : tensor<64x128x16384xf32>
      %383 = stablehlo.multiply %382, %342 : tensor<64x128x16384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %385 = stablehlo.add %383, %384 : tensor<64x128x16384xf32>
      %386 = stablehlo.multiply %385, %342 : tensor<64x128x16384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %388 = stablehlo.add %386, %387 : tensor<64x128x16384xf32>
      %389 = stablehlo.select %340, %366, %388 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
      %390 = stablehlo.multiply %339, %389 : tensor<64x128x16384xf32>
      %391 = stablehlo.select %336, %331, %390 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
      %392 = stablehlo.subtract %333, %391 : tensor<64x128x16384xf32>
      %393 = stablehlo.select %332, %392, %391 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
      %394 = stablehlo.select %308, %330, %393 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
      %395 = stablehlo.multiply %301, %394 : tensor<64x128x16384xf32>
      mhlo.return %395 : tensor<64x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x128x128xf32>, tensor<128xf32>) -> tensor<64x128x16384xf32>
    %36 = "mhlo.fusion"(%35, %arg36) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x128x16384xf32>, %arg209: tensor<32x128xf32>):
      %297 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x128xf32>) -> tensor<1048576x128xf32>
      %299 = stablehlo.dot_general %298, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x128xf32>, tensor<32x128xf32>) -> tensor<1048576x32xf32>
      mhlo.return %299 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x16384xf32>, tensor<32x128xf32>) -> tensor<1048576x32xf32>
    %37 = "mhlo.fusion"(%36, %arg37, %arg3, %arg2, %4, %27, %arg29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32xf32>, %arg212: tensor<64x32x128x128xf32>, %arg213: tensor<1048576x32xf32>, %arg214: tensor<32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<1048576x32xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %300 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %301 = stablehlo.add %300, %arg213 : tensor<1048576x32xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %303 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x128x128xf32>, tensor<f32>) -> tensor<64x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<64x128x128xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x128xf32>) -> tensor<64x16384xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %310 = stablehlo.subtract %304, %309 : tensor<64x16384x32xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<64x16384x32xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %314 = stablehlo.multiply %312, %313 : tensor<64x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x16384xf32>
      %317 = stablehlo.rsqrt %316 : tensor<64x16384xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %319 = stablehlo.multiply %310, %318 : tensor<64x16384x32xf32>
      %320 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<64x16384x32xf32>
      %322 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %323 = stablehlo.add %321, %322 : tensor<64x16384x32xf32>
      %324 = stablehlo.add %302, %323 : tensor<64x16384x32xf32>
      %325 = stablehlo.add %299, %324 : tensor<64x16384x32xf32>
      mhlo.return %325 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x128x128xf32>, tensor<1048576x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %38 = "mhlo.fusion"(%37) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<64x16384xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %301 = stablehlo.subtract %arg208, %300 : tensor<64x16384x32xf32>
      %302 = stablehlo.multiply %301, %301 : tensor<64x16384x32xf32>
      %303 = stablehlo.reduce(%302 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %304 = stablehlo.multiply %303, %298 : tensor<64x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %306 = stablehlo.add %304, %305 : tensor<64x16384xf32>
      %307 = stablehlo.rsqrt %306 : tensor<64x16384xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %309 = stablehlo.multiply %301, %308 : tensor<64x16384x32xf32>
      mhlo.return %309 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>) -> tensor<64x16384x32xf32>
    %39 = mhlo.bitcast %38 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    %40 = "mhlo.fusion"(%39, %arg38, %arg39, %arg40) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<1048576x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %300 = stablehlo.add %298, %299 : tensor<1048576x32xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
      mhlo.return %301 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    %41 = mhlo.bitcast %40 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %42 = "mhlo.fusion"(%arg39, %38, %arg38) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<64x16384x32xf32>, %arg210: tensor<32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %298 = stablehlo.multiply %arg209, %297 : tensor<64x16384x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x16384x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
      mhlo.return %301 : tensor<64x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<64x16384x32xf32>, tensor<32xf32>) -> tensor<64x32x16384xf32>
    %43 = mhlo.bitcast %42 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x16384xf32>) -> tensor<64x32x128x128xf32>
    %44 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%43, %arg46, %arg47) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,32,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x32x128x128xf32>, tensor<32x32x8x8xf32>, tensor<32xf32>) -> tuple<tensor<64x32x16x16xf32>, tensor<0xui8>>
    %45 = stablehlo.get_tuple_element %44[0] : (tuple<tensor<64x32x16x16xf32>, tensor<0xui8>>) -> tensor<64x32x16x16xf32>
    %46 = "mhlo.fusion"(%45) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x32x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x32x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      mhlo.return %297 : tensor<64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x16x16xf32>) -> tensor<64x16x16xf32>
    %47 = "mhlo.fusion"(%45, %46) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x32x16x16xf32>, %arg209: tensor<64x16x16xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %299 = stablehlo.multiply %arg209, %298 : tensor<64x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
      %302 = stablehlo.subtract %297, %301 : tensor<64x32x256xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<64x32x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %304 = stablehlo.reduce(%303 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<64x32x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %304 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x16x16xf32>, tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %48 = "mhlo.fusion"(%45, %47, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x16x16xf32>, %arg209: tensor<64x256xf32>, %arg210: tensor<64x16x16xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %299 = stablehlo.multiply %arg210, %298 : tensor<64x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
      %302 = stablehlo.subtract %297, %301 : tensor<64x32x256xf32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %304 = stablehlo.multiply %arg209, %303 : tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<64x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<64x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x32x256xf32>
      %309 = stablehlo.multiply %302, %308 : tensor<64x32x256xf32>
      mhlo.return %309 : tensor<64x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x16x16xf32>, tensor<64x256xf32>, tensor<64x16x16xf32>) -> tensor<64x32x256xf32>
    %49 = "mhlo.fusion"(%arg42, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32xf32>, %arg209: tensor<32x32xf32>):
      %297 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
      mhlo.return %297 : tensor<64x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %50 = mhlo.bitcast %49 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,64]{0,1}"} : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %51 = "mhlo.fusion"(%48, %arg48, %arg49, %50) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x32x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<64x32x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x32x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<64x32x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x32x256xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<16384x32xf32>
      %303 = stablehlo.dot %302, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x64xf32>) -> tensor<16384x64xf32>
      mhlo.return %303 : tensor<16384x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x256xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x64xf32>) -> tensor<16384x64xf32>
    %52 = "mhlo.fusion"(%51) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x64xf32>):
      %297 = stablehlo.slice %arg208 [0:16384, 0:32] : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
      mhlo.return %297 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
    %53 = mhlo.bitcast %52 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<64x256x32xf32>
    %54 = "mhlo.fusion"(%arg41, %41, %arg43, %53) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<64x16384x32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<64x256x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %298 = stablehlo.add %297, %arg209 : tensor<64x16384x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
      %300 = stablehlo.add %299, %arg211 : tensor<64x256x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x256x32xf32>) -> tensor<64x32x256xf32>
      %302 = stablehlo.dot_general %298, %301, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x32xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x16384x256xf32>
      mhlo.return %304 : tensor<64x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<64x16384x32xf32>, tensor<32xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x256xf32>
    %55 = "mhlo.fusion"(%54) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<64x16384x256xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x256xf32>
      %299 = stablehlo.subtract %arg208, %298 : tensor<64x16384x256xf32>
      %300 = stablehlo.exponential %299 : tensor<64x16384x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%300 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x256xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x256xf32>
      %303 = stablehlo.divide %300, %302 : tensor<64x16384x256xf32>
      mhlo.return %303 : tensor<64x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x256xf32>) -> tensor<64x16384x256xf32>
    %56 = "mhlo.fusion"(%51) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x64xf32>):
      %297 = stablehlo.slice %arg208 [0:16384, 32:64] : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
      mhlo.return %297 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x64xf32>) -> tensor<16384x32xf32>
    %57 = mhlo.bitcast %56 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<64x256x32xf32>
    %58 = "mhlo.fusion"(%55, %arg45, %57) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x256x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
      %298 = stablehlo.add %297, %arg210 : tensor<64x256x32xf32>
      %299 = stablehlo.dot_general %arg208, %298, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x256xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x32xf32>
      mhlo.return %299 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x256xf32>, tensor<32xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x32xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    %60 = "mhlo.fusion"(%59, %arg50) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32x32xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
      mhlo.return %297 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    %61 = "mhlo.fusion"(%37, %60, %arg51) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<1048576x32xf32>, %arg210: tensor<32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %298 = stablehlo.add %297, %arg209 : tensor<1048576x32xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %300 = stablehlo.add %299, %arg208 : tensor<64x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%300 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<64x16384xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %305 = stablehlo.subtract %300, %304 : tensor<64x16384x32xf32>
      %306 = stablehlo.multiply %305, %305 : tensor<64x16384x32xf32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %308 = stablehlo.multiply %307, %302 : tensor<64x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %310 = stablehlo.add %308, %309 : tensor<64x16384xf32>
      %311 = stablehlo.rsqrt %310 : tensor<64x16384xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %313 = stablehlo.multiply %305, %312 : tensor<64x16384x32xf32>
      mhlo.return %313 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<1048576x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %62 = mhlo.bitcast %61 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    %63 = "mhlo.fusion"(%62, %arg52, %arg53, %arg54) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<1048576x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %300 = stablehlo.add %298, %299 : tensor<1048576x32xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<128x32xf32>) -> tensor<1048576x128xf32>
      mhlo.return %301 : tensor<1048576x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32xf32>) -> tensor<1048576x128xf32>
    %64 = "mhlo.fusion"(%63, %arg55) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1048576x128xf32>, %arg209: tensor<128xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<1048576x128xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<1048576x128xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x128xf32>) -> tensor<64x16384x128xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] : (tensor<64x16384x128xf32>) -> tensor<64x128x16384xf32>
      mhlo.return %300 : tensor<64x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x128xf32>, tensor<128xf32>) -> tensor<64x128x16384xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x16384xf32>) -> tensor<64x128x128x128xf32>
    %66 = stablehlo.custom_call @__cudnn$convForward(%65, %arg56) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<64x128x128x128xf32>, tensor<0xui8>>
    %67 = stablehlo.get_tuple_element %66[0] : (tuple<tensor<64x128x128x128xf32>, tensor<0xui8>>) -> tensor<64x128x128x128xf32>
    %68 = "mhlo.fusion"(%67, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %298 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
      %299 = stablehlo.add %arg208, %298 : tensor<64x128x128x128xf32>
      %300 = stablehlo.multiply %297, %299 : tensor<64x128x128x128xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
      %302 = stablehlo.negate %299 : tensor<64x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x128x128x128xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
      %306 = stablehlo.abs %305 : tensor<64x128x16384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %308 = stablehlo.compare LT, %306, %307 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %310 = stablehlo.multiply %305, %305 : tensor<64x128x16384xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x128x16384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x128x16384xf32>
      %314 = stablehlo.multiply %313, %310 : tensor<64x128x16384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x128x16384xf32>
      %317 = stablehlo.multiply %316, %310 : tensor<64x128x16384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %319 = stablehlo.add %317, %318 : tensor<64x128x16384xf32>
      %320 = stablehlo.multiply %319, %310 : tensor<64x128x16384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %322 = stablehlo.add %320, %321 : tensor<64x128x16384xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x128x16384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %325 = stablehlo.add %323, %324 : tensor<64x128x16384xf32>
      %326 = stablehlo.multiply %325, %310 : tensor<64x128x16384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %328 = stablehlo.add %326, %327 : tensor<64x128x16384xf32>
      %329 = stablehlo.multiply %305, %328 : tensor<64x128x16384xf32>
      %330 = stablehlo.subtract %307, %329 : tensor<64x128x16384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %332 = stablehlo.compare LT, %305, %331 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %334 = stablehlo.negate %310 : tensor<64x128x16384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %336 = stablehlo.compare LT, %334, %335 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
      %337 = stablehlo.exponential %334 : tensor<64x128x16384xf32>
      %338 = stablehlo.divide %307, %306 : tensor<64x128x16384xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x128x16384xf32>
      %340 = stablehlo.compare LT, %306, %333 : (tensor<64x128x16384xf32>, tensor<64x128x16384xf32>) -> tensor<64x128x16384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %342 = stablehlo.divide %307, %310 : tensor<64x128x16384xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<64x128x16384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %345 = stablehlo.add %343, %344 : tensor<64x128x16384xf32>
      %346 = stablehlo.multiply %345, %342 : tensor<64x128x16384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %348 = stablehlo.add %346, %347 : tensor<64x128x16384xf32>
      %349 = stablehlo.multiply %348, %342 : tensor<64x128x16384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %351 = stablehlo.add %349, %350 : tensor<64x128x16384xf32>
      %352 = stablehlo.multiply %351, %342 : tensor<64x128x16384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %354 = stablehlo.add %352, %353 : tensor<64x128x16384xf32>
      %355 = stablehlo.multiply %354, %342 : tensor<64x128x16384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %357 = stablehlo.add %355, %356 : tensor<64x128x16384xf32>
      %358 = stablehlo.multiply %357, %342 : tensor<64x128x16384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %360 = stablehlo.add %358, %359 : tensor<64x128x16384xf32>
      %361 = stablehlo.multiply %360, %342 : tensor<64x128x16384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %363 = stablehlo.add %361, %362 : tensor<64x128x16384xf32>
      %364 = stablehlo.multiply %363, %342 : tensor<64x128x16384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %366 = stablehlo.add %364, %365 : tensor<64x128x16384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %368 = stablehlo.multiply %367, %342 : tensor<64x128x16384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %370 = stablehlo.add %368, %369 : tensor<64x128x16384xf32>
      %371 = stablehlo.multiply %370, %342 : tensor<64x128x16384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %372 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %373 = stablehlo.add %371, %372 : tensor<64x128x16384xf32>
      %374 = stablehlo.multiply %373, %342 : tensor<64x128x16384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %376 = stablehlo.add %374, %375 : tensor<64x128x16384xf32>
      %377 = stablehlo.multiply %376, %342 : tensor<64x128x16384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x128x16384xf32>
      %380 = stablehlo.multiply %379, %342 : tensor<64x128x16384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %382 = stablehlo.add %380, %381 : tensor<64x128x16384xf32>
      %383 = stablehlo.multiply %382, %342 : tensor<64x128x16384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %385 = stablehlo.add %383, %384 : tensor<64x128x16384xf32>
      %386 = stablehlo.multiply %385, %342 : tensor<64x128x16384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x128x16384xf32>
      %388 = stablehlo.add %386, %387 : tensor<64x128x16384xf32>
      %389 = stablehlo.select %340, %366, %388 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
      %390 = stablehlo.multiply %339, %389 : tensor<64x128x16384xf32>
      %391 = stablehlo.select %336, %331, %390 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
      %392 = stablehlo.subtract %333, %391 : tensor<64x128x16384xf32>
      %393 = stablehlo.select %332, %392, %391 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
      %394 = stablehlo.select %308, %330, %393 : tensor<64x128x16384xi1>, tensor<64x128x16384xf32>
      %395 = stablehlo.multiply %301, %394 : tensor<64x128x16384xf32>
      mhlo.return %395 : tensor<64x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x128x128xf32>, tensor<128xf32>) -> tensor<64x128x16384xf32>
    %69 = "mhlo.fusion"(%68, %arg58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x128x16384xf32>, %arg209: tensor<32x128xf32>):
      %297 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x128xf32>) -> tensor<1048576x128xf32>
      %299 = stablehlo.dot_general %298, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x128xf32>, tensor<32x128xf32>) -> tensor<1048576x32xf32>
      mhlo.return %299 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x16384xf32>, tensor<32x128xf32>) -> tensor<1048576x32xf32>
    %70:2 = "mhlo.fusion"(%arg185, %arg184, %69, %arg59, %37, %60, %arg51) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1048576x32xf32>, %arg211: tensor<32xf32>, %arg212: tensor<64x16384x32xf32>, %arg213: tensor<1048576x32xf32>, %arg214: tensor<32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %298 = stablehlo.add %297, %arg210 : tensor<1048576x32xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %300 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %301 = stablehlo.add %300, %arg213 : tensor<1048576x32xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %303 = stablehlo.add %302, %arg212 : tensor<64x16384x32xf32>
      %304 = stablehlo.add %299, %303 : tensor<64x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<64x16384xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %309 = stablehlo.subtract %304, %308 : tensor<64x16384x32xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<64x16384x32xf32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %312 = stablehlo.multiply %311, %306 : tensor<64x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %314 = stablehlo.add %312, %313 : tensor<64x16384xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64x16384xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<64x16384x32xf32>
      %318 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<64x16384x32xf32>
      %320 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %321 = stablehlo.add %319, %320 : tensor<64x16384x32xf32>
      %322 = stablehlo.transpose %321, dims = [0, 2, 1] : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
      mhlo.return %322, %317 : tensor<64x32x16384xf32>, tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1048576x32xf32>, tensor<32xf32>, tensor<64x16384x32xf32>, tensor<1048576x32xf32>, tensor<32xf32>) -> (tensor<64x32x16384xf32>, tensor<64x16384x32xf32>)
    %71 = stablehlo.tuple %70#0, %70#1 {xla_shape = "(f32[64,32,16384]{2,1,0}, f32[64,16384,32]{2,1,0})"} : tuple<tensor<64x32x16384xf32>, tensor<64x16384x32xf32>>
    %72 = mhlo.bitcast %70#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x16384xf32>) -> tensor<64x32x128x128xf32>
    %73 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%72, %arg4, %arg5) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x32x128x128xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>) -> tuple<tensor<64x64x64x64xf32>, tensor<0xui8>>
    %74 = stablehlo.get_tuple_element %73[0] : (tuple<tensor<64x64x64x64xf32>, tensor<0xui8>>) -> tensor<64x64x64x64xf32>
    %75 = "mhlo.fusion"(%arg7, %arg6, %74) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x64x64x64xf32>):
      %297 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1] : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x64x64xf32>, tensor<f32>) -> tensor<64x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<64x64x64xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x64xf32>) -> tensor<64x4096xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %304 = stablehlo.subtract %298, %303 : tensor<64x4096x64xf32>
      %305 = stablehlo.multiply %304, %304 : tensor<64x4096x64xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<64x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %310 = stablehlo.add %308, %309 : tensor<64x4096xf32>
      %311 = stablehlo.rsqrt %310 : tensor<64x4096xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %313 = stablehlo.multiply %304, %312 : tensor<64x4096x64xf32>
      %314 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<64x4096x64xf32>
      %316 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %317 = stablehlo.add %315, %316 : tensor<64x4096x64xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %319 = stablehlo.multiply %318, %307 : tensor<64x4096xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<64x4096x64xf32>
      %322 = stablehlo.multiply %321, %321 : tensor<64x4096x64xf32>
      %323 = stablehlo.reduce(%322 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %324 = stablehlo.multiply %323, %307 : tensor<64x4096xf32>
      %325 = stablehlo.add %324, %309 : tensor<64x4096xf32>
      %326 = stablehlo.rsqrt %325 : tensor<64x4096xf32>
      %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %328 = stablehlo.multiply %321, %327 : tensor<64x4096x64xf32>
      mhlo.return %328 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x64x64x64xf32>) -> tensor<64x4096x64xf32>
    %76 = mhlo.bitcast %75 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
    %77 = "mhlo.fusion"(%76, %arg60, %arg61, %arg62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<262144x64xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %300 = stablehlo.add %298, %299 : tensor<262144x64xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
      mhlo.return %301 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %78 = "mhlo.fusion"(%77, %arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<262144x64xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x2x32xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1, 3] : (tensor<64x4096x2x32xf32>) -> tensor<64x2x4096x32xf32>
      mhlo.return %300 : tensor<64x2x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>) -> tensor<64x2x4096x32xf32>
    %79 = mhlo.bitcast %78 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x32xf32>) -> tensor<128x4096x32xf32>
    %80 = "mhlo.fusion"(%arg61, %75, %arg60) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64x4096x64xf32>, %arg210: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %298 = stablehlo.multiply %arg209, %297 : tensor<64x4096x64xf32>
      %299 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x4096x64xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
      mhlo.return %301 : tensor<64x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64x4096x64xf32>, tensor<64xf32>) -> tensor<64x64x4096xf32>
    %81 = mhlo.bitcast %80 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x4096xf32>) -> tensor<64x64x64x64xf32>
    %82 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%81, %arg68, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x64x64xf32>, tensor<64x64x4x4xf32>, tensor<64xf32>) -> tuple<tensor<64x64x16x16xf32>, tensor<0xui8>>
    %83 = stablehlo.get_tuple_element %82[0] : (tuple<tensor<64x64x16x16xf32>, tensor<0xui8>>) -> tensor<64x64x16x16xf32>
    %84 = "mhlo.fusion"(%83) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x64x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      mhlo.return %297 : tensor<64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x16x16xf32>) -> tensor<64x16x16xf32>
    %85 = "mhlo.fusion"(%83, %84) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x64x16x16xf32>, %arg209: tensor<64x16x16xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %299 = stablehlo.multiply %arg209, %298 : tensor<64x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
      %302 = stablehlo.subtract %297, %301 : tensor<64x64x256xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<64x64x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %304 = stablehlo.reduce(%303 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<64x64x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %304 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x16x16xf32>, tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %86 = "mhlo.fusion"(%83, %85, %84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x16x16xf32>, %arg209: tensor<64x256xf32>, %arg210: tensor<64x16x16xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %299 = stablehlo.multiply %arg210, %298 : tensor<64x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
      %302 = stablehlo.subtract %297, %301 : tensor<64x64x256xf32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %304 = stablehlo.multiply %arg209, %303 : tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<64x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<64x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
      %309 = stablehlo.multiply %302, %308 : tensor<64x64x256xf32>
      mhlo.return %309 : tensor<64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x16x16xf32>, tensor<64x256xf32>, tensor<64x16x16xf32>) -> tensor<64x64x256xf32>
    %87 = "mhlo.fusion"(%86, %arg70, %arg71, %arg64, %arg66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x64x256xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>, %arg212: tensor<64x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<64x64x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x64x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<64x64x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x64x256xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x64xf32>) -> tensor<16384x64xf32>
      %303 = stablehlo.concatenate %arg211, %arg212, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
      %305 = stablehlo.dot %302, %304, precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x128xf32>) -> tensor<16384x128xf32>
      mhlo.return %305 : tensor<16384x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<16384x128xf32>
    %88 = "mhlo.fusion"(%87, %arg65) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 0:64] : (tensor<16384x128xf32>) -> tensor<16384x64xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x64xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 3, 1] : (tensor<64x256x2x32xf32>) -> tensor<64x2x32x256xf32>
      mhlo.return %301 : tensor<64x2x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x128xf32>, tensor<64xf32>) -> tensor<64x2x32x256xf32>
    %89 = mhlo.bitcast %88 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x32x256xf32>) -> tensor<128x32x256xf32>
    %90 = "mhlo.fusion"(%79, %89) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x32xf32>, %arg209: tensor<128x32x256xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x32xf32>, tensor<128x32x256xf32>) -> tensor<128x4096x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<128x4096x256xf32>
      mhlo.return %299 : tensor<128x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x32xf32>, tensor<128x32x256xf32>) -> tensor<128x4096x256xf32>
    %91 = "mhlo.fusion"(%90) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x256xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %298 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x4096x256xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x4096xf32>) -> tensor<64x2x4096xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
      %301 = stablehlo.subtract %297, %300 : tensor<64x2x4096x256xf32>
      %302 = stablehlo.exponential %301 : tensor<64x2x4096x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
      %305 = stablehlo.divide %302, %304 : tensor<64x2x4096x256xf32>
      mhlo.return %305 : tensor<64x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %92 = mhlo.bitcast %91 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x256xf32>) -> tensor<128x4096x256xf32>
    %93 = "mhlo.fusion"(%87, %arg67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 64:128] : (tensor<16384x128xf32>) -> tensor<16384x64xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x64xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x256x2x32xf32>) -> tensor<64x2x256x32xf32>
      mhlo.return %301 : tensor<64x2x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x128xf32>, tensor<64xf32>) -> tensor<64x2x256x32xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x256x32xf32>) -> tensor<128x256x32xf32>
    %95 = "mhlo.fusion"(%92, %94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x256xf32>, %arg209: tensor<128x256x32xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x256xf32>, tensor<128x256x32xf32>) -> tensor<128x4096x32xf32>
      mhlo.return %297 : tensor<128x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x256xf32>, tensor<128x256x32xf32>) -> tensor<128x4096x32xf32>
    %96 = "mhlo.fusion"(%95) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x4096x32xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x32xf32>) -> tensor<64x2x4096x32xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] : (tensor<64x2x4096x32xf32>) -> tensor<64x4096x2x32xf32>
      mhlo.return %298 : tensor<64x4096x2x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x32xf32>) -> tensor<64x4096x2x32xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x4096x2x32xf32>) -> tensor<262144x64xf32>
    %98 = "mhlo.fusion"(%97, %arg72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64x64xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
      mhlo.return %297 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %99 = "mhlo.fusion"(%arg7, %arg6, %74, %98, %arg73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x64x64x64xf32>, %arg211: tensor<262144x64xf32>, %arg212: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.add %297, %arg211 : tensor<262144x64xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %300 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x64x64xf32>, tensor<f32>) -> tensor<64x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x64x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x64xf32>) -> tensor<64x4096xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %307 = stablehlo.subtract %301, %306 : tensor<64x4096x64xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<64x4096x64xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x4096xf32>
      %314 = stablehlo.rsqrt %313 : tensor<64x4096xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %316 = stablehlo.multiply %307, %315 : tensor<64x4096x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<64x4096x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<64x4096x64xf32>
      %321 = stablehlo.add %299, %320 : tensor<64x4096x64xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x4096xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %325 = stablehlo.subtract %321, %324 : tensor<64x4096x64xf32>
      %326 = stablehlo.multiply %325, %325 : tensor<64x4096x64xf32>
      %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %328 = stablehlo.multiply %327, %310 : tensor<64x4096xf32>
      %329 = stablehlo.add %328, %312 : tensor<64x4096xf32>
      %330 = stablehlo.rsqrt %329 : tensor<64x4096xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %332 = stablehlo.multiply %325, %331 : tensor<64x4096x64xf32>
      mhlo.return %332 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x64x64x64xf32>, tensor<262144x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %100 = mhlo.bitcast %99 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
    %101 = "mhlo.fusion"(%100, %arg74, %arg75, %arg76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<262144x64xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %300 = stablehlo.add %298, %299 : tensor<262144x64xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<256x64xf32>) -> tensor<262144x256xf32>
      mhlo.return %301 : tensor<262144x256xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<262144x256xf32>
    %102 = "mhlo.fusion"(%101, %arg77) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<262144x256xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<262144x256xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<262144x256xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<64x4096x256xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
      mhlo.return %300 : tensor<64x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x256xf32>, tensor<256xf32>) -> tensor<64x256x4096xf32>
    %103 = mhlo.bitcast %102 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x4096xf32>) -> tensor<64x256x64x64xf32>
    %104 = stablehlo.custom_call @__cudnn$convForward(%103, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>
    %105 = stablehlo.get_tuple_element %104[0] : (tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>) -> tensor<64x256x64x64xf32>
    %106 = "mhlo.fusion"(%105, %arg79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x64x64xf32>, %arg209: tensor<256xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
      %298 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %299 = stablehlo.add %arg208, %298 : tensor<64x256x64x64xf32>
      %300 = stablehlo.multiply %297, %299 : tensor<64x256x64x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
      %302 = stablehlo.negate %299 : tensor<64x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x256x64x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
      %306 = stablehlo.abs %305 : tensor<64x256x4096xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %308 = stablehlo.compare LT, %306, %307 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %310 = stablehlo.multiply %305, %305 : tensor<64x256x4096xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x256x4096xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x256x4096xf32>
      %314 = stablehlo.multiply %313, %310 : tensor<64x256x4096xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x256x4096xf32>
      %317 = stablehlo.multiply %316, %310 : tensor<64x256x4096xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %319 = stablehlo.add %317, %318 : tensor<64x256x4096xf32>
      %320 = stablehlo.multiply %319, %310 : tensor<64x256x4096xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %322 = stablehlo.add %320, %321 : tensor<64x256x4096xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x256x4096xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %325 = stablehlo.add %323, %324 : tensor<64x256x4096xf32>
      %326 = stablehlo.multiply %325, %310 : tensor<64x256x4096xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %328 = stablehlo.add %326, %327 : tensor<64x256x4096xf32>
      %329 = stablehlo.multiply %305, %328 : tensor<64x256x4096xf32>
      %330 = stablehlo.subtract %307, %329 : tensor<64x256x4096xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %332 = stablehlo.compare LT, %305, %331 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %334 = stablehlo.negate %310 : tensor<64x256x4096xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %336 = stablehlo.compare LT, %334, %335 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
      %337 = stablehlo.exponential %334 : tensor<64x256x4096xf32>
      %338 = stablehlo.divide %307, %306 : tensor<64x256x4096xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x4096xf32>
      %340 = stablehlo.compare LT, %306, %333 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %342 = stablehlo.divide %307, %310 : tensor<64x256x4096xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<64x256x4096xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %345 = stablehlo.add %343, %344 : tensor<64x256x4096xf32>
      %346 = stablehlo.multiply %345, %342 : tensor<64x256x4096xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %348 = stablehlo.add %346, %347 : tensor<64x256x4096xf32>
      %349 = stablehlo.multiply %348, %342 : tensor<64x256x4096xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %351 = stablehlo.add %349, %350 : tensor<64x256x4096xf32>
      %352 = stablehlo.multiply %351, %342 : tensor<64x256x4096xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %354 = stablehlo.add %352, %353 : tensor<64x256x4096xf32>
      %355 = stablehlo.multiply %354, %342 : tensor<64x256x4096xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %357 = stablehlo.add %355, %356 : tensor<64x256x4096xf32>
      %358 = stablehlo.multiply %357, %342 : tensor<64x256x4096xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %360 = stablehlo.add %358, %359 : tensor<64x256x4096xf32>
      %361 = stablehlo.multiply %360, %342 : tensor<64x256x4096xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %363 = stablehlo.add %361, %362 : tensor<64x256x4096xf32>
      %364 = stablehlo.multiply %363, %342 : tensor<64x256x4096xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %366 = stablehlo.add %364, %365 : tensor<64x256x4096xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %368 = stablehlo.multiply %367, %342 : tensor<64x256x4096xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %370 = stablehlo.add %368, %369 : tensor<64x256x4096xf32>
      %371 = stablehlo.multiply %370, %342 : tensor<64x256x4096xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %372 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %373 = stablehlo.add %371, %372 : tensor<64x256x4096xf32>
      %374 = stablehlo.multiply %373, %342 : tensor<64x256x4096xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %376 = stablehlo.add %374, %375 : tensor<64x256x4096xf32>
      %377 = stablehlo.multiply %376, %342 : tensor<64x256x4096xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x256x4096xf32>
      %380 = stablehlo.multiply %379, %342 : tensor<64x256x4096xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %382 = stablehlo.add %380, %381 : tensor<64x256x4096xf32>
      %383 = stablehlo.multiply %382, %342 : tensor<64x256x4096xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %385 = stablehlo.add %383, %384 : tensor<64x256x4096xf32>
      %386 = stablehlo.multiply %385, %342 : tensor<64x256x4096xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %388 = stablehlo.add %386, %387 : tensor<64x256x4096xf32>
      %389 = stablehlo.select %340, %366, %388 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
      %390 = stablehlo.multiply %339, %389 : tensor<64x256x4096xf32>
      %391 = stablehlo.select %336, %331, %390 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
      %392 = stablehlo.subtract %333, %391 : tensor<64x256x4096xf32>
      %393 = stablehlo.select %332, %392, %391 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
      %394 = stablehlo.select %308, %330, %393 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
      %395 = stablehlo.multiply %301, %394 : tensor<64x256x4096xf32>
      mhlo.return %395 : tensor<64x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x64x64xf32>, tensor<256xf32>) -> tensor<64x256x4096xf32>
    %107 = "mhlo.fusion"(%106, %arg80) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x4096xf32>, %arg209: tensor<64x256xf32>):
      %297 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x256xf32>) -> tensor<262144x256xf32>
      %299 = stablehlo.dot_general %298, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x256xf32>, tensor<64x256xf32>) -> tensor<262144x64xf32>
      mhlo.return %299 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4096xf32>, tensor<64x256xf32>) -> tensor<262144x64xf32>
    %108 = "mhlo.fusion"(%107, %arg81, %arg7, %arg6, %74, %98, %arg73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64xf32>, %arg212: tensor<64x64x64x64xf32>, %arg213: tensor<262144x64xf32>, %arg214: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<262144x64xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %301 = stablehlo.add %300, %arg213 : tensor<262144x64xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %303 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x64x64xf32>, tensor<f32>) -> tensor<64x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64x64xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<64x64x64xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x64xf32>) -> tensor<64x4096xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %310 = stablehlo.subtract %304, %309 : tensor<64x4096x64xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<64x4096x64xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %314 = stablehlo.multiply %312, %313 : tensor<64x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x4096xf32>
      %317 = stablehlo.rsqrt %316 : tensor<64x4096xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %319 = stablehlo.multiply %310, %318 : tensor<64x4096x64xf32>
      %320 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<64x4096x64xf32>
      %322 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %323 = stablehlo.add %321, %322 : tensor<64x4096x64xf32>
      %324 = stablehlo.add %302, %323 : tensor<64x4096x64xf32>
      %325 = stablehlo.add %299, %324 : tensor<64x4096x64xf32>
      mhlo.return %325 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x64x64xf32>, tensor<262144x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<64x4096xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %301 = stablehlo.subtract %arg208, %300 : tensor<64x4096x64xf32>
      %302 = stablehlo.multiply %301, %301 : tensor<64x4096x64xf32>
      %303 = stablehlo.reduce(%302 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %304 = stablehlo.multiply %303, %298 : tensor<64x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %306 = stablehlo.add %304, %305 : tensor<64x4096xf32>
      %307 = stablehlo.rsqrt %306 : tensor<64x4096xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %309 = stablehlo.multiply %301, %308 : tensor<64x4096x64xf32>
      mhlo.return %309 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>) -> tensor<64x4096x64xf32>
    %110 = mhlo.bitcast %109 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
    %111 = "mhlo.fusion"(%110, %arg82, %arg83, %arg84) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<262144x64xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %300 = stablehlo.add %298, %299 : tensor<262144x64xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
      mhlo.return %301 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %112 = "mhlo.fusion"(%111, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<262144x64xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x2x32xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1, 3] : (tensor<64x4096x2x32xf32>) -> tensor<64x2x4096x32xf32>
      mhlo.return %300 : tensor<64x2x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>) -> tensor<64x2x4096x32xf32>
    %113 = mhlo.bitcast %112 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x32xf32>) -> tensor<128x4096x32xf32>
    %114 = "mhlo.fusion"(%arg83, %109, %arg82) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64x4096x64xf32>, %arg210: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %298 = stablehlo.multiply %arg209, %297 : tensor<64x4096x64xf32>
      %299 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x4096x64xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
      mhlo.return %301 : tensor<64x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64x4096x64xf32>, tensor<64xf32>) -> tensor<64x64x4096xf32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x4096xf32>) -> tensor<64x64x64x64xf32>
    %116 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%115, %arg90, %arg91) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x64x64xf32>, tensor<64x64x4x4xf32>, tensor<64xf32>) -> tuple<tensor<64x64x16x16xf32>, tensor<0xui8>>
    %117 = stablehlo.get_tuple_element %116[0] : (tuple<tensor<64x64x16x16xf32>, tensor<0xui8>>) -> tensor<64x64x16x16xf32>
    %118 = "mhlo.fusion"(%117) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x64x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x64x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      mhlo.return %297 : tensor<64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x16x16xf32>) -> tensor<64x16x16xf32>
    %119 = "mhlo.fusion"(%117, %118) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x64x16x16xf32>, %arg209: tensor<64x16x16xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %299 = stablehlo.multiply %arg209, %298 : tensor<64x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
      %302 = stablehlo.subtract %297, %301 : tensor<64x64x256xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<64x64x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %304 = stablehlo.reduce(%303 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<64x64x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %304 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x16x16xf32>, tensor<64x16x16xf32>) -> tensor<64x256xf32>
    %120 = "mhlo.fusion"(%117, %119, %118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x16x16xf32>, %arg209: tensor<64x256xf32>, %arg210: tensor<64x16x16xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %299 = stablehlo.multiply %arg210, %298 : tensor<64x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
      %302 = stablehlo.subtract %297, %301 : tensor<64x64x256xf32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %304 = stablehlo.multiply %arg209, %303 : tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<64x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<64x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
      %309 = stablehlo.multiply %302, %308 : tensor<64x64x256xf32>
      mhlo.return %309 : tensor<64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x16x16xf32>, tensor<64x256xf32>, tensor<64x16x16xf32>) -> tensor<64x64x256xf32>
    %121 = "mhlo.fusion"(%120, %arg92, %arg93, %arg86, %arg88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x64x256xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>, %arg212: tensor<64x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<64x64x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x64x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<64x64x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x64x256xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x64xf32>) -> tensor<16384x64xf32>
      %303 = stablehlo.concatenate %arg211, %arg212, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
      %305 = stablehlo.dot %302, %304, precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x128xf32>) -> tensor<16384x128xf32>
      mhlo.return %305 : tensor<16384x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<16384x128xf32>
    %122 = "mhlo.fusion"(%121, %arg87) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 0:64] : (tensor<16384x128xf32>) -> tensor<16384x64xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x64xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 3, 1] : (tensor<64x256x2x32xf32>) -> tensor<64x2x32x256xf32>
      mhlo.return %301 : tensor<64x2x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x128xf32>, tensor<64xf32>) -> tensor<64x2x32x256xf32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x32x256xf32>) -> tensor<128x32x256xf32>
    %124 = "mhlo.fusion"(%113, %123) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x32xf32>, %arg209: tensor<128x32x256xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x32xf32>, tensor<128x32x256xf32>) -> tensor<128x4096x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<128x4096x256xf32>
      mhlo.return %299 : tensor<128x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x32xf32>, tensor<128x32x256xf32>) -> tensor<128x4096x256xf32>
    %125 = "mhlo.fusion"(%124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x256xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %298 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x4096x256xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x4096xf32>) -> tensor<64x2x4096xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
      %301 = stablehlo.subtract %297, %300 : tensor<64x2x4096x256xf32>
      %302 = stablehlo.exponential %301 : tensor<64x2x4096x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
      %305 = stablehlo.divide %302, %304 : tensor<64x2x4096x256xf32>
      mhlo.return %305 : tensor<64x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %126 = mhlo.bitcast %125 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x256xf32>) -> tensor<128x4096x256xf32>
    %127 = "mhlo.fusion"(%121, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 64:128] : (tensor<16384x128xf32>) -> tensor<16384x64xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x64xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x256x2x32xf32>) -> tensor<64x2x256x32xf32>
      mhlo.return %301 : tensor<64x2x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x128xf32>, tensor<64xf32>) -> tensor<64x2x256x32xf32>
    %128 = mhlo.bitcast %127 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x256x32xf32>) -> tensor<128x256x32xf32>
    %129 = "mhlo.fusion"(%126, %128) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x256xf32>, %arg209: tensor<128x256x32xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x256xf32>, tensor<128x256x32xf32>) -> tensor<128x4096x32xf32>
      mhlo.return %297 : tensor<128x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x256xf32>, tensor<128x256x32xf32>) -> tensor<128x4096x32xf32>
    %130 = "mhlo.fusion"(%129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x4096x32xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x32xf32>) -> tensor<64x2x4096x32xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] : (tensor<64x2x4096x32xf32>) -> tensor<64x4096x2x32xf32>
      mhlo.return %298 : tensor<64x4096x2x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x32xf32>) -> tensor<64x4096x2x32xf32>
    %131 = mhlo.bitcast %130 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x4096x2x32xf32>) -> tensor<262144x64xf32>
    %132 = "mhlo.fusion"(%131, %arg94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64x64xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
      mhlo.return %297 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %133 = "mhlo.fusion"(%108, %132, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<262144x64xf32>, %arg210: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.add %297, %arg209 : tensor<262144x64xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %300 = stablehlo.add %299, %arg208 : tensor<64x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%300 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<64x4096xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %305 = stablehlo.subtract %300, %304 : tensor<64x4096x64xf32>
      %306 = stablehlo.multiply %305, %305 : tensor<64x4096x64xf32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %308 = stablehlo.multiply %307, %302 : tensor<64x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %310 = stablehlo.add %308, %309 : tensor<64x4096xf32>
      %311 = stablehlo.rsqrt %310 : tensor<64x4096xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %313 = stablehlo.multiply %305, %312 : tensor<64x4096x64xf32>
      mhlo.return %313 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<262144x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %134 = mhlo.bitcast %133 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
    %135 = "mhlo.fusion"(%134, %arg96, %arg97, %arg98) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<262144x64xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %300 = stablehlo.add %298, %299 : tensor<262144x64xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<256x64xf32>) -> tensor<262144x256xf32>
      mhlo.return %301 : tensor<262144x256xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<262144x256xf32>
    %136 = "mhlo.fusion"(%135, %arg99) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<262144x256xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<262144x256xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<262144x256xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<64x4096x256xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
      mhlo.return %300 : tensor<64x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x256xf32>, tensor<256xf32>) -> tensor<64x256x4096xf32>
    %137 = mhlo.bitcast %136 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x4096xf32>) -> tensor<64x256x64x64xf32>
    %138 = stablehlo.custom_call @__cudnn$convForward(%137, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>
    %139 = stablehlo.get_tuple_element %138[0] : (tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>) -> tensor<64x256x64x64xf32>
    %140 = "mhlo.fusion"(%139, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x64x64xf32>, %arg209: tensor<256xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
      %298 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %299 = stablehlo.add %arg208, %298 : tensor<64x256x64x64xf32>
      %300 = stablehlo.multiply %297, %299 : tensor<64x256x64x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
      %302 = stablehlo.negate %299 : tensor<64x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x256x64x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
      %306 = stablehlo.abs %305 : tensor<64x256x4096xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %308 = stablehlo.compare LT, %306, %307 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %310 = stablehlo.multiply %305, %305 : tensor<64x256x4096xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x256x4096xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x256x4096xf32>
      %314 = stablehlo.multiply %313, %310 : tensor<64x256x4096xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x256x4096xf32>
      %317 = stablehlo.multiply %316, %310 : tensor<64x256x4096xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %319 = stablehlo.add %317, %318 : tensor<64x256x4096xf32>
      %320 = stablehlo.multiply %319, %310 : tensor<64x256x4096xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %322 = stablehlo.add %320, %321 : tensor<64x256x4096xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x256x4096xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %325 = stablehlo.add %323, %324 : tensor<64x256x4096xf32>
      %326 = stablehlo.multiply %325, %310 : tensor<64x256x4096xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %328 = stablehlo.add %326, %327 : tensor<64x256x4096xf32>
      %329 = stablehlo.multiply %305, %328 : tensor<64x256x4096xf32>
      %330 = stablehlo.subtract %307, %329 : tensor<64x256x4096xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %332 = stablehlo.compare LT, %305, %331 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %334 = stablehlo.negate %310 : tensor<64x256x4096xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %336 = stablehlo.compare LT, %334, %335 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
      %337 = stablehlo.exponential %334 : tensor<64x256x4096xf32>
      %338 = stablehlo.divide %307, %306 : tensor<64x256x4096xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x4096xf32>
      %340 = stablehlo.compare LT, %306, %333 : (tensor<64x256x4096xf32>, tensor<64x256x4096xf32>) -> tensor<64x256x4096xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %342 = stablehlo.divide %307, %310 : tensor<64x256x4096xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<64x256x4096xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %345 = stablehlo.add %343, %344 : tensor<64x256x4096xf32>
      %346 = stablehlo.multiply %345, %342 : tensor<64x256x4096xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %348 = stablehlo.add %346, %347 : tensor<64x256x4096xf32>
      %349 = stablehlo.multiply %348, %342 : tensor<64x256x4096xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %351 = stablehlo.add %349, %350 : tensor<64x256x4096xf32>
      %352 = stablehlo.multiply %351, %342 : tensor<64x256x4096xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %354 = stablehlo.add %352, %353 : tensor<64x256x4096xf32>
      %355 = stablehlo.multiply %354, %342 : tensor<64x256x4096xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %357 = stablehlo.add %355, %356 : tensor<64x256x4096xf32>
      %358 = stablehlo.multiply %357, %342 : tensor<64x256x4096xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %360 = stablehlo.add %358, %359 : tensor<64x256x4096xf32>
      %361 = stablehlo.multiply %360, %342 : tensor<64x256x4096xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %363 = stablehlo.add %361, %362 : tensor<64x256x4096xf32>
      %364 = stablehlo.multiply %363, %342 : tensor<64x256x4096xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %366 = stablehlo.add %364, %365 : tensor<64x256x4096xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %368 = stablehlo.multiply %367, %342 : tensor<64x256x4096xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %370 = stablehlo.add %368, %369 : tensor<64x256x4096xf32>
      %371 = stablehlo.multiply %370, %342 : tensor<64x256x4096xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %372 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %373 = stablehlo.add %371, %372 : tensor<64x256x4096xf32>
      %374 = stablehlo.multiply %373, %342 : tensor<64x256x4096xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %376 = stablehlo.add %374, %375 : tensor<64x256x4096xf32>
      %377 = stablehlo.multiply %376, %342 : tensor<64x256x4096xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x256x4096xf32>
      %380 = stablehlo.multiply %379, %342 : tensor<64x256x4096xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %382 = stablehlo.add %380, %381 : tensor<64x256x4096xf32>
      %383 = stablehlo.multiply %382, %342 : tensor<64x256x4096xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %385 = stablehlo.add %383, %384 : tensor<64x256x4096xf32>
      %386 = stablehlo.multiply %385, %342 : tensor<64x256x4096xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x256x4096xf32>
      %388 = stablehlo.add %386, %387 : tensor<64x256x4096xf32>
      %389 = stablehlo.select %340, %366, %388 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
      %390 = stablehlo.multiply %339, %389 : tensor<64x256x4096xf32>
      %391 = stablehlo.select %336, %331, %390 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
      %392 = stablehlo.subtract %333, %391 : tensor<64x256x4096xf32>
      %393 = stablehlo.select %332, %392, %391 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
      %394 = stablehlo.select %308, %330, %393 : tensor<64x256x4096xi1>, tensor<64x256x4096xf32>
      %395 = stablehlo.multiply %301, %394 : tensor<64x256x4096xf32>
      mhlo.return %395 : tensor<64x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x64x64xf32>, tensor<256xf32>) -> tensor<64x256x4096xf32>
    %141 = "mhlo.fusion"(%140, %arg102) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x4096xf32>, %arg209: tensor<64x256xf32>):
      %297 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x256xf32>) -> tensor<262144x256xf32>
      %299 = stablehlo.dot_general %298, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x256xf32>, tensor<64x256xf32>) -> tensor<262144x64xf32>
      mhlo.return %299 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4096xf32>, tensor<64x256xf32>) -> tensor<262144x64xf32>
    %142:2 = "mhlo.fusion"(%arg187, %arg186, %141, %arg103, %108, %132, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<262144x64xf32>, %arg211: tensor<64xf32>, %arg212: tensor<64x4096x64xf32>, %arg213: tensor<262144x64xf32>, %arg214: tensor<64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %298 = stablehlo.add %297, %arg210 : tensor<262144x64xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %301 = stablehlo.add %300, %arg213 : tensor<262144x64xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %303 = stablehlo.add %302, %arg212 : tensor<64x4096x64xf32>
      %304 = stablehlo.add %299, %303 : tensor<64x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<64x4096xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %309 = stablehlo.subtract %304, %308 : tensor<64x4096x64xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<64x4096x64xf32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      %312 = stablehlo.multiply %311, %306 : tensor<64x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %314 = stablehlo.add %312, %313 : tensor<64x4096xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64x4096xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<64x4096x64xf32>
      %318 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<64x4096x64xf32>
      %320 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %321 = stablehlo.add %319, %320 : tensor<64x4096x64xf32>
      %322 = stablehlo.transpose %321, dims = [0, 2, 1] : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
      mhlo.return %322, %317 : tensor<64x64x4096xf32>, tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<262144x64xf32>, tensor<64xf32>, tensor<64x4096x64xf32>, tensor<262144x64xf32>, tensor<64xf32>) -> (tensor<64x64x4096xf32>, tensor<64x4096x64xf32>)
    %143 = stablehlo.tuple %142#0, %142#1 {xla_shape = "(f32[64,64,4096]{2,1,0}, f32[64,4096,64]{2,1,0})"} : tuple<tensor<64x64x4096xf32>, tensor<64x4096x64xf32>>
    %144 = mhlo.bitcast %142#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x4096xf32>) -> tensor<64x64x64x64xf32>
    %145 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%144, %arg8, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,160,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x64x64xf32>, tensor<160x64x3x3xf32>, tensor<160xf32>) -> tuple<tensor<64x160x32x32xf32>, tensor<0xui8>>
    %146 = stablehlo.get_tuple_element %145[0] : (tuple<tensor<64x160x32x32xf32>, tensor<0xui8>>) -> tensor<64x160x32x32xf32>
    %147 = "mhlo.fusion"(%arg11, %arg10, %146) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x160x32x32xf32>):
      %297 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1] : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x32x32xf32>, tensor<f32>) -> tensor<64x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x32x32xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<64x32x32xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x32xf32>) -> tensor<64x1024xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %304 = stablehlo.subtract %298, %303 : tensor<64x1024x160xf32>
      %305 = stablehlo.multiply %304, %304 : tensor<64x1024x160xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<64x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %310 = stablehlo.add %308, %309 : tensor<64x1024xf32>
      %311 = stablehlo.rsqrt %310 : tensor<64x1024xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %313 = stablehlo.multiply %304, %312 : tensor<64x1024x160xf32>
      %314 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<64x1024x160xf32>
      %316 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %317 = stablehlo.add %315, %316 : tensor<64x1024x160xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %319 = stablehlo.multiply %318, %307 : tensor<64x1024xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<64x1024x160xf32>
      %322 = stablehlo.multiply %321, %321 : tensor<64x1024x160xf32>
      %323 = stablehlo.reduce(%322 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %324 = stablehlo.multiply %323, %307 : tensor<64x1024xf32>
      %325 = stablehlo.add %324, %309 : tensor<64x1024xf32>
      %326 = stablehlo.rsqrt %325 : tensor<64x1024xf32>
      %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %328 = stablehlo.multiply %321, %327 : tensor<64x1024x160xf32>
      mhlo.return %328 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x160x32x32xf32>) -> tensor<64x1024x160xf32>
    %148 = mhlo.bitcast %147 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
    %149 = "mhlo.fusion"(%148, %arg104, %arg105, %arg106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<65536x160xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %300 = stablehlo.add %298, %299 : tensor<65536x160xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %301 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %150 = "mhlo.fusion"(%149, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<65536x160xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x5x32xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1, 3] : (tensor<64x1024x5x32xf32>) -> tensor<64x5x1024x32xf32>
      mhlo.return %300 : tensor<64x5x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>) -> tensor<64x5x1024x32xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x32xf32>) -> tensor<320x1024x32xf32>
    %152 = "mhlo.fusion"(%arg105, %147, %arg104) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<64x1024x160xf32>, %arg210: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %298 = stablehlo.multiply %arg209, %297 : tensor<64x1024x160xf32>
      %299 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x1024x160xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
      mhlo.return %301 : tensor<64x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<64x1024x160xf32>, tensor<160xf32>) -> tensor<64x160x1024xf32>
    %153 = mhlo.bitcast %152 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x160x1024xf32>) -> tensor<64x160x32x32xf32>
    %154 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%153, %arg112, %arg113) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,160,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x160x32x32xf32>, tensor<160x160x2x2xf32>, tensor<160xf32>) -> tuple<tensor<64x160x16x16xf32>, tensor<0xui8>>
    %155 = stablehlo.get_tuple_element %154[0] : (tuple<tensor<64x160x16x16xf32>, tensor<0xui8>>) -> tensor<64x160x16x16xf32>
    %156 = "mhlo.fusion"(%155) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x160x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      mhlo.return %297 : tensor<64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x160x16x16xf32>) -> tensor<64x16x16xf32>
    %157 = "mhlo.fusion"(%156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x16x16xf32>
      mhlo.return %298 : tensor<64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16xf32>) -> tensor<64x16x16xf32>
    %158 = "mhlo.fusion"(%157, %155) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x16x16xf32>, %arg209: tensor<64x160x16x16xf32>):
      %297 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %298 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
      %300 = stablehlo.subtract %297, %299 : tensor<64x160x256xf32>
      %301 = stablehlo.multiply %300, %300 : tensor<64x160x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %302 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16xf32>, tensor<64x160x16x16xf32>) -> tensor<64x256xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x256xf32>
      %301 = stablehlo.rsqrt %300 : tensor<64x256xf32>
      mhlo.return %301 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %160 = "mhlo.fusion"(%159, %157, %155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16xf32>, %arg210: tensor<64x160x16x16xf32>):
      %297 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %298 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
      %300 = stablehlo.subtract %297, %299 : tensor<64x160x256xf32>
      %301 = stablehlo.broadcast_in_dim %arg208, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<64x160x256xf32>
      mhlo.return %302 : tensor<64x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16xf32>, tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %161 = "mhlo.fusion"(%arg108, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160xf32>, %arg209: tensor<160x160xf32>):
      %297 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
      mhlo.return %297 : tensor<320x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    %162 = mhlo.bitcast %161 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[160,320]{0,1}"} : (tensor<320x160xf32>) -> tensor<160x320xf32>
    %163 = "mhlo.fusion"(%160, %arg114, %arg115, %162) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x160x256xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x320xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<64x160x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x160x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<64x160x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x160x256xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x160xf32>) -> tensor<16384x160xf32>
      %303 = stablehlo.dot %302, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x320xf32>) -> tensor<16384x320xf32>
      mhlo.return %303 : tensor<16384x320xf32>
    }) {output_operand_aliasing = []} : (tensor<64x160x256xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x320xf32>) -> tensor<16384x320xf32>
    %164 = "mhlo.fusion"(%163, %arg109) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384x320xf32>, %arg209: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 0:160] : (tensor<16384x320xf32>) -> tensor<16384x160xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x160xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 3, 1] : (tensor<64x256x5x32xf32>) -> tensor<64x5x32x256xf32>
      mhlo.return %301 : tensor<64x5x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x320xf32>, tensor<160xf32>) -> tensor<64x5x32x256xf32>
    %165 = mhlo.bitcast %164 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x32x256xf32>) -> tensor<320x32x256xf32>
    %166 = "mhlo.fusion"(%151, %165) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<320x1024x32xf32>, %arg209: tensor<320x32x256xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x32xf32>, tensor<320x32x256xf32>) -> tensor<320x1024x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<320x1024x256xf32>
      mhlo.return %299 : tensor<320x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x32xf32>, tensor<320x32x256xf32>) -> tensor<320x1024x256xf32>
    %167 = "mhlo.fusion"(%166) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<320x1024x256xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %298 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<320x1024x256xf32>, tensor<f32>) -> tensor<320x1024xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<320x1024xf32>) -> tensor<64x5x1024xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
      %301 = stablehlo.subtract %297, %300 : tensor<64x5x1024x256xf32>
      %302 = stablehlo.exponential %301 : tensor<64x5x1024x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
      %305 = stablehlo.divide %302, %304 : tensor<64x5x1024x256xf32>
      mhlo.return %305 : tensor<64x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %168 = mhlo.bitcast %167 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x256xf32>) -> tensor<320x1024x256xf32>
    %169 = "mhlo.fusion"(%163, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x320xf32>, %arg209: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 160:320] : (tensor<16384x320xf32>) -> tensor<16384x160xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x160xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x256x5x32xf32>) -> tensor<64x5x256x32xf32>
      mhlo.return %301 : tensor<64x5x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x320xf32>, tensor<160xf32>) -> tensor<64x5x256x32xf32>
    %170 = mhlo.bitcast %169 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x256x32xf32>) -> tensor<320x256x32xf32>
    %171 = "mhlo.fusion"(%168, %170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<320x1024x256xf32>, %arg209: tensor<320x256x32xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x256xf32>, tensor<320x256x32xf32>) -> tensor<320x1024x32xf32>
      mhlo.return %297 : tensor<320x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x256xf32>, tensor<320x256x32xf32>) -> tensor<320x1024x32xf32>
    %172 = "mhlo.fusion"(%171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<320x1024x32xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x32xf32>) -> tensor<64x5x1024x32xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] : (tensor<64x5x1024x32xf32>) -> tensor<64x1024x5x32xf32>
      mhlo.return %298 : tensor<64x1024x5x32xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x32xf32>) -> tensor<64x1024x5x32xf32>
    %173 = mhlo.bitcast %172 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x5x32xf32>) -> tensor<65536x160xf32>
    %174 = "mhlo.fusion"(%173, %arg116) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160x160xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %297 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %175 = "mhlo.fusion"(%arg11, %arg10, %146, %174, %arg117) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x160x32x32xf32>, %arg211: tensor<65536x160xf32>, %arg212: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.add %297, %arg211 : tensor<65536x160xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %300 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x32x32xf32>, tensor<f32>) -> tensor<64x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x32x32xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x32x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x32xf32>) -> tensor<64x1024xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %307 = stablehlo.subtract %301, %306 : tensor<64x1024x160xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<64x1024x160xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x1024xf32>
      %314 = stablehlo.rsqrt %313 : tensor<64x1024xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %316 = stablehlo.multiply %307, %315 : tensor<64x1024x160xf32>
      %317 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<64x1024x160xf32>
      %319 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %320 = stablehlo.add %318, %319 : tensor<64x1024x160xf32>
      %321 = stablehlo.add %299, %320 : tensor<64x1024x160xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x1024xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %325 = stablehlo.subtract %321, %324 : tensor<64x1024x160xf32>
      %326 = stablehlo.multiply %325, %325 : tensor<64x1024x160xf32>
      %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %328 = stablehlo.multiply %327, %310 : tensor<64x1024xf32>
      %329 = stablehlo.add %328, %312 : tensor<64x1024xf32>
      %330 = stablehlo.rsqrt %329 : tensor<64x1024xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %332 = stablehlo.multiply %325, %331 : tensor<64x1024x160xf32>
      mhlo.return %332 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x160x32x32xf32>, tensor<65536x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %176 = mhlo.bitcast %175 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
    %177 = "mhlo.fusion"(%176, %arg118, %arg119, %arg120) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<640x160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<65536x160xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %300 = stablehlo.add %298, %299 : tensor<65536x160xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<640x160xf32>) -> tensor<65536x640xf32>
      mhlo.return %301 : tensor<65536x640xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<640x160xf32>) -> tensor<65536x640xf32>
    %178 = "mhlo.fusion"(%177, %arg121) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x640xf32>, %arg209: tensor<640xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<65536x640xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<65536x640xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x640xf32>) -> tensor<64x1024x640xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] : (tensor<64x1024x640xf32>) -> tensor<64x640x1024xf32>
      mhlo.return %300 : tensor<64x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x640xf32>, tensor<640xf32>) -> tensor<64x640x1024xf32>
    %179 = mhlo.bitcast %178 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x640x1024xf32>) -> tensor<64x640x32x32xf32>
    %180 = stablehlo.custom_call @__cudnn$convForward(%179, %arg122) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,640,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x640x32x32xf32>, tensor<640x1x3x3xf32>) -> tuple<tensor<64x640x32x32xf32>, tensor<0xui8>>
    %181 = stablehlo.get_tuple_element %180[0] : (tuple<tensor<64x640x32x32xf32>, tensor<0xui8>>) -> tensor<64x640x32x32xf32>
    %182 = "mhlo.fusion"(%181, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x640x32x32xf32>, %arg209: tensor<640xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x640x32x32xf32>
      %298 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<64x640x32x32xf32>
      %299 = stablehlo.add %arg208, %298 : tensor<64x640x32x32xf32>
      %300 = stablehlo.multiply %297, %299 : tensor<64x640x32x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
      %302 = stablehlo.negate %299 : tensor<64x640x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x640x32x32xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x640x32x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
      %306 = stablehlo.abs %305 : tensor<64x640x1024xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %308 = stablehlo.compare LT, %306, %307 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %310 = stablehlo.multiply %305, %305 : tensor<64x640x1024xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x640x1024xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x640x1024xf32>
      %314 = stablehlo.multiply %313, %310 : tensor<64x640x1024xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x640x1024xf32>
      %317 = stablehlo.multiply %316, %310 : tensor<64x640x1024xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %319 = stablehlo.add %317, %318 : tensor<64x640x1024xf32>
      %320 = stablehlo.multiply %319, %310 : tensor<64x640x1024xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %322 = stablehlo.add %320, %321 : tensor<64x640x1024xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x640x1024xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %325 = stablehlo.add %323, %324 : tensor<64x640x1024xf32>
      %326 = stablehlo.multiply %325, %310 : tensor<64x640x1024xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<64x640x1024xf32>
      %329 = stablehlo.multiply %305, %328 : tensor<64x640x1024xf32>
      %330 = stablehlo.subtract %307, %329 : tensor<64x640x1024xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %332 = stablehlo.compare LT, %305, %331 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %334 = stablehlo.negate %310 : tensor<64x640x1024xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %336 = stablehlo.compare LT, %334, %335 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
      %337 = stablehlo.exponential %334 : tensor<64x640x1024xf32>
      %338 = stablehlo.divide %307, %306 : tensor<64x640x1024xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x640x1024xf32>
      %340 = stablehlo.compare LT, %306, %333 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %342 = stablehlo.divide %307, %310 : tensor<64x640x1024xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<64x640x1024xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %345 = stablehlo.add %343, %344 : tensor<64x640x1024xf32>
      %346 = stablehlo.multiply %345, %342 : tensor<64x640x1024xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %348 = stablehlo.add %346, %347 : tensor<64x640x1024xf32>
      %349 = stablehlo.multiply %348, %342 : tensor<64x640x1024xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %351 = stablehlo.add %349, %350 : tensor<64x640x1024xf32>
      %352 = stablehlo.multiply %351, %342 : tensor<64x640x1024xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %354 = stablehlo.add %352, %353 : tensor<64x640x1024xf32>
      %355 = stablehlo.multiply %354, %342 : tensor<64x640x1024xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %357 = stablehlo.add %355, %356 : tensor<64x640x1024xf32>
      %358 = stablehlo.multiply %357, %342 : tensor<64x640x1024xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %360 = stablehlo.add %358, %359 : tensor<64x640x1024xf32>
      %361 = stablehlo.multiply %360, %342 : tensor<64x640x1024xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %363 = stablehlo.add %361, %362 : tensor<64x640x1024xf32>
      %364 = stablehlo.multiply %363, %342 : tensor<64x640x1024xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %366 = stablehlo.add %364, %365 : tensor<64x640x1024xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %368 = stablehlo.multiply %367, %342 : tensor<64x640x1024xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %370 = stablehlo.add %368, %369 : tensor<64x640x1024xf32>
      %371 = stablehlo.multiply %370, %342 : tensor<64x640x1024xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %372 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %373 = stablehlo.add %371, %372 : tensor<64x640x1024xf32>
      %374 = stablehlo.multiply %373, %342 : tensor<64x640x1024xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %376 = stablehlo.add %374, %375 : tensor<64x640x1024xf32>
      %377 = stablehlo.multiply %376, %342 : tensor<64x640x1024xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x640x1024xf32>
      %380 = stablehlo.multiply %379, %342 : tensor<64x640x1024xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %382 = stablehlo.add %380, %381 : tensor<64x640x1024xf32>
      %383 = stablehlo.multiply %382, %342 : tensor<64x640x1024xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %385 = stablehlo.add %383, %384 : tensor<64x640x1024xf32>
      %386 = stablehlo.multiply %385, %342 : tensor<64x640x1024xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %388 = stablehlo.add %386, %387 : tensor<64x640x1024xf32>
      %389 = stablehlo.select %340, %366, %388 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
      %390 = stablehlo.multiply %339, %389 : tensor<64x640x1024xf32>
      %391 = stablehlo.select %336, %331, %390 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
      %392 = stablehlo.subtract %333, %391 : tensor<64x640x1024xf32>
      %393 = stablehlo.select %332, %392, %391 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
      %394 = stablehlo.select %308, %330, %393 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
      %395 = stablehlo.multiply %301, %394 : tensor<64x640x1024xf32>
      mhlo.return %395 : tensor<64x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x640x32x32xf32>, tensor<640xf32>) -> tensor<64x640x1024xf32>
    %183 = "mhlo.fusion"(%182, %arg124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x640x1024xf32>, %arg209: tensor<160x640xf32>):
      %297 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x640xf32>) -> tensor<65536x640xf32>
      %299 = stablehlo.dot_general %298, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x640xf32>, tensor<160x640xf32>) -> tensor<65536x160xf32>
      mhlo.return %299 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x640x1024xf32>, tensor<160x640xf32>) -> tensor<65536x160xf32>
    %184 = "mhlo.fusion"(%183, %arg125, %arg11, %arg10, %146, %174, %arg117) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160xf32>, %arg212: tensor<64x160x32x32xf32>, %arg213: tensor<65536x160xf32>, %arg214: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<65536x160xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %300 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %301 = stablehlo.add %300, %arg213 : tensor<65536x160xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %303 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x32x32xf32>, tensor<f32>) -> tensor<64x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x32x32xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<64x32x32xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x32xf32>) -> tensor<64x1024xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %310 = stablehlo.subtract %304, %309 : tensor<64x1024x160xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<64x1024x160xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %314 = stablehlo.multiply %312, %313 : tensor<64x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x1024xf32>
      %317 = stablehlo.rsqrt %316 : tensor<64x1024xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %319 = stablehlo.multiply %310, %318 : tensor<64x1024x160xf32>
      %320 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<64x1024x160xf32>
      %322 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %323 = stablehlo.add %321, %322 : tensor<64x1024x160xf32>
      %324 = stablehlo.add %302, %323 : tensor<64x1024x160xf32>
      %325 = stablehlo.add %299, %324 : tensor<64x1024x160xf32>
      mhlo.return %325 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<64x160x32x32xf32>, tensor<65536x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %185 = "mhlo.fusion"(%184) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<64x1024xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %301 = stablehlo.subtract %arg208, %300 : tensor<64x1024x160xf32>
      %302 = stablehlo.multiply %301, %301 : tensor<64x1024x160xf32>
      %303 = stablehlo.reduce(%302 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %304 = stablehlo.multiply %303, %298 : tensor<64x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %306 = stablehlo.add %304, %305 : tensor<64x1024xf32>
      %307 = stablehlo.rsqrt %306 : tensor<64x1024xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %309 = stablehlo.multiply %301, %308 : tensor<64x1024x160xf32>
      mhlo.return %309 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>) -> tensor<64x1024x160xf32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
    %187 = "mhlo.fusion"(%186, %arg126, %arg127, %arg128) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<65536x160xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %300 = stablehlo.add %298, %299 : tensor<65536x160xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %301 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %188 = "mhlo.fusion"(%187, %arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<65536x160xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x5x32xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1, 3] : (tensor<64x1024x5x32xf32>) -> tensor<64x5x1024x32xf32>
      mhlo.return %300 : tensor<64x5x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>) -> tensor<64x5x1024x32xf32>
    %189 = mhlo.bitcast %188 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x32xf32>) -> tensor<320x1024x32xf32>
    %190 = "mhlo.fusion"(%arg127, %185, %arg126) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<64x1024x160xf32>, %arg210: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %298 = stablehlo.multiply %arg209, %297 : tensor<64x1024x160xf32>
      %299 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x1024x160xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
      mhlo.return %301 : tensor<64x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<64x1024x160xf32>, tensor<160xf32>) -> tensor<64x160x1024xf32>
    %191 = mhlo.bitcast %190 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x160x1024xf32>) -> tensor<64x160x32x32xf32>
    %192 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%191, %arg134, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,160,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x160x32x32xf32>, tensor<160x160x2x2xf32>, tensor<160xf32>) -> tuple<tensor<64x160x16x16xf32>, tensor<0xui8>>
    %193 = stablehlo.get_tuple_element %192[0] : (tuple<tensor<64x160x16x16xf32>, tensor<0xui8>>) -> tensor<64x160x16x16xf32>
    %194 = "mhlo.fusion"(%193) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x160x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      mhlo.return %297 : tensor<64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x160x16x16xf32>) -> tensor<64x16x16xf32>
    %195 = "mhlo.fusion"(%194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x16x16xf32>
      mhlo.return %298 : tensor<64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16xf32>) -> tensor<64x16x16xf32>
    %196 = "mhlo.fusion"(%195, %193) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64x16x16xf32>, %arg209: tensor<64x160x16x16xf32>):
      %297 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %298 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
      %300 = stablehlo.subtract %297, %299 : tensor<64x160x256xf32>
      %301 = stablehlo.multiply %300, %300 : tensor<64x160x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x160x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %302 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16xf32>, tensor<64x160x16x16xf32>) -> tensor<64x256xf32>
    %197 = "mhlo.fusion"(%196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x256xf32>
      %301 = stablehlo.rsqrt %300 : tensor<64x256xf32>
      mhlo.return %301 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %198 = "mhlo.fusion"(%197, %195, %193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16xf32>, %arg210: tensor<64x160x16x16xf32>):
      %297 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %298 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
      %300 = stablehlo.subtract %297, %299 : tensor<64x160x256xf32>
      %301 = stablehlo.broadcast_in_dim %arg208, dims = [0, 2] : (tensor<64x256xf32>) -> tensor<64x160x256xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<64x160x256xf32>
      mhlo.return %302 : tensor<64x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16xf32>, tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %199 = "mhlo.fusion"(%arg130, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160xf32>, %arg209: tensor<160x160xf32>):
      %297 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
      mhlo.return %297 : tensor<320x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    %200 = mhlo.bitcast %199 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[160,320]{0,1}"} : (tensor<320x160xf32>) -> tensor<160x320xf32>
    %201 = "mhlo.fusion"(%198, %arg136, %arg137, %200) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x160x256xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x320xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<64x160x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x160x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<64x160x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x160x256xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x160xf32>) -> tensor<16384x160xf32>
      %303 = stablehlo.dot %302, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x320xf32>) -> tensor<16384x320xf32>
      mhlo.return %303 : tensor<16384x320xf32>
    }) {output_operand_aliasing = []} : (tensor<64x160x256xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x320xf32>) -> tensor<16384x320xf32>
    %202 = "mhlo.fusion"(%201, %arg131) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384x320xf32>, %arg209: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 0:160] : (tensor<16384x320xf32>) -> tensor<16384x160xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x160xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 3, 1] : (tensor<64x256x5x32xf32>) -> tensor<64x5x32x256xf32>
      mhlo.return %301 : tensor<64x5x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x320xf32>, tensor<160xf32>) -> tensor<64x5x32x256xf32>
    %203 = mhlo.bitcast %202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x32x256xf32>) -> tensor<320x32x256xf32>
    %204 = "mhlo.fusion"(%189, %203) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<320x1024x32xf32>, %arg209: tensor<320x32x256xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x32xf32>, tensor<320x32x256xf32>) -> tensor<320x1024x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<320x1024x256xf32>
      mhlo.return %299 : tensor<320x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x32xf32>, tensor<320x32x256xf32>) -> tensor<320x1024x256xf32>
    %205 = "mhlo.fusion"(%204) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<320x1024x256xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %298 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<320x1024x256xf32>, tensor<f32>) -> tensor<320x1024xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<320x1024xf32>) -> tensor<64x5x1024xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
      %301 = stablehlo.subtract %297, %300 : tensor<64x5x1024x256xf32>
      %302 = stablehlo.exponential %301 : tensor<64x5x1024x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
      %305 = stablehlo.divide %302, %304 : tensor<64x5x1024x256xf32>
      mhlo.return %305 : tensor<64x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %206 = mhlo.bitcast %205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x256xf32>) -> tensor<320x1024x256xf32>
    %207 = "mhlo.fusion"(%201, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x320xf32>, %arg209: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 160:320] : (tensor<16384x320xf32>) -> tensor<16384x160xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x160xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x256x5x32xf32>) -> tensor<64x5x256x32xf32>
      mhlo.return %301 : tensor<64x5x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x320xf32>, tensor<160xf32>) -> tensor<64x5x256x32xf32>
    %208 = mhlo.bitcast %207 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x256x32xf32>) -> tensor<320x256x32xf32>
    %209 = "mhlo.fusion"(%206, %208) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<320x1024x256xf32>, %arg209: tensor<320x256x32xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x256xf32>, tensor<320x256x32xf32>) -> tensor<320x1024x32xf32>
      mhlo.return %297 : tensor<320x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x256xf32>, tensor<320x256x32xf32>) -> tensor<320x1024x32xf32>
    %210 = "mhlo.fusion"(%209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<320x1024x32xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x32xf32>) -> tensor<64x5x1024x32xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] : (tensor<64x5x1024x32xf32>) -> tensor<64x1024x5x32xf32>
      mhlo.return %298 : tensor<64x1024x5x32xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x32xf32>) -> tensor<64x1024x5x32xf32>
    %211 = mhlo.bitcast %210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x5x32xf32>) -> tensor<65536x160xf32>
    %212 = "mhlo.fusion"(%211, %arg138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160x160xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %297 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %213 = "mhlo.fusion"(%184, %212, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<65536x160xf32>, %arg210: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.add %297, %arg209 : tensor<65536x160xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %300 = stablehlo.add %299, %arg208 : tensor<64x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%300 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<64x1024xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %305 = stablehlo.subtract %300, %304 : tensor<64x1024x160xf32>
      %306 = stablehlo.multiply %305, %305 : tensor<64x1024x160xf32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %308 = stablehlo.multiply %307, %302 : tensor<64x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %310 = stablehlo.add %308, %309 : tensor<64x1024xf32>
      %311 = stablehlo.rsqrt %310 : tensor<64x1024xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %313 = stablehlo.multiply %305, %312 : tensor<64x1024x160xf32>
      mhlo.return %313 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<65536x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %214 = mhlo.bitcast %213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
    %215 = "mhlo.fusion"(%214, %arg140, %arg141, %arg142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<640x160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<65536x160xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %300 = stablehlo.add %298, %299 : tensor<65536x160xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<640x160xf32>) -> tensor<65536x640xf32>
      mhlo.return %301 : tensor<65536x640xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<640x160xf32>) -> tensor<65536x640xf32>
    %216 = "mhlo.fusion"(%215, %arg143) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x640xf32>, %arg209: tensor<640xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<65536x640xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<65536x640xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x640xf32>) -> tensor<64x1024x640xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] : (tensor<64x1024x640xf32>) -> tensor<64x640x1024xf32>
      mhlo.return %300 : tensor<64x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x640xf32>, tensor<640xf32>) -> tensor<64x640x1024xf32>
    %217 = mhlo.bitcast %216 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x640x1024xf32>) -> tensor<64x640x32x32xf32>
    %218 = stablehlo.custom_call @__cudnn$convForward(%217, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,640,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x640x32x32xf32>, tensor<640x1x3x3xf32>) -> tuple<tensor<64x640x32x32xf32>, tensor<0xui8>>
    %219 = stablehlo.get_tuple_element %218[0] : (tuple<tensor<64x640x32x32xf32>, tensor<0xui8>>) -> tensor<64x640x32x32xf32>
    %220 = "mhlo.fusion"(%219, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x640x32x32xf32>, %arg209: tensor<640xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x640x32x32xf32>
      %298 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<64x640x32x32xf32>
      %299 = stablehlo.add %arg208, %298 : tensor<64x640x32x32xf32>
      %300 = stablehlo.multiply %297, %299 : tensor<64x640x32x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
      %302 = stablehlo.negate %299 : tensor<64x640x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x640x32x32xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x640x32x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
      %306 = stablehlo.abs %305 : tensor<64x640x1024xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %308 = stablehlo.compare LT, %306, %307 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %310 = stablehlo.multiply %305, %305 : tensor<64x640x1024xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x640x1024xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x640x1024xf32>
      %314 = stablehlo.multiply %313, %310 : tensor<64x640x1024xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x640x1024xf32>
      %317 = stablehlo.multiply %316, %310 : tensor<64x640x1024xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %319 = stablehlo.add %317, %318 : tensor<64x640x1024xf32>
      %320 = stablehlo.multiply %319, %310 : tensor<64x640x1024xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %322 = stablehlo.add %320, %321 : tensor<64x640x1024xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x640x1024xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %325 = stablehlo.add %323, %324 : tensor<64x640x1024xf32>
      %326 = stablehlo.multiply %325, %310 : tensor<64x640x1024xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<64x640x1024xf32>
      %329 = stablehlo.multiply %305, %328 : tensor<64x640x1024xf32>
      %330 = stablehlo.subtract %307, %329 : tensor<64x640x1024xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %332 = stablehlo.compare LT, %305, %331 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %334 = stablehlo.negate %310 : tensor<64x640x1024xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %336 = stablehlo.compare LT, %334, %335 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
      %337 = stablehlo.exponential %334 : tensor<64x640x1024xf32>
      %338 = stablehlo.divide %307, %306 : tensor<64x640x1024xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x640x1024xf32>
      %340 = stablehlo.compare LT, %306, %333 : (tensor<64x640x1024xf32>, tensor<64x640x1024xf32>) -> tensor<64x640x1024xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %342 = stablehlo.divide %307, %310 : tensor<64x640x1024xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<64x640x1024xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %345 = stablehlo.add %343, %344 : tensor<64x640x1024xf32>
      %346 = stablehlo.multiply %345, %342 : tensor<64x640x1024xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %348 = stablehlo.add %346, %347 : tensor<64x640x1024xf32>
      %349 = stablehlo.multiply %348, %342 : tensor<64x640x1024xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %351 = stablehlo.add %349, %350 : tensor<64x640x1024xf32>
      %352 = stablehlo.multiply %351, %342 : tensor<64x640x1024xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %354 = stablehlo.add %352, %353 : tensor<64x640x1024xf32>
      %355 = stablehlo.multiply %354, %342 : tensor<64x640x1024xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %357 = stablehlo.add %355, %356 : tensor<64x640x1024xf32>
      %358 = stablehlo.multiply %357, %342 : tensor<64x640x1024xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %360 = stablehlo.add %358, %359 : tensor<64x640x1024xf32>
      %361 = stablehlo.multiply %360, %342 : tensor<64x640x1024xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %363 = stablehlo.add %361, %362 : tensor<64x640x1024xf32>
      %364 = stablehlo.multiply %363, %342 : tensor<64x640x1024xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %366 = stablehlo.add %364, %365 : tensor<64x640x1024xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %368 = stablehlo.multiply %367, %342 : tensor<64x640x1024xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %370 = stablehlo.add %368, %369 : tensor<64x640x1024xf32>
      %371 = stablehlo.multiply %370, %342 : tensor<64x640x1024xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %372 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %373 = stablehlo.add %371, %372 : tensor<64x640x1024xf32>
      %374 = stablehlo.multiply %373, %342 : tensor<64x640x1024xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %376 = stablehlo.add %374, %375 : tensor<64x640x1024xf32>
      %377 = stablehlo.multiply %376, %342 : tensor<64x640x1024xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x640x1024xf32>
      %380 = stablehlo.multiply %379, %342 : tensor<64x640x1024xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %382 = stablehlo.add %380, %381 : tensor<64x640x1024xf32>
      %383 = stablehlo.multiply %382, %342 : tensor<64x640x1024xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %385 = stablehlo.add %383, %384 : tensor<64x640x1024xf32>
      %386 = stablehlo.multiply %385, %342 : tensor<64x640x1024xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x640x1024xf32>
      %388 = stablehlo.add %386, %387 : tensor<64x640x1024xf32>
      %389 = stablehlo.select %340, %366, %388 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
      %390 = stablehlo.multiply %339, %389 : tensor<64x640x1024xf32>
      %391 = stablehlo.select %336, %331, %390 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
      %392 = stablehlo.subtract %333, %391 : tensor<64x640x1024xf32>
      %393 = stablehlo.select %332, %392, %391 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
      %394 = stablehlo.select %308, %330, %393 : tensor<64x640x1024xi1>, tensor<64x640x1024xf32>
      %395 = stablehlo.multiply %301, %394 : tensor<64x640x1024xf32>
      mhlo.return %395 : tensor<64x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x640x32x32xf32>, tensor<640xf32>) -> tensor<64x640x1024xf32>
    %221 = "mhlo.fusion"(%220, %arg146) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x640x1024xf32>, %arg209: tensor<160x640xf32>):
      %297 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x640xf32>) -> tensor<65536x640xf32>
      %299 = stablehlo.dot_general %298, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x640xf32>, tensor<160x640xf32>) -> tensor<65536x160xf32>
      mhlo.return %299 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x640x1024xf32>, tensor<160x640xf32>) -> tensor<65536x160xf32>
    %222:2 = "mhlo.fusion"(%arg189, %arg188, %221, %arg147, %184, %212, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<65536x160xf32>, %arg211: tensor<160xf32>, %arg212: tensor<64x1024x160xf32>, %arg213: tensor<65536x160xf32>, %arg214: tensor<160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %298 = stablehlo.add %297, %arg210 : tensor<65536x160xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %300 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %301 = stablehlo.add %300, %arg213 : tensor<65536x160xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %303 = stablehlo.add %302, %arg212 : tensor<64x1024x160xf32>
      %304 = stablehlo.add %299, %303 : tensor<64x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<64x1024xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %309 = stablehlo.subtract %304, %308 : tensor<64x1024x160xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<64x1024x160xf32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      %312 = stablehlo.multiply %311, %306 : tensor<64x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %314 = stablehlo.add %312, %313 : tensor<64x1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64x1024xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<64x1024x160xf32>
      %318 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<64x1024x160xf32>
      %320 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %321 = stablehlo.add %319, %320 : tensor<64x1024x160xf32>
      %322 = stablehlo.transpose %321, dims = [0, 2, 1] : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
      mhlo.return %322, %317 : tensor<64x160x1024xf32>, tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<65536x160xf32>, tensor<160xf32>, tensor<64x1024x160xf32>, tensor<65536x160xf32>, tensor<160xf32>) -> (tensor<64x160x1024xf32>, tensor<64x1024x160xf32>)
    %223 = stablehlo.tuple %222#0, %222#1 {xla_shape = "(f32[64,160,1024]{2,1,0}, f32[64,1024,160]{2,1,0})"} : tuple<tensor<64x160x1024xf32>, tensor<64x1024x160xf32>>
    %224 = mhlo.bitcast %222#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x160x1024xf32>) -> tensor<64x160x32x32xf32>
    %225 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%224, %arg12, %arg13) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x160x32x32xf32>, tensor<256x160x3x3xf32>, tensor<256xf32>) -> tuple<tensor<64x256x16x16xf32>, tensor<0xui8>>
    %226 = stablehlo.get_tuple_element %225[0] : (tuple<tensor<64x256x16x16xf32>, tensor<0xui8>>) -> tensor<64x256x16x16xf32>
    %227 = "mhlo.fusion"(%arg15, %arg14, %226) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256x16x16xf32>):
      %297 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1] : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<64x16x16xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %304 = stablehlo.subtract %298, %303 : tensor<64x256x256xf32>
      %305 = stablehlo.multiply %304, %304 : tensor<64x256x256xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<64x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %310 = stablehlo.add %308, %309 : tensor<64x256xf32>
      %311 = stablehlo.rsqrt %310 : tensor<64x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %313 = stablehlo.multiply %304, %312 : tensor<64x256x256xf32>
      %314 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<64x256x256xf32>
      %316 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %317 = stablehlo.add %315, %316 : tensor<64x256x256xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %319 = stablehlo.multiply %318, %307 : tensor<64x256xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<64x256x256xf32>
      %322 = stablehlo.multiply %321, %321 : tensor<64x256x256xf32>
      %323 = stablehlo.reduce(%322 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %324 = stablehlo.multiply %323, %307 : tensor<64x256xf32>
      %325 = stablehlo.add %324, %309 : tensor<64x256xf32>
      %326 = stablehlo.rsqrt %325 : tensor<64x256xf32>
      %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %328 = stablehlo.multiply %321, %327 : tensor<64x256x256xf32>
      mhlo.return %328 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %228 = mhlo.bitcast %227 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
    %229 = "mhlo.fusion"(%228, %arg148, %arg149, %arg150, %arg152, %arg154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>, %arg212: tensor<256x256xf32>, %arg213: tensor<256x256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<16384x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<16384x256xf32>
      %301 = stablehlo.concatenate %arg211, %arg212, %arg213, dim = 0 : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<768x256xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
      %303 = stablehlo.dot %300, %302, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x768xf32>) -> tensor<16384x768xf32>
      mhlo.return %303 : tensor<16384x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<16384x768xf32>
    %230 = "mhlo.fusion"(%229, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x768xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 0:256] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x256xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
      mhlo.return %301 : tensor<64x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x768xf32>, tensor<256xf32>) -> tensor<64x8x256x32xf32>
    %231 = mhlo.bitcast %230 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
    %232 = "mhlo.fusion"(%229, %arg153) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384x768xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 256:512] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x256xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 3, 1] : (tensor<64x256x8x32xf32>) -> tensor<64x8x32x256xf32>
      mhlo.return %301 : tensor<64x8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x768xf32>, tensor<256xf32>) -> tensor<64x8x32x256xf32>
    %233 = mhlo.bitcast %232 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x32x256xf32>) -> tensor<512x32x256xf32>
    %234 = "mhlo.fusion"(%231, %233) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x256x32xf32>, %arg209: tensor<512x32x256xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x32xf32>, tensor<512x32x256xf32>) -> tensor<512x256x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<512x256x256xf32>
      mhlo.return %299 : tensor<512x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x32xf32>, tensor<512x32x256xf32>) -> tensor<512x256x256xf32>
    %235 = "mhlo.fusion"(%234) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x256x256xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %298 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<512x256x256xf32>, tensor<f32>) -> tensor<512x256xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x256xf32>) -> tensor<64x8x256xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
      %301 = stablehlo.subtract %297, %300 : tensor<64x8x256x256xf32>
      %302 = stablehlo.exponential %301 : tensor<64x8x256x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
      %305 = stablehlo.divide %302, %304 : tensor<64x8x256x256xf32>
      mhlo.return %305 : tensor<64x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %236 = mhlo.bitcast %235 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x256xf32>) -> tensor<512x256x256xf32>
    %237 = "mhlo.fusion"(%229, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x768xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 512:768] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x256xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
      mhlo.return %301 : tensor<64x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x768xf32>, tensor<256xf32>) -> tensor<64x8x256x32xf32>
    %238 = mhlo.bitcast %237 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
    %239 = "mhlo.fusion"(%236, %238) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x256x256xf32>, %arg209: tensor<512x256x32xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x256xf32>, tensor<512x256x32xf32>) -> tensor<512x256x32xf32>
      mhlo.return %297 : tensor<512x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x256xf32>, tensor<512x256x32xf32>) -> tensor<512x256x32xf32>
    %240 = "mhlo.fusion"(%239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<512x256x32xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x32xf32>) -> tensor<64x8x256x32xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] : (tensor<64x8x256x32xf32>) -> tensor<64x256x8x32xf32>
      mhlo.return %298 : tensor<64x256x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x32xf32>) -> tensor<64x256x8x32xf32>
    %241 = mhlo.bitcast %240 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x8x32xf32>) -> tensor<16384x256xf32>
    %242 = "mhlo.fusion"(%241, %arg156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256x256xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
      mhlo.return %297 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %243 = "mhlo.fusion"(%arg15, %arg14, %226, %242, %arg157) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256x16x16xf32>, %arg211: tensor<16384x256xf32>, %arg212: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.add %297, %arg211 : tensor<16384x256xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %300 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1] : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x16x16xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %307 = stablehlo.subtract %301, %306 : tensor<64x256x256xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<64x256x256xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x256xf32>
      %314 = stablehlo.rsqrt %313 : tensor<64x256xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %316 = stablehlo.multiply %307, %315 : tensor<64x256x256xf32>
      %317 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<64x256x256xf32>
      %319 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<64x256x256xf32>
      %321 = stablehlo.add %299, %320 : tensor<64x256x256xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x256xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %325 = stablehlo.subtract %321, %324 : tensor<64x256x256xf32>
      %326 = stablehlo.multiply %325, %325 : tensor<64x256x256xf32>
      %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %328 = stablehlo.multiply %327, %310 : tensor<64x256xf32>
      %329 = stablehlo.add %328, %312 : tensor<64x256xf32>
      %330 = stablehlo.rsqrt %329 : tensor<64x256xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %332 = stablehlo.multiply %325, %331 : tensor<64x256x256xf32>
      mhlo.return %332 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x16x16xf32>, tensor<16384x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %244 = mhlo.bitcast %243 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
    %245 = "mhlo.fusion"(%244, %arg158, %arg159, %arg160) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1024x256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<16384x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<16384x256xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<1024x256xf32>) -> tensor<16384x1024xf32>
      mhlo.return %301 : tensor<16384x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256xf32>) -> tensor<16384x1024xf32>
    %246 = "mhlo.fusion"(%245, %arg161) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384x1024xf32>, %arg209: tensor<1024xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<16384x1024xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<16384x1024xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x1024xf32>) -> tensor<64x256x1024xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] : (tensor<64x256x1024xf32>) -> tensor<64x1024x256xf32>
      mhlo.return %300 : tensor<64x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x1024xf32>, tensor<1024xf32>) -> tensor<64x1024x256xf32>
    %247 = mhlo.bitcast %246 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x256xf32>) -> tensor<64x1024x16x16xf32>
    %248 = stablehlo.custom_call @__cudnn$convForward(%247, %arg162) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x16x16xf32>, tensor<1024x1x3x3xf32>) -> tuple<tensor<64x1024x16x16xf32>, tensor<0xui8>>
    %249 = stablehlo.get_tuple_element %248[0] : (tuple<tensor<64x1024x16x16xf32>, tensor<0xui8>>) -> tensor<64x1024x16x16xf32>
    %250 = "mhlo.fusion"(%249, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x16x16xf32>, %arg209: tensor<1024xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x16x16xf32>
      %298 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x16x16xf32>
      %299 = stablehlo.add %arg208, %298 : tensor<64x1024x16x16xf32>
      %300 = stablehlo.multiply %297, %299 : tensor<64x1024x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
      %302 = stablehlo.negate %299 : tensor<64x1024x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024x16x16xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x1024x16x16xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
      %306 = stablehlo.abs %305 : tensor<64x1024x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %308 = stablehlo.compare LT, %306, %307 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %310 = stablehlo.multiply %305, %305 : tensor<64x1024x256xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x1024x256xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x1024x256xf32>
      %314 = stablehlo.multiply %313, %310 : tensor<64x1024x256xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x1024x256xf32>
      %317 = stablehlo.multiply %316, %310 : tensor<64x1024x256xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %319 = stablehlo.add %317, %318 : tensor<64x1024x256xf32>
      %320 = stablehlo.multiply %319, %310 : tensor<64x1024x256xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %322 = stablehlo.add %320, %321 : tensor<64x1024x256xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x1024x256xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %325 = stablehlo.add %323, %324 : tensor<64x1024x256xf32>
      %326 = stablehlo.multiply %325, %310 : tensor<64x1024x256xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %328 = stablehlo.add %326, %327 : tensor<64x1024x256xf32>
      %329 = stablehlo.multiply %305, %328 : tensor<64x1024x256xf32>
      %330 = stablehlo.subtract %307, %329 : tensor<64x1024x256xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %332 = stablehlo.compare LT, %305, %331 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %334 = stablehlo.negate %310 : tensor<64x1024x256xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %336 = stablehlo.compare LT, %334, %335 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
      %337 = stablehlo.exponential %334 : tensor<64x1024x256xf32>
      %338 = stablehlo.divide %307, %306 : tensor<64x1024x256xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x1024x256xf32>
      %340 = stablehlo.compare LT, %306, %333 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %342 = stablehlo.divide %307, %310 : tensor<64x1024x256xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<64x1024x256xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %345 = stablehlo.add %343, %344 : tensor<64x1024x256xf32>
      %346 = stablehlo.multiply %345, %342 : tensor<64x1024x256xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %348 = stablehlo.add %346, %347 : tensor<64x1024x256xf32>
      %349 = stablehlo.multiply %348, %342 : tensor<64x1024x256xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %351 = stablehlo.add %349, %350 : tensor<64x1024x256xf32>
      %352 = stablehlo.multiply %351, %342 : tensor<64x1024x256xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %354 = stablehlo.add %352, %353 : tensor<64x1024x256xf32>
      %355 = stablehlo.multiply %354, %342 : tensor<64x1024x256xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %357 = stablehlo.add %355, %356 : tensor<64x1024x256xf32>
      %358 = stablehlo.multiply %357, %342 : tensor<64x1024x256xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %360 = stablehlo.add %358, %359 : tensor<64x1024x256xf32>
      %361 = stablehlo.multiply %360, %342 : tensor<64x1024x256xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %363 = stablehlo.add %361, %362 : tensor<64x1024x256xf32>
      %364 = stablehlo.multiply %363, %342 : tensor<64x1024x256xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %366 = stablehlo.add %364, %365 : tensor<64x1024x256xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %368 = stablehlo.multiply %367, %342 : tensor<64x1024x256xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %370 = stablehlo.add %368, %369 : tensor<64x1024x256xf32>
      %371 = stablehlo.multiply %370, %342 : tensor<64x1024x256xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %372 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %373 = stablehlo.add %371, %372 : tensor<64x1024x256xf32>
      %374 = stablehlo.multiply %373, %342 : tensor<64x1024x256xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %376 = stablehlo.add %374, %375 : tensor<64x1024x256xf32>
      %377 = stablehlo.multiply %376, %342 : tensor<64x1024x256xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x1024x256xf32>
      %380 = stablehlo.multiply %379, %342 : tensor<64x1024x256xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %382 = stablehlo.add %380, %381 : tensor<64x1024x256xf32>
      %383 = stablehlo.multiply %382, %342 : tensor<64x1024x256xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %385 = stablehlo.add %383, %384 : tensor<64x1024x256xf32>
      %386 = stablehlo.multiply %385, %342 : tensor<64x1024x256xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %388 = stablehlo.add %386, %387 : tensor<64x1024x256xf32>
      %389 = stablehlo.select %340, %366, %388 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
      %390 = stablehlo.multiply %339, %389 : tensor<64x1024x256xf32>
      %391 = stablehlo.select %336, %331, %390 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
      %392 = stablehlo.subtract %333, %391 : tensor<64x1024x256xf32>
      %393 = stablehlo.select %332, %392, %391 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
      %394 = stablehlo.select %308, %330, %393 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
      %395 = stablehlo.multiply %301, %394 : tensor<64x1024x256xf32>
      mhlo.return %395 : tensor<64x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x16x16xf32>, tensor<1024xf32>) -> tensor<64x1024x256xf32>
    %251 = "mhlo.fusion"(%250, %arg164) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x256xf32>, %arg209: tensor<256x1024xf32>):
      %297 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<16384x1024xf32>
      %299 = stablehlo.dot_general %298, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x1024xf32>, tensor<256x1024xf32>) -> tensor<16384x256xf32>
      mhlo.return %299 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x256xf32>, tensor<256x1024xf32>) -> tensor<16384x256xf32>
    %252 = "mhlo.fusion"(%251, %arg165, %arg15, %arg14, %226, %242, %arg157) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<64x256x16x16xf32>, %arg213: tensor<16384x256xf32>, %arg214: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<16384x256xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %300 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %301 = stablehlo.add %300, %arg213 : tensor<16384x256xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %303 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256x16x16xf32>, tensor<f32>) -> tensor<64x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16x16xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<64x16x16xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16x16xf32>) -> tensor<64x256xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %310 = stablehlo.subtract %304, %309 : tensor<64x256x256xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<64x256x256xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %314 = stablehlo.multiply %312, %313 : tensor<64x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x256xf32>
      %317 = stablehlo.rsqrt %316 : tensor<64x256xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %319 = stablehlo.multiply %310, %318 : tensor<64x256x256xf32>
      %320 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<64x256x256xf32>
      %322 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %323 = stablehlo.add %321, %322 : tensor<64x256x256xf32>
      %324 = stablehlo.add %302, %323 : tensor<64x256x256xf32>
      %325 = stablehlo.add %299, %324 : tensor<64x256x256xf32>
      mhlo.return %325 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x16x16xf32>, tensor<16384x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %253 = "mhlo.fusion"(%252) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<64x256xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %301 = stablehlo.subtract %arg208, %300 : tensor<64x256x256xf32>
      %302 = stablehlo.multiply %301, %301 : tensor<64x256x256xf32>
      %303 = stablehlo.reduce(%302 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %304 = stablehlo.multiply %303, %298 : tensor<64x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<64x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<64x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %309 = stablehlo.multiply %301, %308 : tensor<64x256x256xf32>
      mhlo.return %309 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %254 = mhlo.bitcast %253 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
    %255 = "mhlo.fusion"(%254, %arg166, %arg167, %arg168, %arg170, %arg172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>, %arg212: tensor<256x256xf32>, %arg213: tensor<256x256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<16384x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<16384x256xf32>
      %301 = stablehlo.concatenate %arg211, %arg212, %arg213, dim = 0 : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<768x256xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
      %303 = stablehlo.dot %300, %302, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x768xf32>) -> tensor<16384x768xf32>
      mhlo.return %303 : tensor<16384x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<16384x768xf32>
    %256 = "mhlo.fusion"(%255, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x768xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 0:256] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x256xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
      mhlo.return %301 : tensor<64x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x768xf32>, tensor<256xf32>) -> tensor<64x8x256x32xf32>
    %257 = mhlo.bitcast %256 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
    %258 = "mhlo.fusion"(%255, %arg171) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384x768xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 256:512] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x256xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 3, 1] : (tensor<64x256x8x32xf32>) -> tensor<64x8x32x256xf32>
      mhlo.return %301 : tensor<64x8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x768xf32>, tensor<256xf32>) -> tensor<64x8x32x256xf32>
    %259 = mhlo.bitcast %258 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x32x256xf32>) -> tensor<512x32x256xf32>
    %260 = "mhlo.fusion"(%257, %259) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x256x32xf32>, %arg209: tensor<512x32x256xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x32xf32>, tensor<512x32x256xf32>) -> tensor<512x256x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<512x256x256xf32>
      mhlo.return %299 : tensor<512x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x32xf32>, tensor<512x32x256xf32>) -> tensor<512x256x256xf32>
    %261 = "mhlo.fusion"(%260) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x256x256xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %298 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<512x256x256xf32>, tensor<f32>) -> tensor<512x256xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x256xf32>) -> tensor<64x8x256xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
      %301 = stablehlo.subtract %297, %300 : tensor<64x8x256x256xf32>
      %302 = stablehlo.exponential %301 : tensor<64x8x256x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
      %305 = stablehlo.divide %302, %304 : tensor<64x8x256x256xf32>
      mhlo.return %305 : tensor<64x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %262 = mhlo.bitcast %261 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x256xf32>) -> tensor<512x256x256xf32>
    %263 = "mhlo.fusion"(%255, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x768xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.slice %arg208 [0:16384, 512:768] : (tensor<16384x768xf32>) -> tensor<16384x256xf32>
      %299 = stablehlo.add %297, %298 : tensor<16384x256xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
      mhlo.return %301 : tensor<64x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x768xf32>, tensor<256xf32>) -> tensor<64x8x256x32xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
    %265 = "mhlo.fusion"(%262, %264) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x256x256xf32>, %arg209: tensor<512x256x32xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x256xf32>, tensor<512x256x32xf32>) -> tensor<512x256x32xf32>
      mhlo.return %297 : tensor<512x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x256xf32>, tensor<512x256x32xf32>) -> tensor<512x256x32xf32>
    %266 = "mhlo.fusion"(%265) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<512x256x32xf32>):
      %297 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x32xf32>) -> tensor<64x8x256x32xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] : (tensor<64x8x256x32xf32>) -> tensor<64x256x8x32xf32>
      mhlo.return %298 : tensor<64x256x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x32xf32>) -> tensor<64x256x8x32xf32>
    %267 = mhlo.bitcast %266 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x8x32xf32>) -> tensor<16384x256xf32>
    %268 = "mhlo.fusion"(%267, %arg174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256x256xf32>):
      %297 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
      mhlo.return %297 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %269 = "mhlo.fusion"(%252, %268, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<16384x256xf32>, %arg210: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.add %297, %arg209 : tensor<16384x256xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %300 = stablehlo.add %299, %arg208 : tensor<64x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%300 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<64x256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %305 = stablehlo.subtract %300, %304 : tensor<64x256x256xf32>
      %306 = stablehlo.multiply %305, %305 : tensor<64x256x256xf32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %308 = stablehlo.multiply %307, %302 : tensor<64x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %310 = stablehlo.add %308, %309 : tensor<64x256xf32>
      %311 = stablehlo.rsqrt %310 : tensor<64x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %313 = stablehlo.multiply %305, %312 : tensor<64x256x256xf32>
      mhlo.return %313 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<16384x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %270 = mhlo.bitcast %269 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
    %271 = "mhlo.fusion"(%270, %arg176, %arg177, %arg178) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1024x256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<16384x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<16384x256xf32>
      %301 = stablehlo.dot_general %300, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<1024x256xf32>) -> tensor<16384x1024xf32>
      mhlo.return %301 : tensor<16384x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256xf32>) -> tensor<16384x1024xf32>
    %272 = "mhlo.fusion"(%271, %arg179) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384x1024xf32>, %arg209: tensor<1024xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<16384x1024xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<16384x1024xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x1024xf32>) -> tensor<64x256x1024xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] : (tensor<64x256x1024xf32>) -> tensor<64x1024x256xf32>
      mhlo.return %300 : tensor<64x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x1024xf32>, tensor<1024xf32>) -> tensor<64x1024x256xf32>
    %273 = mhlo.bitcast %272 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x256xf32>) -> tensor<64x1024x16x16xf32>
    %274 = stablehlo.custom_call @__cudnn$convForward(%273, %arg180) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,1024,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x16x16xf32>, tensor<1024x1x3x3xf32>) -> tuple<tensor<64x1024x16x16xf32>, tensor<0xui8>>
    %275 = stablehlo.get_tuple_element %274[0] : (tuple<tensor<64x1024x16x16xf32>, tensor<0xui8>>) -> tensor<64x1024x16x16xf32>
    %276 = "mhlo.fusion"(%275, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x16x16xf32>, %arg209: tensor<1024xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024x16x16xf32>
      %298 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x16x16xf32>
      %299 = stablehlo.add %arg208, %298 : tensor<64x1024x16x16xf32>
      %300 = stablehlo.multiply %297, %299 : tensor<64x1024x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
      %302 = stablehlo.negate %299 : tensor<64x1024x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024x16x16xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<64x1024x16x16xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
      %306 = stablehlo.abs %305 : tensor<64x1024x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %308 = stablehlo.compare LT, %306, %307 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %310 = stablehlo.multiply %305, %305 : tensor<64x1024x256xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<64x1024x256xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %313 = stablehlo.add %311, %312 : tensor<64x1024x256xf32>
      %314 = stablehlo.multiply %313, %310 : tensor<64x1024x256xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %316 = stablehlo.add %314, %315 : tensor<64x1024x256xf32>
      %317 = stablehlo.multiply %316, %310 : tensor<64x1024x256xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %319 = stablehlo.add %317, %318 : tensor<64x1024x256xf32>
      %320 = stablehlo.multiply %319, %310 : tensor<64x1024x256xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %322 = stablehlo.add %320, %321 : tensor<64x1024x256xf32>
      %323 = stablehlo.multiply %322, %310 : tensor<64x1024x256xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %325 = stablehlo.add %323, %324 : tensor<64x1024x256xf32>
      %326 = stablehlo.multiply %325, %310 : tensor<64x1024x256xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %328 = stablehlo.add %326, %327 : tensor<64x1024x256xf32>
      %329 = stablehlo.multiply %305, %328 : tensor<64x1024x256xf32>
      %330 = stablehlo.subtract %307, %329 : tensor<64x1024x256xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %332 = stablehlo.compare LT, %305, %331 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %334 = stablehlo.negate %310 : tensor<64x1024x256xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %336 = stablehlo.compare LT, %334, %335 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
      %337 = stablehlo.exponential %334 : tensor<64x1024x256xf32>
      %338 = stablehlo.divide %307, %306 : tensor<64x1024x256xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x1024x256xf32>
      %340 = stablehlo.compare LT, %306, %333 : (tensor<64x1024x256xf32>, tensor<64x1024x256xf32>) -> tensor<64x1024x256xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %342 = stablehlo.divide %307, %310 : tensor<64x1024x256xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<64x1024x256xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %345 = stablehlo.add %343, %344 : tensor<64x1024x256xf32>
      %346 = stablehlo.multiply %345, %342 : tensor<64x1024x256xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %348 = stablehlo.add %346, %347 : tensor<64x1024x256xf32>
      %349 = stablehlo.multiply %348, %342 : tensor<64x1024x256xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %351 = stablehlo.add %349, %350 : tensor<64x1024x256xf32>
      %352 = stablehlo.multiply %351, %342 : tensor<64x1024x256xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %354 = stablehlo.add %352, %353 : tensor<64x1024x256xf32>
      %355 = stablehlo.multiply %354, %342 : tensor<64x1024x256xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %357 = stablehlo.add %355, %356 : tensor<64x1024x256xf32>
      %358 = stablehlo.multiply %357, %342 : tensor<64x1024x256xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %360 = stablehlo.add %358, %359 : tensor<64x1024x256xf32>
      %361 = stablehlo.multiply %360, %342 : tensor<64x1024x256xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %363 = stablehlo.add %361, %362 : tensor<64x1024x256xf32>
      %364 = stablehlo.multiply %363, %342 : tensor<64x1024x256xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %366 = stablehlo.add %364, %365 : tensor<64x1024x256xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %368 = stablehlo.multiply %367, %342 : tensor<64x1024x256xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %370 = stablehlo.add %368, %369 : tensor<64x1024x256xf32>
      %371 = stablehlo.multiply %370, %342 : tensor<64x1024x256xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %372 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %373 = stablehlo.add %371, %372 : tensor<64x1024x256xf32>
      %374 = stablehlo.multiply %373, %342 : tensor<64x1024x256xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %376 = stablehlo.add %374, %375 : tensor<64x1024x256xf32>
      %377 = stablehlo.multiply %376, %342 : tensor<64x1024x256xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x1024x256xf32>
      %380 = stablehlo.multiply %379, %342 : tensor<64x1024x256xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %382 = stablehlo.add %380, %381 : tensor<64x1024x256xf32>
      %383 = stablehlo.multiply %382, %342 : tensor<64x1024x256xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %385 = stablehlo.add %383, %384 : tensor<64x1024x256xf32>
      %386 = stablehlo.multiply %385, %342 : tensor<64x1024x256xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x1024x256xf32>
      %388 = stablehlo.add %386, %387 : tensor<64x1024x256xf32>
      %389 = stablehlo.select %340, %366, %388 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
      %390 = stablehlo.multiply %339, %389 : tensor<64x1024x256xf32>
      %391 = stablehlo.select %336, %331, %390 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
      %392 = stablehlo.subtract %333, %391 : tensor<64x1024x256xf32>
      %393 = stablehlo.select %332, %392, %391 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
      %394 = stablehlo.select %308, %330, %393 : tensor<64x1024x256xi1>, tensor<64x1024x256xf32>
      %395 = stablehlo.multiply %301, %394 : tensor<64x1024x256xf32>
      mhlo.return %395 : tensor<64x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x16x16xf32>, tensor<1024xf32>) -> tensor<64x1024x256xf32>
    %277 = "mhlo.fusion"(%276, %arg182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x256xf32>, %arg209: tensor<256x1024xf32>):
      %297 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<16384x1024xf32>
      %299 = stablehlo.dot_general %298, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x1024xf32>, tensor<256x1024xf32>) -> tensor<16384x256xf32>
      mhlo.return %299 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x256xf32>, tensor<256x1024xf32>) -> tensor<16384x256xf32>
    %278 = "mhlo.fusion"(%277, %arg183, %252, %268, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256x256xf32>, %arg211: tensor<16384x256xf32>, %arg212: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %298 = stablehlo.add %297, %arg208 : tensor<16384x256xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %300 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %301 = stablehlo.add %300, %arg211 : tensor<16384x256xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %303 = stablehlo.add %302, %arg210 : tensor<64x256x256xf32>
      %304 = stablehlo.add %299, %303 : tensor<64x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<64x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %309 = stablehlo.subtract %304, %308 : tensor<64x256x256xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<64x256x256xf32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      %312 = stablehlo.multiply %311, %306 : tensor<64x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %314 = stablehlo.add %312, %313 : tensor<64x256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64x256xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<64x256x256xf32>
      mhlo.return %317 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>, tensor<64x256x256xf32>, tensor<16384x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %279 = "mhlo.fusion"(%278, %arg190, %arg191, %arg198) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x256x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x256x256xf32>
      %301 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<64x256x256xf32>
      %303 = stablehlo.dot_general %300, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x256x256xf32>, tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      mhlo.return %303 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<64x256x256xf32>
    %280 = "mhlo.fusion"(%279, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %298 = stablehlo.add %arg208, %297 : tensor<64x256x256xf32>
      mhlo.return %298 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %281 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %299 = stablehlo.iota dim = 0 : tensor<128xi32>
      %300 = stablehlo.convert %299 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %302 = stablehlo.add %300, %301 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %306 = stablehlo.add %304, %305 : tensor<128xf32>
      %307 = stablehlo.maximum %298, %306 : tensor<128xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %309 = stablehlo.convert %308 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %310 = stablehlo.convert %309 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %311 = stablehlo.subtract %308, %310 : tensor<128x1xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %313 = stablehlo.clamp %297, %311, %312 : tensor<128x1xf32>
      mhlo.return %313 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %282 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %298 = stablehlo.iota dim = 0 : tensor<128xi32>
      %299 = stablehlo.convert %298 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %301 = stablehlo.add %299, %300 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.add %303, %304 : tensor<128xf32>
      %306 = stablehlo.maximum %297, %305 : tensor<128xf32>
      %307 = stablehlo.convert %306 : (tensor<128xf32>) -> tensor<128xi32>
      %308 = stablehlo.convert %307 : (tensor<128xi32>) -> tensor<128xf32>
      %309 = stablehlo.subtract %306, %308 : tensor<128xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %311 = stablehlo.clamp %297, %309, %310 : tensor<128xf32>
      mhlo.return %311 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %283 = "mhlo.fusion"(%222#1, %arg188, %arg189, %arg196) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<256x160xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x1024x160xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x1024x160xf32>
      %301 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x160xf32>) -> tensor<160x256xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<64x160x256xf32>
      %303 = stablehlo.dot_general %300, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1024x160xf32>, tensor<64x160x256xf32>) -> tensor<64x1024x256xf32>
      mhlo.return %303 : tensor<64x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<256x160xf32>) -> tensor<64x1024x256xf32>
    %284 = "mhlo.fusion"(%283, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x256xf32>, %arg209: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x1024x256xf32>
      %298 = stablehlo.add %arg208, %297 : tensor<64x1024x256xf32>
      mhlo.return %298 : tensor<64x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x256xf32>, tensor<256xf32>) -> tensor<64x1024x256xf32>
    %285 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %299 = stablehlo.iota dim = 0 : tensor<128xi32>
      %300 = stablehlo.convert %299 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %302 = stablehlo.add %300, %301 : tensor<128xf32>
      %303 = stablehlo.multiply %302, %301 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.add %303, %304 : tensor<128xf32>
      %306 = stablehlo.maximum %298, %305 : tensor<128xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %308 = stablehlo.convert %307 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %309 = stablehlo.convert %308 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %310 = stablehlo.subtract %307, %309 : tensor<128x1xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %312 = stablehlo.clamp %297, %310, %311 : tensor<128x1xf32>
      mhlo.return %312 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %286 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %299 = stablehlo.iota dim = 0 : tensor<128xi32>
      %300 = stablehlo.convert %299 : (tensor<128xi32>) -> tensor<128xf32>
      %301 = stablehlo.maximum %298, %300 : tensor<128xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %303 = stablehlo.convert %302 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %304 = stablehlo.convert %303 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %305 = stablehlo.subtract %302, %304 : tensor<128x1xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %307 = stablehlo.clamp %297, %305, %306 : tensor<128x1xf32>
      mhlo.return %307 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %287 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %298 = stablehlo.iota dim = 0 : tensor<128xi32>
      %299 = stablehlo.convert %298 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %301 = stablehlo.add %299, %300 : tensor<128xf32>
      %302 = stablehlo.multiply %301, %300 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %304 = stablehlo.add %302, %303 : tensor<128xf32>
      %305 = stablehlo.maximum %297, %304 : tensor<128xf32>
      %306 = stablehlo.convert %305 : (tensor<128xf32>) -> tensor<128xi32>
      %307 = stablehlo.convert %306 : (tensor<128xi32>) -> tensor<128xf32>
      %308 = stablehlo.subtract %305, %307 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %310 = stablehlo.clamp %297, %308, %309 : tensor<128xf32>
      mhlo.return %310 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %288 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %298 = stablehlo.iota dim = 0 : tensor<128xi32>
      %299 = stablehlo.convert %298 : (tensor<128xi32>) -> tensor<128xf32>
      %300 = stablehlo.maximum %297, %299 : tensor<128xf32>
      %301 = stablehlo.convert %300 : (tensor<128xf32>) -> tensor<128xi32>
      %302 = stablehlo.convert %301 : (tensor<128xi32>) -> tensor<128xf32>
      %303 = stablehlo.subtract %300, %302 : tensor<128xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.clamp %297, %303, %304 : tensor<128xf32>
      mhlo.return %305 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %289 = "mhlo.fusion"(%70#1, %arg184, %arg185, %arg192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<256x32xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x16384x32xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x16384x32xf32>
      %301 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x32xf32>) -> tensor<32x256xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<64x32x256xf32>
      %303 = stablehlo.dot_general %300, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x32xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
      mhlo.return %303 : tensor<64x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<256x32xf32>) -> tensor<64x16384x256xf32>
    %290 = "mhlo.fusion"(%142#1, %arg186, %arg187, %arg194) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %297 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %298 = stablehlo.multiply %arg208, %297 : tensor<64x4096x64xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %300 = stablehlo.add %298, %299 : tensor<64x4096x64xf32>
      %301 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
      %303 = stablehlo.dot_general %300, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x4096x64xf32>, tensor<64x64x256xf32>) -> tensor<64x4096x256xf32>
      mhlo.return %303 : tensor<64x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<64x4096x256xf32>
    %291 = "mhlo.fusion"(%1, %2, %280, %281, %282, %284, %285, %286, %287, %288, %289, %arg193, %290, %arg195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x1xf32>, %arg209: tensor<128xf32>, %arg210: tensor<64x256x256xf32>, %arg211: tensor<128x1xf32>, %arg212: tensor<128xf32>, %arg213: tensor<64x1024x256xf32>, %arg214: tensor<128x1xf32>, %arg215: tensor<128x1xf32>, %arg216: tensor<128xf32>, %arg217: tensor<128xf32>, %arg218: tensor<64x16384x256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<64x4096x256xf32>, %arg221: tensor<256xf32>):
      %297 = mhlo.bitcast %arg210 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %299 = stablehlo.iota dim = 0 : tensor<128xi32>
      %300 = stablehlo.convert %299 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %302 = stablehlo.add %300, %301 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %306 = stablehlo.add %304, %305 : tensor<128xf32>
      %307 = stablehlo.maximum %298, %306 : tensor<128xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %309 = stablehlo.convert %308 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %310 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %311 = stablehlo.compare LT, %309, %310 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_3 = stablehlo.constant dense<16> : tensor<i32>
      %312 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %313 = stablehlo.add %309, %312 : tensor<128x1xi32>
      %314 = stablehlo.select %311, %313, %309 : tensor<128x1xi1>, tensor<128x1xi32>
      %315 = mhlo.bitcast %314 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %317 = stablehlo.convert %307 : (tensor<128xf32>) -> tensor<128xi32>
      %318 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %319 = stablehlo.compare LT, %317, %318 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %320 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %321 = stablehlo.add %317, %320 : tensor<128xi32>
      %322 = stablehlo.select %319, %321, %317 : tensor<128xi1>, tensor<128xi32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %324 = stablehlo.concatenate %316, %323, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %326 = "stablehlo.gather"(%297, %325) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %c_4 = stablehlo.constant dense<15> : tensor<i32>
      %328 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %c_5 = stablehlo.constant dense<1> : tensor<i32>
      %329 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %330 = stablehlo.add %317, %329 : tensor<128xi32>
      %331 = stablehlo.minimum %328, %330 : tensor<128xi32>
      %332 = stablehlo.compare LT, %331, %318 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %333 = stablehlo.add %331, %320 : tensor<128xi32>
      %334 = stablehlo.select %332, %333, %331 : tensor<128xi1>, tensor<128xi32>
      %335 = stablehlo.broadcast_in_dim %334, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %336 = stablehlo.concatenate %316, %335, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %337 = mhlo.bitcast %336 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %338 = "stablehlo.gather"(%297, %337) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %340 = stablehlo.subtract %339, %327 : tensor<64x256x128x128xf32>
      %341 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x256x128x128xf32>
      %343 = stablehlo.add %327, %342 : tensor<64x256x128x128xf32>
      %344 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %345 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %346 = stablehlo.add %309, %345 : tensor<128x1xi32>
      %347 = stablehlo.minimum %344, %346 : tensor<128x1xi32>
      %348 = stablehlo.compare LT, %347, %310 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %349 = stablehlo.add %347, %312 : tensor<128x1xi32>
      %350 = stablehlo.select %348, %349, %347 : tensor<128x1xi1>, tensor<128x1xi32>
      %351 = mhlo.bitcast %350 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %352 = stablehlo.broadcast_in_dim %351, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %353 = stablehlo.concatenate %352, %323, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %354 = mhlo.bitcast %353 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %355 = "stablehlo.gather"(%297, %354) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %356 = mhlo.bitcast %355 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %357 = stablehlo.concatenate %352, %335, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %358 = mhlo.bitcast %357 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %359 = "stablehlo.gather"(%297, %358) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %360 = mhlo.bitcast %359 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %361 = stablehlo.subtract %360, %356 : tensor<64x256x128x128xf32>
      %362 = stablehlo.multiply %361, %341 : tensor<64x256x128x128xf32>
      %363 = stablehlo.add %356, %362 : tensor<64x256x128x128xf32>
      %364 = stablehlo.subtract %363, %343 : tensor<64x256x128x128xf32>
      %365 = mhlo.bitcast %arg208 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %366 = stablehlo.broadcast_in_dim %365, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %367 = stablehlo.multiply %364, %366 : tensor<64x256x128x128xf32>
      %368 = stablehlo.add %343, %367 : tensor<64x256x128x128xf32>
      %369 = mhlo.bitcast %arg213 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32,32]{1,3,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x32x32xf32>
      %cst_6 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %371 = stablehlo.multiply %302, %370 : tensor<128xf32>
      %372 = stablehlo.add %371, %305 : tensor<128xf32>
      %373 = stablehlo.maximum %298, %372 : tensor<128xf32>
      %374 = mhlo.bitcast %373 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %375 = stablehlo.convert %374 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %376 = stablehlo.compare LT, %375, %310 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<32> : tensor<i32>
      %377 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %378 = stablehlo.add %375, %377 : tensor<128x1xi32>
      %379 = stablehlo.select %376, %378, %375 : tensor<128x1xi1>, tensor<128x1xi32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %382 = stablehlo.convert %373 : (tensor<128xf32>) -> tensor<128xi32>
      %383 = stablehlo.compare LT, %382, %318 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %384 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %385 = stablehlo.add %382, %384 : tensor<128xi32>
      %386 = stablehlo.select %383, %385, %382 : tensor<128xi1>, tensor<128xi32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %388 = stablehlo.concatenate %381, %387, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %389 = mhlo.bitcast %388 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %390 = "stablehlo.gather"(%369, %389) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %391 = mhlo.bitcast %390 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %c_8 = stablehlo.constant dense<31> : tensor<i32>
      %392 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %393 = stablehlo.add %382, %329 : tensor<128xi32>
      %394 = stablehlo.minimum %392, %393 : tensor<128xi32>
      %395 = stablehlo.compare LT, %394, %318 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %396 = stablehlo.add %394, %384 : tensor<128xi32>
      %397 = stablehlo.select %395, %396, %394 : tensor<128xi1>, tensor<128xi32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %399 = stablehlo.concatenate %381, %398, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %401 = "stablehlo.gather"(%369, %400) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %403 = stablehlo.subtract %402, %391 : tensor<64x256x128x128xf32>
      %404 = stablehlo.broadcast_in_dim %arg212, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<64x256x128x128xf32>
      %406 = stablehlo.add %391, %405 : tensor<64x256x128x128xf32>
      %407 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %408 = stablehlo.add %375, %345 : tensor<128x1xi32>
      %409 = stablehlo.minimum %407, %408 : tensor<128x1xi32>
      %410 = stablehlo.compare LT, %409, %310 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %411 = stablehlo.add %409, %377 : tensor<128x1xi32>
      %412 = stablehlo.select %410, %411, %409 : tensor<128x1xi1>, tensor<128x1xi32>
      %413 = mhlo.bitcast %412 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %414 = stablehlo.broadcast_in_dim %413, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %415 = stablehlo.concatenate %414, %387, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %417 = "stablehlo.gather"(%369, %416) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %419 = stablehlo.concatenate %414, %398, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %421 = "stablehlo.gather"(%369, %420) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %423 = stablehlo.subtract %422, %418 : tensor<64x256x128x128xf32>
      %424 = stablehlo.multiply %423, %404 : tensor<64x256x128x128xf32>
      %425 = stablehlo.add %418, %424 : tensor<64x256x128x128xf32>
      %426 = stablehlo.subtract %425, %406 : tensor<64x256x128x128xf32>
      %427 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %428 = stablehlo.broadcast_in_dim %427, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %429 = stablehlo.multiply %426, %428 : tensor<64x256x128x128xf32>
      %430 = stablehlo.add %406, %429 : tensor<64x256x128x128xf32>
      %431 = stablehlo.broadcast_in_dim %arg221, dims = [2] : (tensor<256xf32>) -> tensor<64x4096x256xf32>
      %432 = stablehlo.add %arg220, %431 : tensor<64x4096x256xf32>
      %433 = mhlo.bitcast %432 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x4096x256xf32>) -> tensor<64x256x64x64xf32>
      %434 = stablehlo.multiply %302, %301 : tensor<128xf32>
      %435 = stablehlo.add %434, %305 : tensor<128xf32>
      %436 = stablehlo.maximum %298, %435 : tensor<128xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %438 = stablehlo.convert %437 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %439 = stablehlo.compare LT, %438, %310 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<64> : tensor<i32>
      %440 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %441 = stablehlo.add %438, %440 : tensor<128x1xi32>
      %442 = stablehlo.select %439, %441, %438 : tensor<128x1xi1>, tensor<128x1xi32>
      %443 = mhlo.bitcast %442 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %444 = stablehlo.broadcast_in_dim %443, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %445 = stablehlo.convert %436 : (tensor<128xf32>) -> tensor<128xi32>
      %446 = stablehlo.compare LT, %445, %318 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %447 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %448 = stablehlo.add %445, %447 : tensor<128xi32>
      %449 = stablehlo.select %446, %448, %445 : tensor<128xi1>, tensor<128xi32>
      %450 = stablehlo.broadcast_in_dim %449, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %451 = stablehlo.concatenate %444, %450, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %452 = mhlo.bitcast %451 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %453 = "stablehlo.gather"(%433, %452) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %454 = mhlo.bitcast %453 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %c_10 = stablehlo.constant dense<63> : tensor<i32>
      %455 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %456 = stablehlo.add %445, %329 : tensor<128xi32>
      %457 = stablehlo.minimum %455, %456 : tensor<128xi32>
      %458 = stablehlo.compare LT, %457, %318 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %459 = stablehlo.add %457, %447 : tensor<128xi32>
      %460 = stablehlo.select %458, %459, %457 : tensor<128xi1>, tensor<128xi32>
      %461 = stablehlo.broadcast_in_dim %460, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %462 = stablehlo.concatenate %444, %461, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %464 = "stablehlo.gather"(%433, %463) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %466 = stablehlo.subtract %465, %454 : tensor<64x256x128x128xf32>
      %467 = stablehlo.broadcast_in_dim %arg216, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<64x256x128x128xf32>
      %469 = stablehlo.add %454, %468 : tensor<64x256x128x128xf32>
      %470 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %471 = stablehlo.add %438, %345 : tensor<128x1xi32>
      %472 = stablehlo.minimum %470, %471 : tensor<128x1xi32>
      %473 = stablehlo.compare LT, %472, %310 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %474 = stablehlo.add %472, %440 : tensor<128x1xi32>
      %475 = stablehlo.select %473, %474, %472 : tensor<128x1xi1>, tensor<128x1xi32>
      %476 = mhlo.bitcast %475 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %477 = stablehlo.broadcast_in_dim %476, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %478 = stablehlo.concatenate %477, %450, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %479 = mhlo.bitcast %478 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %480 = "stablehlo.gather"(%433, %479) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %481 = mhlo.bitcast %480 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %482 = stablehlo.concatenate %477, %461, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %483 = mhlo.bitcast %482 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %484 = "stablehlo.gather"(%433, %483) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %485 = mhlo.bitcast %484 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %486 = stablehlo.subtract %485, %481 : tensor<64x256x128x128xf32>
      %487 = stablehlo.multiply %486, %467 : tensor<64x256x128x128xf32>
      %488 = stablehlo.add %481, %487 : tensor<64x256x128x128xf32>
      %489 = stablehlo.subtract %488, %469 : tensor<64x256x128x128xf32>
      %490 = mhlo.bitcast %arg214 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %491 = stablehlo.broadcast_in_dim %490, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %492 = stablehlo.multiply %489, %491 : tensor<64x256x128x128xf32>
      %493 = stablehlo.add %469, %492 : tensor<64x256x128x128xf32>
      %494 = stablehlo.broadcast_in_dim %arg219, dims = [2] : (tensor<256xf32>) -> tensor<64x16384x256xf32>
      %495 = stablehlo.add %arg218, %494 : tensor<64x16384x256xf32>
      %496 = mhlo.bitcast %495 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,128,128]{1,3,2,0}"} : (tensor<64x16384x256xf32>) -> tensor<64x256x128x128xf32>
      %497 = stablehlo.maximum %298, %300 : tensor<128xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %499 = stablehlo.convert %498 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %500 = stablehlo.compare LT, %499, %310 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_11 = stablehlo.constant dense<128> : tensor<i32>
      %501 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %502 = stablehlo.add %499, %501 : tensor<128x1xi32>
      %503 = stablehlo.select %500, %502, %499 : tensor<128x1xi1>, tensor<128x1xi32>
      %504 = mhlo.bitcast %503 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %505 = stablehlo.broadcast_in_dim %504, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %506 = stablehlo.convert %497 : (tensor<128xf32>) -> tensor<128xi32>
      %507 = stablehlo.compare LT, %506, %318 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %508 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %509 = stablehlo.add %506, %508 : tensor<128xi32>
      %510 = stablehlo.select %507, %509, %506 : tensor<128xi1>, tensor<128xi32>
      %511 = stablehlo.broadcast_in_dim %510, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %512 = stablehlo.concatenate %505, %511, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %513 = mhlo.bitcast %512 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %514 = "stablehlo.gather"(%496, %513) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %515 = mhlo.bitcast %514 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %c_12 = stablehlo.constant dense<127> : tensor<i32>
      %516 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %517 = stablehlo.add %506, %329 : tensor<128xi32>
      %518 = stablehlo.minimum %516, %517 : tensor<128xi32>
      %519 = stablehlo.compare LT, %518, %318 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %520 = stablehlo.add %518, %508 : tensor<128xi32>
      %521 = stablehlo.select %519, %520, %518 : tensor<128xi1>, tensor<128xi32>
      %522 = stablehlo.broadcast_in_dim %521, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %523 = stablehlo.concatenate %505, %522, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %525 = "stablehlo.gather"(%496, %524) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %526 = mhlo.bitcast %525 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %527 = stablehlo.subtract %526, %515 : tensor<64x256x128x128xf32>
      %528 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %529 = stablehlo.multiply %527, %528 : tensor<64x256x128x128xf32>
      %530 = stablehlo.add %515, %529 : tensor<64x256x128x128xf32>
      %531 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %532 = stablehlo.add %499, %345 : tensor<128x1xi32>
      %533 = stablehlo.minimum %531, %532 : tensor<128x1xi32>
      %534 = stablehlo.compare LT, %533, %310 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %535 = stablehlo.add %533, %501 : tensor<128x1xi32>
      %536 = stablehlo.select %534, %535, %533 : tensor<128x1xi1>, tensor<128x1xi32>
      %537 = mhlo.bitcast %536 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %538 = stablehlo.broadcast_in_dim %537, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %539 = stablehlo.concatenate %538, %511, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %540 = mhlo.bitcast %539 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %541 = "stablehlo.gather"(%496, %540) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %542 = mhlo.bitcast %541 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %543 = stablehlo.concatenate %538, %522, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %544 = mhlo.bitcast %543 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %545 = "stablehlo.gather"(%496, %544) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,64,256,1,1]{0,2,1,4,3}"} : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %546 = mhlo.bitcast %545 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<64x256x128x128xf32>
      %547 = stablehlo.subtract %546, %542 : tensor<64x256x128x128xf32>
      %548 = stablehlo.multiply %547, %528 : tensor<64x256x128x128xf32>
      %549 = stablehlo.add %542, %548 : tensor<64x256x128x128xf32>
      %550 = stablehlo.subtract %549, %530 : tensor<64x256x128x128xf32>
      %551 = mhlo.bitcast %arg215 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %552 = stablehlo.broadcast_in_dim %551, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %553 = stablehlo.multiply %550, %552 : tensor<64x256x128x128xf32>
      %554 = stablehlo.add %530, %553 : tensor<64x256x128x128xf32>
      %555 = stablehlo.concatenate %368, %430, %493, %554, dim = 1 : (tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>) -> tensor<64x1024x128x128xf32>
      mhlo.return %555 : tensor<64x1024x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1xf32>, tensor<128xf32>, tensor<64x256x256xf32>, tensor<128x1xf32>, tensor<128xf32>, tensor<64x1024x256xf32>, tensor<128x1xf32>, tensor<128x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x16384x256xf32>, tensor<256xf32>, tensor<64x4096x256xf32>, tensor<256xf32>) -> tensor<64x1024x128x128xf32>
    %292 = stablehlo.custom_call @__cudnn$convForward(%291, %arg200) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x1024x128x128xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<64x256x128x128xf32>, tensor<0xui8>>
    %293 = stablehlo.get_tuple_element %292[0] : (tuple<tensor<64x256x128x128xf32>, tensor<0xui8>>) -> tensor<64x256x128x128xf32>
    %294 = "mhlo.fusion"(%arg202, %arg201, %0, %293, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<64x256x128x128xf32>, %arg212: tensor<256xf32>):
      %297 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<64x256x128x128xf32>
      %298 = stablehlo.subtract %arg211, %297 : tensor<64x256x128x128xf32>
      %299 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<64x256x128x128xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<64x256x128x128xf32>
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<64x256x128x128xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<64x256x128x128xf32>
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<256xf32>) -> tensor<64x256x128x128xf32>
      %304 = stablehlo.add %302, %303 : tensor<64x256x128x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x128x128xf32>
      %306 = stablehlo.maximum %304, %305 : tensor<64x256x128x128xf32>
      mhlo.return %306 : tensor<64x256x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x128x128xf32>, tensor<256xf32>) -> tensor<64x256x128x128xf32>
    %295 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%294, %arg203, %arg204) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,150,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x128x128xf32>, tensor<150x256x1x1xf32>, tensor<150xf32>) -> tuple<tensor<64x150x128x128xf32>, tensor<0xui8>>
    %296 = stablehlo.get_tuple_element %295[0] : (tuple<tensor<64x150x128x128xf32>, tensor<0xui8>>) -> tensor<64x150x128x128xf32>
    return %296 : tensor<64x150x128x128xf32>
  }
}
