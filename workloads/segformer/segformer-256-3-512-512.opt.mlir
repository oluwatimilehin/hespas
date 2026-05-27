module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.68_computation(%arg0: tensor<4194304x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>) -> tensor<4194304x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4194304x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<4194304x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
    return %4 : tensor<4194304x32xf32>
  }
  func.func private @gemm_fusion_dot_computation(%arg0: tensor<256x32x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x64xf32>) -> tensor<65536x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<256x32x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<256x32x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x32x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x32x256xf32>) -> tensor<256x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x32xf32>) -> tensor<65536x32xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<65536x32xf32>, tensor<32x64xf32>) -> tensor<65536x64xf32>
    return %6 : tensor<65536x64xf32>
  }
  func.func private @gemm_fusion_dot_general.71_computation(%arg0: tensor<32xf32>, %arg1: tensor<256x16384x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<256x256x32xf32>) -> tensor<256x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<256x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<256x256x32xf32>
    %3 = stablehlo.add %2, %arg3 : tensor<256x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x256x32xf32>) -> tensor<256x32x256xf32>
    %5 = stablehlo.dot_general %1, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x32xf32>, tensor<256x32x256xf32>) -> tensor<256x16384x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16384x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x16384x256xf32>
    return %7 : tensor<256x16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.72_computation(%arg0: tensor<256x16384x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<256x256x32xf32>) -> tensor<256x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<256x256x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<256x256x32xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x256xf32>, tensor<256x256x32xf32>) -> tensor<256x16384x32xf32>
    return %2 : tensor<256x16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.73_computation(%arg0: tensor<4194304x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<4194304x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
    return %0 : tensor<4194304x32xf32>
  }
  func.func private @gemm_fusion_dot_general.74_computation(%arg0: tensor<4194304x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x32xf32>) -> tensor<4194304x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4194304x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<4194304x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<128x32xf32>) -> tensor<4194304x128xf32>
    return %4 : tensor<4194304x128xf32>
  }
  func.func private @gemm_fusion_dot_general.75_computation(%arg0: tensor<256x128x16384xf32>, %arg1: tensor<32x128xf32>) -> tensor<4194304x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<256x128x16384xf32>) -> tensor<256x16384x128xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x128xf32>) -> tensor<4194304x128xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x128xf32>, tensor<32x128xf32>) -> tensor<4194304x32xf32>
    return %2 : tensor<4194304x32xf32>
  }
  func.func private @gemm_fusion_dot_general.76_computation(%arg0: tensor<4194304x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>) -> tensor<4194304x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4194304x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<4194304x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
    return %4 : tensor<4194304x32xf32>
  }
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<256x32x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x64xf32>) -> tensor<65536x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<256x32x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<256x32x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x32x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x32x256xf32>) -> tensor<256x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x32xf32>) -> tensor<65536x32xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<65536x32xf32>, tensor<32x64xf32>) -> tensor<65536x64xf32>
    return %6 : tensor<65536x64xf32>
  }
  func.func private @gemm_fusion_dot_general.79_computation(%arg0: tensor<32xf32>, %arg1: tensor<256x16384x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<256x256x32xf32>) -> tensor<256x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<256x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<256x256x32xf32>
    %3 = stablehlo.add %2, %arg3 : tensor<256x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x256x32xf32>) -> tensor<256x32x256xf32>
    %5 = stablehlo.dot_general %1, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x32xf32>, tensor<256x32x256xf32>) -> tensor<256x16384x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16384x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x16384x256xf32>
    return %7 : tensor<256x16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.80_computation(%arg0: tensor<256x16384x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<256x256x32xf32>) -> tensor<256x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<256x256x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<256x256x32xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x256xf32>, tensor<256x256x32xf32>) -> tensor<256x16384x32xf32>
    return %2 : tensor<256x16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.81_computation(%arg0: tensor<4194304x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<4194304x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
    return %0 : tensor<4194304x32xf32>
  }
  func.func private @gemm_fusion_dot_general.82_computation(%arg0: tensor<4194304x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x32xf32>) -> tensor<4194304x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4194304x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<4194304x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<128x32xf32>) -> tensor<4194304x128xf32>
    return %4 : tensor<4194304x128xf32>
  }
  func.func private @gemm_fusion_dot_general.83_computation(%arg0: tensor<256x128x16384xf32>, %arg1: tensor<32x128xf32>) -> tensor<4194304x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<256x128x16384xf32>) -> tensor<256x16384x128xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x128xf32>) -> tensor<4194304x128xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x128xf32>, tensor<32x128xf32>) -> tensor<4194304x32xf32>
    return %2 : tensor<4194304x32xf32>
  }
  func.func private @gemm_fusion_dot_general.84_computation(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>) -> tensor<1048576x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1048576x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<1048576x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
    return %4 : tensor<1048576x64xf32>
  }
  func.func private @gemm_fusion_dot.2_computation(%arg0: tensor<256x64x256xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<65536x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x64x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x64x256xf32>) -> tensor<256x256x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x64xf32>) -> tensor<65536x64xf32>
    %6 = stablehlo.concatenate %arg3, %arg4, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %8 = stablehlo.dot %5, %7, precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<64x128xf32>) -> tensor<65536x128xf32>
    return %8 : tensor<65536x128xf32>
  }
  func.func private @gemm_fusion_dot_general.87_computation(%arg0: tensor<512x4096x32xf32>, %arg1: tensor<512x32x256xf32>) -> tensor<512x4096x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x4096x32xf32>, tensor<512x32x256xf32>) -> tensor<512x4096x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x4096x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x4096x256xf32>
    return %2 : tensor<512x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.88_computation(%arg0: tensor<512x4096x256xf32>, %arg1: tensor<512x256x32xf32>) -> tensor<512x4096x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x4096x256xf32>, tensor<512x256x32xf32>) -> tensor<512x4096x32xf32>
    return %0 : tensor<512x4096x32xf32>
  }
  func.func private @gemm_fusion_dot_general.89_computation(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<1048576x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
    return %0 : tensor<1048576x64xf32>
  }
  func.func private @gemm_fusion_dot_general.90_computation(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<1048576x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1048576x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<1048576x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<256x64xf32>) -> tensor<1048576x256xf32>
    return %4 : tensor<1048576x256xf32>
  }
  func.func private @gemm_fusion_dot_general.91_computation(%arg0: tensor<256x256x4096xf32>, %arg1: tensor<64x256xf32>) -> tensor<1048576x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<256x256x4096xf32>) -> tensor<256x4096x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x256xf32>) -> tensor<1048576x256xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x256xf32>, tensor<64x256xf32>) -> tensor<1048576x64xf32>
    return %2 : tensor<1048576x64xf32>
  }
  func.func private @gemm_fusion_dot_general.92_computation(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>) -> tensor<1048576x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1048576x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<1048576x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
    return %4 : tensor<1048576x64xf32>
  }
  func.func private @gemm_fusion_dot.3_computation(%arg0: tensor<256x64x256xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<65536x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x64x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x64x256xf32>) -> tensor<256x256x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x64xf32>) -> tensor<65536x64xf32>
    %6 = stablehlo.concatenate %arg3, %arg4, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %8 = stablehlo.dot %5, %7, precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<64x128xf32>) -> tensor<65536x128xf32>
    return %8 : tensor<65536x128xf32>
  }
  func.func private @gemm_fusion_dot_general.95_computation(%arg0: tensor<512x4096x32xf32>, %arg1: tensor<512x32x256xf32>) -> tensor<512x4096x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x4096x32xf32>, tensor<512x32x256xf32>) -> tensor<512x4096x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x4096x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x4096x256xf32>
    return %2 : tensor<512x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.96_computation(%arg0: tensor<512x4096x256xf32>, %arg1: tensor<512x256x32xf32>) -> tensor<512x4096x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x4096x256xf32>, tensor<512x256x32xf32>) -> tensor<512x4096x32xf32>
    return %0 : tensor<512x4096x32xf32>
  }
  func.func private @gemm_fusion_dot_general.97_computation(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<1048576x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
    return %0 : tensor<1048576x64xf32>
  }
  func.func private @gemm_fusion_dot_general.98_computation(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<1048576x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1048576x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<1048576x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<256x64xf32>) -> tensor<1048576x256xf32>
    return %4 : tensor<1048576x256xf32>
  }
  func.func private @gemm_fusion_dot_general.99_computation(%arg0: tensor<256x256x4096xf32>, %arg1: tensor<64x256xf32>) -> tensor<1048576x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<256x256x4096xf32>) -> tensor<256x4096x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x256xf32>) -> tensor<1048576x256xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x256xf32>, tensor<64x256xf32>) -> tensor<1048576x64xf32>
    return %2 : tensor<1048576x64xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<262144x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<262144x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<262144x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<262144x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
    return %4 : tensor<262144x160xf32>
  }
  func.func private @gemm_fusion_dot_general.101_computation(%arg0: tensor<256x160x256xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x160x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x160x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x160x256xf32>) -> tensor<256x256x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x160xf32>) -> tensor<65536x160xf32>
    %6 = stablehlo.dot_general %5, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    return %6 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<1280x1024x32xf32>, %arg1: tensor<1280x32x256xf32>) -> tensor<1280x1024x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x1024x32xf32>, tensor<1280x32x256xf32>) -> tensor<1280x1024x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280x1024x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1280x1024x256xf32>
    return %2 : tensor<1280x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.102_computation(%arg0: tensor<256x160x256xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x160x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x160x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x160x256xf32>) -> tensor<256x256x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x160xf32>) -> tensor<65536x160xf32>
    %6 = stablehlo.dot_general %5, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    return %6 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<1280x1024x256xf32>, %arg1: tensor<1280x256x32xf32>) -> tensor<1280x1024x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x1024x256xf32>, tensor<1280x256x32xf32>) -> tensor<1280x1024x32xf32>
    return %0 : tensor<1280x1024x32xf32>
  }
  func.func private @gemm_fusion_dot_general.105_computation(%arg0: tensor<262144x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<262144x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
    return %0 : tensor<262144x160xf32>
  }
  func.func private @gemm_fusion_dot_general.106_computation(%arg0: tensor<262144x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<640x160xf32>) -> tensor<262144x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<262144x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<262144x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<640x160xf32>) -> tensor<262144x640xf32>
    return %4 : tensor<262144x640xf32>
  }
  func.func private @gemm_fusion_dot_general.107_computation(%arg0: tensor<256x640x1024xf32>, %arg1: tensor<160x640xf32>) -> tensor<262144x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<256x640x1024xf32>) -> tensor<256x1024x640xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x640xf32>) -> tensor<262144x640xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x640xf32>, tensor<160x640xf32>) -> tensor<262144x160xf32>
    return %2 : tensor<262144x160xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<262144x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<262144x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<262144x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<262144x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
    return %4 : tensor<262144x160xf32>
  }
  func.func private @gemm_fusion_dot_general.109_computation(%arg0: tensor<256x160x256xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x160x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x160x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x160x256xf32>) -> tensor<256x256x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x160xf32>) -> tensor<65536x160xf32>
    %6 = stablehlo.dot_general %5, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    return %6 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<1280x1024x32xf32>, %arg1: tensor<1280x32x256xf32>) -> tensor<1280x1024x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x1024x32xf32>, tensor<1280x32x256xf32>) -> tensor<1280x1024x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280x1024x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1280x1024x256xf32>
    return %2 : tensor<1280x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.110_computation(%arg0: tensor<256x160x256xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x160x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x160x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x160x256xf32>) -> tensor<256x256x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x160xf32>) -> tensor<65536x160xf32>
    %6 = stablehlo.dot_general %5, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    return %6 : tensor<65536x160xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<1280x1024x256xf32>, %arg1: tensor<1280x256x32xf32>) -> tensor<1280x1024x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x1024x256xf32>, tensor<1280x256x32xf32>) -> tensor<1280x1024x32xf32>
    return %0 : tensor<1280x1024x32xf32>
  }
  func.func private @gemm_fusion_dot_general.113_computation(%arg0: tensor<262144x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<262144x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
    return %0 : tensor<262144x160xf32>
  }
  func.func private @gemm_fusion_dot_general.114_computation(%arg0: tensor<262144x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<640x160xf32>) -> tensor<262144x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<262144x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<262144x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<640x160xf32>) -> tensor<262144x640xf32>
    return %4 : tensor<262144x640xf32>
  }
  func.func private @gemm_fusion_dot_general.115_computation(%arg0: tensor<256x640x1024xf32>, %arg1: tensor<160x640xf32>) -> tensor<262144x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<256x640x1024xf32>) -> tensor<256x1024x640xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x640xf32>) -> tensor<262144x640xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x640xf32>, tensor<160x640xf32>) -> tensor<262144x160xf32>
    return %2 : tensor<262144x160xf32>
  }
  func.func private @gemm_fusion_dot_general.116_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    return %4 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.117_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    return %4 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<2048x256x32xf32>, %arg1: tensor<2048x32x256xf32>) -> tensor<2048x256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256x32xf32>, tensor<2048x32x256xf32>) -> tensor<2048x256x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048x256x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<2048x256x256xf32>
    return %2 : tensor<2048x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.118_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    return %4 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<2048x256x256xf32>, %arg1: tensor<2048x256x32xf32>) -> tensor<2048x256x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256x256xf32>, tensor<2048x256x32xf32>) -> tensor<2048x256x32xf32>
    return %0 : tensor<2048x256x32xf32>
  }
  func.func private @gemm_fusion_dot_general.121_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    return %0 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.122_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256xf32>) -> tensor<65536x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<1024x256xf32>) -> tensor<65536x1024xf32>
    return %4 : tensor<65536x1024xf32>
  }
  func.func private @gemm_fusion_dot_general.123_computation(%arg0: tensor<256x1024x256xf32>, %arg1: tensor<256x1024xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<256x1024x256xf32>) -> tensor<256x256x1024xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x1024xf32>) -> tensor<65536x1024xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x1024xf32>, tensor<256x1024xf32>) -> tensor<65536x256xf32>
    return %2 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.124_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    return %4 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.125_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    return %4 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<2048x256x32xf32>, %arg1: tensor<2048x32x256xf32>) -> tensor<2048x256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256x32xf32>, tensor<2048x32x256xf32>) -> tensor<2048x256x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048x256x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<2048x256x256xf32>
    return %2 : tensor<2048x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    return %4 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<2048x256x256xf32>, %arg1: tensor<2048x256x32xf32>) -> tensor<2048x256x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256x256xf32>, tensor<2048x256x32xf32>) -> tensor<2048x256x32xf32>
    return %0 : tensor<2048x256x32xf32>
  }
  func.func private @gemm_fusion_dot_general.129_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    return %0 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.130_computation(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256xf32>) -> tensor<65536x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<65536x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<65536x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<1024x256xf32>) -> tensor<65536x1024xf32>
    return %4 : tensor<65536x1024xf32>
  }
  func.func private @gemm_fusion_dot_general.131_computation(%arg0: tensor<256x1024x256xf32>, %arg1: tensor<256x1024xf32>) -> tensor<65536x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<256x1024x256xf32>) -> tensor<256x256x1024xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x1024xf32>) -> tensor<65536x1024xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x1024xf32>, tensor<256x1024xf32>) -> tensor<65536x256xf32>
    return %2 : tensor<65536x256xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<256x256x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<256x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x256x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x256x256xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256x256xf32>, tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
    return %6 : tensor<256x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<256x1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<256x160xf32>) -> tensor<256x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x1024x160xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x160xf32>) -> tensor<160x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<256x160x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1024x160xf32>, tensor<256x160x256xf32>) -> tensor<256x1024x256xf32>
    return %6 : tensor<256x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<256x4096x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<256x4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x4096x64xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<256x64x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x64xf32>, tensor<256x64x256xf32>) -> tensor<256x4096x256xf32>
    return %6 : tensor<256x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<256x16384x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<256x32xf32>) -> tensor<256x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x16384x32xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x32xf32>) -> tensor<32x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<256x32x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x32xf32>, tensor<256x32x256xf32>) -> tensor<256x16384x256xf32>
    return %6 : tensor<256x16384x256xf32>
  }
  func.func private @region_0.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.60(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x256x256xf32>, %arg3: tensor<65536x256xf32>, %arg4: tensor<256xf32>) -> tensor<256x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<65536x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<256x256x256xf32>
    %7 = stablehlo.add %2, %6 : tensor<256x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<256x256x256xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<256x256x256xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<256x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %17 = stablehlo.add %15, %16 : tensor<256x256xf32>
    %18 = stablehlo.rsqrt %17 : tensor<256x256xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<256x256x256xf32>
    return %20 : tensor<256x256x256xf32>
  }
  func.func private @region_0.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.62(%arg0: tensor<256x256x256xf32>, %arg1: tensor<65536x256xf32>, %arg2: tensor<256xf32>) -> tensor<256x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<256x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<256x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<256x256x256xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<256x256x256xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<256x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x256xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<256x256x256xf32>
    return %16 : tensor<256x256x256xf32>
  }
  func.func private @region_0.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.64(%arg0: tensor<2048x256x256xf32>) -> tensor<256x8x256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2048x256x256xf32>) -> tensor<256x8x256x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<2048x256x256xf32>, tensor<f32>) -> tensor<2048x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<256x8x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<256x8x256xf32>) -> tensor<256x8x256x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<256x8x256x256xf32>
    %5 = stablehlo.exponential %4 : tensor<256x8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x8x256x256xf32>, tensor<f32>) -> tensor<256x8x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<256x8x256xf32>) -> tensor<256x8x256x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<256x8x256x256xf32>
    return %8 : tensor<256x8x256x256xf32>
  }
  func.func private @region_0.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.66(%arg0: tensor<256x256x256xf32>) -> tensor<256x256x256xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<256x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<256x256x256xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<256x256x256xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<256x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<256x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<256x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<256x256x256xf32>
    return %12 : tensor<256x256x256xf32>
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
  func.func private @triton_softmax_computation.68(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x256x16x16xf32>, %arg3: tensor<65536x256xf32>, %arg4: tensor<256xf32>) -> tensor<256x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x16x16xf32>) -> tensor<256x256x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x256x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<256x256x256xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<256x256x256xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x256xf32>
    %17 = stablehlo.rsqrt %16 : tensor<256x256xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<256x256x256xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<256x256x256xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %23 = stablehlo.add %21, %22 : tensor<256x256x256xf32>
    %24 = stablehlo.add %2, %23 : tensor<256x256x256xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x256xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<256x256x256xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<256x256x256xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<256x256xf32>
    %32 = stablehlo.add %31, %15 : tensor<256x256xf32>
    %33 = stablehlo.rsqrt %32 : tensor<256x256xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<256x256x256xf32>
    return %35 : tensor<256x256x256xf32>
  }
  func.func private @region_0.1.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.70(%arg0: tensor<2048x256x256xf32>) -> tensor<256x8x256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2048x256x256xf32>) -> tensor<256x8x256x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<2048x256x256xf32>, tensor<f32>) -> tensor<2048x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<256x8x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<256x8x256xf32>) -> tensor<256x8x256x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<256x8x256x256xf32>
    %5 = stablehlo.exponential %4 : tensor<256x8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x8x256x256xf32>, tensor<f32>) -> tensor<256x8x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<256x8x256xf32>) -> tensor<256x8x256x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<256x8x256x256xf32>
    return %8 : tensor<256x8x256x256xf32>
  }
  func.func private @region_0.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.72(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x256x16x16xf32>) -> tensor<256x256x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x16x16xf32>) -> tensor<256x256x256xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x256x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<256x16x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<256x256x256xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<256x256x256xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<256x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x256xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<256x256x256xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<256x256x256xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %20 = stablehlo.add %18, %19 : tensor<256x256x256xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<256x256xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<256x256x256xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<256x256x256xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<256x256xf32>
    %28 = stablehlo.add %27, %12 : tensor<256x256xf32>
    %29 = stablehlo.rsqrt %28 : tensor<256x256xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<256x256x256xf32>
    return %31 : tensor<256x256x256xf32>
  }
  func.func private @region_0.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.77(%arg0: tensor<256x1024x160xf32>, %arg1: tensor<262144x160xf32>, %arg2: tensor<160xf32>) -> tensor<256x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<262144x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<256x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<256x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<256x1024x160xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<256x1024x160xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<256x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x1024xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256x1024xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<256x1024x160xf32>
    return %16 : tensor<256x1024x160xf32>
  }
  func.func private @region_0.1.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.79(%arg0: tensor<1280x1024x256xf32>) -> tensor<256x5x1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1280x1024x256xf32>) -> tensor<256x5x1024x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1280x1024x256xf32>, tensor<f32>) -> tensor<1280x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1280x1024xf32>) -> tensor<256x5x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<256x5x1024xf32>) -> tensor<256x5x1024x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<256x5x1024x256xf32>
    %5 = stablehlo.exponential %4 : tensor<256x5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x5x1024x256xf32>, tensor<f32>) -> tensor<256x5x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<256x5x1024xf32>) -> tensor<256x5x1024x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<256x5x1024x256xf32>
    return %8 : tensor<256x5x1024x256xf32>
  }
  func.func private @region_0.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.81(%arg0: tensor<256x1024x160xf32>) -> tensor<256x1024x160xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<256x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<256x1024x160xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<256x1024x160xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<256x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %9 = stablehlo.add %7, %8 : tensor<256x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<256x1024xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<256x1024x160xf32>
    return %12 : tensor<256x1024x160xf32>
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
  func.func private @triton_softmax_computation.83(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<256x160x32x32xf32>, %arg3: tensor<262144x160xf32>, %arg4: tensor<160xf32>) -> tensor<256x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<262144x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x32x32xf32>) -> tensor<256x160x1024xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x160x1024xf32>) -> tensor<256x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x32x32xf32>, tensor<f32>) -> tensor<256x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x32xf32>) -> tensor<256x1024xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<256x1024x160xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<256x1024x160xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x1024xf32>
    %17 = stablehlo.rsqrt %16 : tensor<256x1024xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<256x1024x160xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<256x1024x160xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %23 = stablehlo.add %21, %22 : tensor<256x1024x160xf32>
    %24 = stablehlo.add %2, %23 : tensor<256x1024x160xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x1024xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<256x1024x160xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<256x1024x160xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<256x1024xf32>
    %32 = stablehlo.add %31, %15 : tensor<256x1024xf32>
    %33 = stablehlo.rsqrt %32 : tensor<256x1024xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<256x1024x160xf32>
    return %35 : tensor<256x1024x160xf32>
  }
  func.func private @region_0.1.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.85(%arg0: tensor<1280x1024x256xf32>) -> tensor<256x5x1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1280x1024x256xf32>) -> tensor<256x5x1024x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1280x1024x256xf32>, tensor<f32>) -> tensor<1280x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1280x1024xf32>) -> tensor<256x5x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<256x5x1024xf32>) -> tensor<256x5x1024x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<256x5x1024x256xf32>
    %5 = stablehlo.exponential %4 : tensor<256x5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x5x1024x256xf32>, tensor<f32>) -> tensor<256x5x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<256x5x1024xf32>) -> tensor<256x5x1024x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<256x5x1024x256xf32>
    return %8 : tensor<256x5x1024x256xf32>
  }
  func.func private @region_0.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.87(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<256x160x32x32xf32>) -> tensor<256x1024x160xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x32x32xf32>) -> tensor<256x160x1024xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<256x160x1024xf32>) -> tensor<256x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x32x32xf32>, tensor<f32>) -> tensor<256x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x32x32xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<256x32x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x32xf32>) -> tensor<256x1024xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<256x1024x160xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<256x1024x160xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<256x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x1024xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256x1024xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<256x1024x160xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<256x1024x160xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %20 = stablehlo.add %18, %19 : tensor<256x1024x160xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<256x1024xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<256x1024x160xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<256x1024x160xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<256x1024xf32>
    %28 = stablehlo.add %27, %12 : tensor<256x1024xf32>
    %29 = stablehlo.rsqrt %28 : tensor<256x1024xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<256x1024x160xf32>
    return %31 : tensor<256x1024x160xf32>
  }
  func.func private @region_0.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.92(%arg0: tensor<256x4096x64xf32>, %arg1: tensor<1048576x64xf32>, %arg2: tensor<64xf32>) -> tensor<256x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1048576x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<256x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<256x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<256x4096x64xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<256x4096x64xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<256x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x4096xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256x4096xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<256x4096x64xf32>
    return %16 : tensor<256x4096x64xf32>
  }
  func.func private @region_0.1.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.94(%arg0: tensor<512x4096x256xf32>) -> tensor<256x2x4096x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x4096x256xf32>) -> tensor<256x2x4096x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<512x4096x256xf32>, tensor<f32>) -> tensor<512x4096xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x4096xf32>) -> tensor<256x2x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<256x2x4096xf32>) -> tensor<256x2x4096x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<256x2x4096x256xf32>
    %5 = stablehlo.exponential %4 : tensor<256x2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x2x4096x256xf32>, tensor<f32>) -> tensor<256x2x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<256x2x4096xf32>) -> tensor<256x2x4096x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<256x2x4096x256xf32>
    return %8 : tensor<256x2x4096x256xf32>
  }
  func.func private @region_0.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.96(%arg0: tensor<256x4096x64xf32>) -> tensor<256x4096x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<256x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<256x4096x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<256x4096x64xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<256x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %9 = stablehlo.add %7, %8 : tensor<256x4096xf32>
    %10 = stablehlo.rsqrt %9 : tensor<256x4096xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<256x4096x64xf32>
    return %12 : tensor<256x4096x64xf32>
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
  func.func private @triton_softmax_computation.98(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x64x64x64xf32>, %arg3: tensor<1048576x64xf32>, %arg4: tensor<64xf32>) -> tensor<256x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<1048576x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x64x64xf32>) -> tensor<256x64x4096xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x64x4096xf32>) -> tensor<256x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x64x64xf32>, tensor<f32>) -> tensor<256x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x64xf32>) -> tensor<256x4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<256x4096x64xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<256x4096x64xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x4096xf32>
    %17 = stablehlo.rsqrt %16 : tensor<256x4096xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<256x4096x64xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<256x4096x64xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %23 = stablehlo.add %21, %22 : tensor<256x4096x64xf32>
    %24 = stablehlo.add %2, %23 : tensor<256x4096x64xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x4096xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<256x4096x64xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<256x4096x64xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<256x4096xf32>
    %32 = stablehlo.add %31, %15 : tensor<256x4096xf32>
    %33 = stablehlo.rsqrt %32 : tensor<256x4096xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<256x4096x64xf32>
    return %35 : tensor<256x4096x64xf32>
  }
  func.func private @region_0.1.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.100(%arg0: tensor<512x4096x256xf32>) -> tensor<256x2x4096x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x4096x256xf32>) -> tensor<256x2x4096x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<512x4096x256xf32>, tensor<f32>) -> tensor<512x4096xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x4096xf32>) -> tensor<256x2x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<256x2x4096xf32>) -> tensor<256x2x4096x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<256x2x4096x256xf32>
    %5 = stablehlo.exponential %4 : tensor<256x2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x2x4096x256xf32>, tensor<f32>) -> tensor<256x2x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<256x2x4096xf32>) -> tensor<256x2x4096x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<256x2x4096x256xf32>
    return %8 : tensor<256x2x4096x256xf32>
  }
  func.func private @region_0.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.102(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x64x64x64xf32>) -> tensor<256x4096x64xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x64x64xf32>) -> tensor<256x64x4096xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<256x64x4096xf32>) -> tensor<256x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x64x64xf32>, tensor<f32>) -> tensor<256x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x64x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<256x64x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x64xf32>) -> tensor<256x4096xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<256x4096x64xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<256x4096x64xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<256x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x4096xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256x4096xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<256x4096x64xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<256x4096x64xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %20 = stablehlo.add %18, %19 : tensor<256x4096x64xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<256x4096xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<256x4096x64xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<256x4096x64xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<256x4096xf32>
    %28 = stablehlo.add %27, %12 : tensor<256x4096xf32>
    %29 = stablehlo.rsqrt %28 : tensor<256x4096xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<256x4096x64xf32>
    return %31 : tensor<256x4096x64xf32>
  }
  func.func private @region_0.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.107(%arg0: tensor<256x16384x32xf32>, %arg1: tensor<4194304x32xf32>, %arg2: tensor<32xf32>) -> tensor<256x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<4194304x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<256x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<256x16384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<256x16384x32xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<256x16384x32xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<256x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x16384xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256x16384xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<256x16384x32xf32>
    return %16 : tensor<256x16384x32xf32>
  }
  func.func private @region_0.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.109(%arg0: tensor<256x16384x256xf32>) -> tensor<256x16384x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<256x16384x256xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<256x16384x256xf32>
    %3 = stablehlo.exponential %2 : tensor<256x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x256xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<256x16384x256xf32>
    return %6 : tensor<256x16384x256xf32>
  }
  func.func private @region_0.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.111(%arg0: tensor<256x16384x32xf32>) -> tensor<256x16384x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<256x16384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<256x16384x32xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<256x16384x32xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<256x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %9 = stablehlo.add %7, %8 : tensor<256x16384xf32>
    %10 = stablehlo.rsqrt %9 : tensor<256x16384xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<256x16384x32xf32>
    return %12 : tensor<256x16384x32xf32>
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
  func.func private @triton_softmax_computation.113(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<256x32x128x128xf32>, %arg3: tensor<4194304x32xf32>, %arg4: tensor<32xf32>) -> tensor<256x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<4194304x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x128x128xf32>) -> tensor<256x32x16384xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x32x16384xf32>) -> tensor<256x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x128x128xf32>, tensor<f32>) -> tensor<256x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x128x128xf32>) -> tensor<256x16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<256x16384x32xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<256x16384x32xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x16384xf32>
    %17 = stablehlo.rsqrt %16 : tensor<256x16384xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<256x16384x32xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<256x16384x32xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %23 = stablehlo.add %21, %22 : tensor<256x16384x32xf32>
    %24 = stablehlo.add %2, %23 : tensor<256x16384x32xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x16384xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<256x16384x32xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<256x16384x32xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<256x16384xf32>
    %32 = stablehlo.add %31, %15 : tensor<256x16384xf32>
    %33 = stablehlo.rsqrt %32 : tensor<256x16384xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<256x16384x32xf32>
    return %35 : tensor<256x16384x32xf32>
  }
  func.func private @region_0.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.115(%arg0: tensor<256x16384x256xf32>) -> tensor<256x16384x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<256x16384x256xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<256x16384x256xf32>
    %3 = stablehlo.exponential %2 : tensor<256x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x256xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<256x16384x256xf32>
    return %6 : tensor<256x16384x256xf32>
  }
  func.func private @region_0.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.117(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<256x32x128x128xf32>) -> tensor<256x16384x32xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x128x128xf32>) -> tensor<256x32x16384xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<256x32x16384xf32>) -> tensor<256x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x128x128xf32>, tensor<f32>) -> tensor<256x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<256x128x128xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x128x128xf32>) -> tensor<256x16384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<256x16384x32xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<256x16384x32xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<256x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x16384xf32>
    %14 = stablehlo.rsqrt %13 : tensor<256x16384xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<256x16384x32xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<256x16384x32xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %20 = stablehlo.add %18, %19 : tensor<256x16384x32xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<256x16384xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<256x16384x32xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<256x16384x32xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<256x16384xf32>
    %28 = stablehlo.add %27, %12 : tensor<256x16384xf32>
    %29 = stablehlo.rsqrt %28 : tensor<256x16384xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<256x16384x32xf32>
    return %31 : tensor<256x16384x32xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<2048x256x32xf32>) -> tensor<256x256x8x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2048x256x32xf32>) -> tensor<256x8x256x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<256x8x256x32xf32>) -> tensor<256x256x8x32xf32>
    return %1 : tensor<256x256x8x32xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<2048x256x32xf32>) -> tensor<256x256x8x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2048x256x32xf32>) -> tensor<256x8x256x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<256x8x256x32xf32>) -> tensor<256x256x8x32xf32>
    return %1 : tensor<256x256x8x32xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<1280x1024x32xf32>) -> tensor<256x1024x5x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1280x1024x32xf32>) -> tensor<256x5x1024x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<256x5x1024x32xf32>) -> tensor<256x1024x5x32xf32>
    return %1 : tensor<256x1024x5x32xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<1280x1024x32xf32>) -> tensor<256x1024x5x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1280x1024x32xf32>) -> tensor<256x5x1024x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<256x5x1024x32xf32>) -> tensor<256x1024x5x32xf32>
    return %1 : tensor<256x1024x5x32xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<512x4096x32xf32>) -> tensor<256x4096x2x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x4096x32xf32>) -> tensor<256x2x4096x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<256x2x4096x32xf32>) -> tensor<256x4096x2x32xf32>
    return %1 : tensor<256x4096x2x32xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<512x4096x32xf32>) -> tensor<256x4096x2x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x4096x32xf32>) -> tensor<256x2x4096x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<256x2x4096x32xf32>) -> tensor<256x4096x2x32xf32>
    return %1 : tensor<256x4096x2x32xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<256x1024x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<256x1024x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x1024x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x16x16xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<256x1024x16x16xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<256x1024x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x16x16xf32>) -> tensor<256x1024x256xf32>
    %5 = stablehlo.negate %2 : tensor<256x1024x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x1024x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x16x16xf32>) -> tensor<256x1024x256xf32>
    %9 = stablehlo.abs %8 : tensor<256x1024x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<256x1024x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x1024x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x1024x256xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<256x1024x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x1024x256xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<256x1024x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %22 = stablehlo.add %20, %21 : tensor<256x1024x256xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<256x1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<256x1024x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x1024x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %28 = stablehlo.add %26, %27 : tensor<256x1024x256xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<256x1024x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<256x1024x256xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<256x1024x256xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<256x1024x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %37 = stablehlo.negate %13 : tensor<256x1024x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
    %40 = stablehlo.exponential %37 : tensor<256x1024x256xf32>
    %41 = stablehlo.divide %10, %9 : tensor<256x1024x256xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<256x1024x256xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %45 = stablehlo.divide %10, %13 : tensor<256x1024x256xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<256x1024x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %48 = stablehlo.add %46, %47 : tensor<256x1024x256xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<256x1024x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %51 = stablehlo.add %49, %50 : tensor<256x1024x256xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<256x1024x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x1024x256xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<256x1024x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %57 = stablehlo.add %55, %56 : tensor<256x1024x256xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<256x1024x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %60 = stablehlo.add %58, %59 : tensor<256x1024x256xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<256x1024x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %63 = stablehlo.add %61, %62 : tensor<256x1024x256xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<256x1024x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %66 = stablehlo.add %64, %65 : tensor<256x1024x256xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<256x1024x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %69 = stablehlo.add %67, %68 : tensor<256x1024x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<256x1024x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %73 = stablehlo.add %71, %72 : tensor<256x1024x256xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<256x1024x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %76 = stablehlo.add %74, %75 : tensor<256x1024x256xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<256x1024x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %79 = stablehlo.add %77, %78 : tensor<256x1024x256xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<256x1024x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %82 = stablehlo.add %80, %81 : tensor<256x1024x256xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<256x1024x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %85 = stablehlo.add %83, %84 : tensor<256x1024x256xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<256x1024x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %88 = stablehlo.add %86, %87 : tensor<256x1024x256xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<256x1024x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %91 = stablehlo.add %89, %90 : tensor<256x1024x256xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<256x1024x256xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<256x1024x256xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<256x1024x256xf32>
    return %98 : tensor<256x1024x256xf32>
  }
  func.func private @fused_transpose.6(%arg0: tensor<65536x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<256x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<65536x1024xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x1024xf32>) -> tensor<256x256x1024xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<256x256x1024xf32>) -> tensor<256x1024x256xf32>
    return %3 : tensor<256x1024x256xf32>
  }
  func.func private @fused_transpose.7(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x256x8x32xf32>) -> tensor<256x8x256x32xf32>
    return %3 : tensor<256x8x256x32xf32>
  }
  func.func private @fused_transpose.8(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x8x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] : (tensor<256x256x8x32xf32>) -> tensor<256x8x32x256xf32>
    return %3 : tensor<256x8x32x256xf32>
  }
  func.func private @fused_transpose.9(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x256x8x32xf32>) -> tensor<256x8x256x32xf32>
    return %3 : tensor<256x8x256x32xf32>
  }
  func.func private @fused_computation.69(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256x256x16x16xf32>, %arg5: tensor<65536x256xf32>, %arg6: tensor<256xf32>) -> tensor<256x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<65536x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x16x16xf32>) -> tensor<256x256x256xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x256x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x16x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<256x256x256xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<256x256x256xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<256x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x256xf32>
    %20 = stablehlo.rsqrt %19 : tensor<256x256xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<256x256x256xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<256x256x256xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %26 = stablehlo.add %24, %25 : tensor<256x256x256xf32>
    %27 = stablehlo.add %5, %26 : tensor<256x256x256xf32>
    %28 = stablehlo.add %2, %27 : tensor<256x256x256xf32>
    return %28 : tensor<256x256x256xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<256x1024x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<256x1024x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x1024x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x16x16xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<256x1024x16x16xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<256x1024x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x16x16xf32>) -> tensor<256x1024x256xf32>
    %5 = stablehlo.negate %2 : tensor<256x1024x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x1024x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x16x16xf32>) -> tensor<256x1024x256xf32>
    %9 = stablehlo.abs %8 : tensor<256x1024x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<256x1024x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x1024x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x1024x256xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<256x1024x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x1024x256xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<256x1024x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %22 = stablehlo.add %20, %21 : tensor<256x1024x256xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<256x1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<256x1024x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x1024x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %28 = stablehlo.add %26, %27 : tensor<256x1024x256xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<256x1024x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<256x1024x256xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<256x1024x256xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<256x1024x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %37 = stablehlo.negate %13 : tensor<256x1024x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
    %40 = stablehlo.exponential %37 : tensor<256x1024x256xf32>
    %41 = stablehlo.divide %10, %9 : tensor<256x1024x256xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<256x1024x256xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %45 = stablehlo.divide %10, %13 : tensor<256x1024x256xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<256x1024x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %48 = stablehlo.add %46, %47 : tensor<256x1024x256xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<256x1024x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %51 = stablehlo.add %49, %50 : tensor<256x1024x256xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<256x1024x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x1024x256xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<256x1024x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %57 = stablehlo.add %55, %56 : tensor<256x1024x256xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<256x1024x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %60 = stablehlo.add %58, %59 : tensor<256x1024x256xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<256x1024x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %63 = stablehlo.add %61, %62 : tensor<256x1024x256xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<256x1024x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %66 = stablehlo.add %64, %65 : tensor<256x1024x256xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<256x1024x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %69 = stablehlo.add %67, %68 : tensor<256x1024x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<256x1024x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %73 = stablehlo.add %71, %72 : tensor<256x1024x256xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<256x1024x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %76 = stablehlo.add %74, %75 : tensor<256x1024x256xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<256x1024x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %79 = stablehlo.add %77, %78 : tensor<256x1024x256xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<256x1024x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %82 = stablehlo.add %80, %81 : tensor<256x1024x256xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<256x1024x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %85 = stablehlo.add %83, %84 : tensor<256x1024x256xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<256x1024x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %88 = stablehlo.add %86, %87 : tensor<256x1024x256xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<256x1024x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
    %91 = stablehlo.add %89, %90 : tensor<256x1024x256xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<256x1024x256xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<256x1024x256xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<256x1024x256xf32>
    return %98 : tensor<256x1024x256xf32>
  }
  func.func private @fused_transpose.10(%arg0: tensor<65536x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<256x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<65536x1024xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x1024xf32>) -> tensor<256x256x1024xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<256x256x1024xf32>) -> tensor<256x1024x256xf32>
    return %3 : tensor<256x1024x256xf32>
  }
  func.func private @fused_transpose.11(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x256x8x32xf32>) -> tensor<256x8x256x32xf32>
    return %3 : tensor<256x8x256x32xf32>
  }
  func.func private @fused_transpose.12(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x8x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] : (tensor<256x256x8x32xf32>) -> tensor<256x8x32x256xf32>
    return %3 : tensor<256x8x32x256xf32>
  }
  func.func private @fused_transpose.13(%arg0: tensor<65536x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x256x8x32xf32>) -> tensor<256x8x256x32xf32>
    return %3 : tensor<256x8x256x32xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<256x640x32x32xf32>, %arg1: tensor<640xf32>) -> tensor<256x640x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x640x32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<256x640x32x32xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<256x640x32x32xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<256x640x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x640x32x32xf32>) -> tensor<256x640x1024xf32>
    %5 = stablehlo.negate %2 : tensor<256x640x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x640x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x640x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x640x32x32xf32>) -> tensor<256x640x1024xf32>
    %9 = stablehlo.abs %8 : tensor<256x640x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<256x640x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x640x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x640x1024xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<256x640x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x640x1024xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<256x640x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<256x640x1024xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<256x640x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<256x640x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x640x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<256x640x1024xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<256x640x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<256x640x1024xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<256x640x1024xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<256x640x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %37 = stablehlo.negate %13 : tensor<256x640x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
    %40 = stablehlo.exponential %37 : tensor<256x640x1024xf32>
    %41 = stablehlo.divide %10, %9 : tensor<256x640x1024xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<256x640x1024xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %45 = stablehlo.divide %10, %13 : tensor<256x640x1024xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<256x640x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %48 = stablehlo.add %46, %47 : tensor<256x640x1024xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<256x640x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %51 = stablehlo.add %49, %50 : tensor<256x640x1024xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<256x640x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x640x1024xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<256x640x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %57 = stablehlo.add %55, %56 : tensor<256x640x1024xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<256x640x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %60 = stablehlo.add %58, %59 : tensor<256x640x1024xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<256x640x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %63 = stablehlo.add %61, %62 : tensor<256x640x1024xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<256x640x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %66 = stablehlo.add %64, %65 : tensor<256x640x1024xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<256x640x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %69 = stablehlo.add %67, %68 : tensor<256x640x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<256x640x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %73 = stablehlo.add %71, %72 : tensor<256x640x1024xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<256x640x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<256x640x1024xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<256x640x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %79 = stablehlo.add %77, %78 : tensor<256x640x1024xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<256x640x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %82 = stablehlo.add %80, %81 : tensor<256x640x1024xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<256x640x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %85 = stablehlo.add %83, %84 : tensor<256x640x1024xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<256x640x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %88 = stablehlo.add %86, %87 : tensor<256x640x1024xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<256x640x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %91 = stablehlo.add %89, %90 : tensor<256x640x1024xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<256x640x1024xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<256x640x1024xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<256x640x1024xf32>
    return %98 : tensor<256x640x1024xf32>
  }
  func.func private @fused_transpose.14(%arg0: tensor<262144x640xf32>, %arg1: tensor<640xf32>) -> tensor<256x640x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<262144x640xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x640xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x640xf32>) -> tensor<256x1024x640xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<256x1024x640xf32>) -> tensor<256x640x1024xf32>
    return %3 : tensor<256x640x1024xf32>
  }
  func.func private @fused_transpose.15(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>) -> tensor<256x5x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<256x256x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x256x5x32xf32>) -> tensor<256x5x256x32xf32>
    return %3 : tensor<256x5x256x32xf32>
  }
  func.func private @fused_transpose.16(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>) -> tensor<256x5x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<256x256x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] : (tensor<256x256x5x32xf32>) -> tensor<256x5x32x256xf32>
    return %3 : tensor<256x5x32x256xf32>
  }
  func.func private @fused_transpose.17(%arg0: tensor<262144x160xf32>, %arg1: tensor<160xf32>) -> tensor<256x5x1024x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x1024x5x32xf32>) -> tensor<256x5x1024x32xf32>
    return %3 : tensor<256x5x1024x32xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<262144x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160xf32>, %arg4: tensor<256x160x32x32xf32>, %arg5: tensor<262144x160xf32>, %arg6: tensor<160xf32>) -> tensor<256x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<262144x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x32x32xf32>) -> tensor<256x160x1024xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<256x160x1024xf32>) -> tensor<256x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x32x32xf32>, tensor<f32>) -> tensor<256x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x32x32xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x32x32xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x32xf32>) -> tensor<256x1024xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<256x1024x160xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<256x1024x160xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<256x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x1024xf32>
    %20 = stablehlo.rsqrt %19 : tensor<256x1024xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<256x1024x160xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<256x1024x160xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %26 = stablehlo.add %24, %25 : tensor<256x1024x160xf32>
    %27 = stablehlo.add %5, %26 : tensor<256x1024x160xf32>
    %28 = stablehlo.add %2, %27 : tensor<256x1024x160xf32>
    return %28 : tensor<256x1024x160xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<256x640x32x32xf32>, %arg1: tensor<640xf32>) -> tensor<256x640x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x640x32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<256x640x32x32xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<256x640x32x32xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<256x640x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x640x32x32xf32>) -> tensor<256x640x1024xf32>
    %5 = stablehlo.negate %2 : tensor<256x640x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x640x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x640x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x640x32x32xf32>) -> tensor<256x640x1024xf32>
    %9 = stablehlo.abs %8 : tensor<256x640x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<256x640x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x640x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x640x1024xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<256x640x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x640x1024xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<256x640x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<256x640x1024xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<256x640x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<256x640x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x640x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<256x640x1024xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<256x640x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<256x640x1024xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<256x640x1024xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<256x640x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %37 = stablehlo.negate %13 : tensor<256x640x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
    %40 = stablehlo.exponential %37 : tensor<256x640x1024xf32>
    %41 = stablehlo.divide %10, %9 : tensor<256x640x1024xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<256x640x1024xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %45 = stablehlo.divide %10, %13 : tensor<256x640x1024xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<256x640x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %48 = stablehlo.add %46, %47 : tensor<256x640x1024xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<256x640x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %51 = stablehlo.add %49, %50 : tensor<256x640x1024xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<256x640x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x640x1024xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<256x640x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %57 = stablehlo.add %55, %56 : tensor<256x640x1024xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<256x640x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %60 = stablehlo.add %58, %59 : tensor<256x640x1024xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<256x640x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %63 = stablehlo.add %61, %62 : tensor<256x640x1024xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<256x640x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %66 = stablehlo.add %64, %65 : tensor<256x640x1024xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<256x640x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %69 = stablehlo.add %67, %68 : tensor<256x640x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<256x640x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %73 = stablehlo.add %71, %72 : tensor<256x640x1024xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<256x640x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<256x640x1024xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<256x640x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %79 = stablehlo.add %77, %78 : tensor<256x640x1024xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<256x640x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %82 = stablehlo.add %80, %81 : tensor<256x640x1024xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<256x640x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %85 = stablehlo.add %83, %84 : tensor<256x640x1024xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<256x640x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %88 = stablehlo.add %86, %87 : tensor<256x640x1024xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<256x640x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
    %91 = stablehlo.add %89, %90 : tensor<256x640x1024xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<256x640x1024xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<256x640x1024xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<256x640x1024xf32>
    return %98 : tensor<256x640x1024xf32>
  }
  func.func private @fused_transpose.18(%arg0: tensor<262144x640xf32>, %arg1: tensor<640xf32>) -> tensor<256x640x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<262144x640xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x640xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x640xf32>) -> tensor<256x1024x640xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<256x1024x640xf32>) -> tensor<256x640x1024xf32>
    return %3 : tensor<256x640x1024xf32>
  }
  func.func private @fused_transpose.19(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>) -> tensor<256x5x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<256x256x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x256x5x32xf32>) -> tensor<256x5x256x32xf32>
    return %3 : tensor<256x5x256x32xf32>
  }
  func.func private @fused_transpose.20(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>) -> tensor<256x5x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<256x256x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] : (tensor<256x256x5x32xf32>) -> tensor<256x5x32x256xf32>
    return %3 : tensor<256x5x32x256xf32>
  }
  func.func private @fused_transpose.21(%arg0: tensor<262144x160xf32>, %arg1: tensor<160xf32>) -> tensor<256x5x1024x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x1024x5x32xf32>) -> tensor<256x5x1024x32xf32>
    return %3 : tensor<256x5x1024x32xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<256x256x64x64xf32>, %arg1: tensor<256xf32>) -> tensor<256x256x4096xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256x64x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x64x64xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<256x256x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<256x256x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x64x64xf32>) -> tensor<256x256x4096xf32>
    %5 = stablehlo.negate %2 : tensor<256x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x64x64xf32>) -> tensor<256x256x4096xf32>
    %9 = stablehlo.abs %8 : tensor<256x256x4096xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<256x256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x256x4096xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x256x4096xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<256x256x4096xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x256x4096xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<256x256x4096xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<256x256x4096xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<256x256x4096xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %25 = stablehlo.add %23, %24 : tensor<256x256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x256x4096xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %28 = stablehlo.add %26, %27 : tensor<256x256x4096xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<256x256x4096xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %31 = stablehlo.add %29, %30 : tensor<256x256x4096xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<256x256x4096xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<256x256x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %37 = stablehlo.negate %13 : tensor<256x256x4096xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
    %40 = stablehlo.exponential %37 : tensor<256x256x4096xf32>
    %41 = stablehlo.divide %10, %9 : tensor<256x256x4096xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<256x256x4096xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %45 = stablehlo.divide %10, %13 : tensor<256x256x4096xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<256x256x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %48 = stablehlo.add %46, %47 : tensor<256x256x4096xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<256x256x4096xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %51 = stablehlo.add %49, %50 : tensor<256x256x4096xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<256x256x4096xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x256x4096xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<256x256x4096xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %57 = stablehlo.add %55, %56 : tensor<256x256x4096xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<256x256x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %60 = stablehlo.add %58, %59 : tensor<256x256x4096xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<256x256x4096xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %63 = stablehlo.add %61, %62 : tensor<256x256x4096xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<256x256x4096xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %66 = stablehlo.add %64, %65 : tensor<256x256x4096xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<256x256x4096xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %69 = stablehlo.add %67, %68 : tensor<256x256x4096xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<256x256x4096xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %73 = stablehlo.add %71, %72 : tensor<256x256x4096xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<256x256x4096xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %76 = stablehlo.add %74, %75 : tensor<256x256x4096xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<256x256x4096xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %79 = stablehlo.add %77, %78 : tensor<256x256x4096xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<256x256x4096xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %82 = stablehlo.add %80, %81 : tensor<256x256x4096xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<256x256x4096xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %85 = stablehlo.add %83, %84 : tensor<256x256x4096xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<256x256x4096xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %88 = stablehlo.add %86, %87 : tensor<256x256x4096xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<256x256x4096xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %91 = stablehlo.add %89, %90 : tensor<256x256x4096xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<256x256x4096xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<256x256x4096xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<256x256x4096xf32>
    return %98 : tensor<256x256x4096xf32>
  }
  func.func private @fused_transpose.22(%arg0: tensor<1048576x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x256x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1048576x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x256xf32>) -> tensor<256x4096x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<256x4096x256xf32>) -> tensor<256x256x4096xf32>
    return %3 : tensor<256x256x4096xf32>
  }
  func.func private @fused_transpose.23(%arg0: tensor<65536x128xf32>, %arg1: tensor<64xf32>) -> tensor<256x2x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
    %1 = stablehlo.slice %arg0 [0:65536, 64:128] : (tensor<65536x128xf32>) -> tensor<65536x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<65536x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<256x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<256x256x2x32xf32>) -> tensor<256x2x256x32xf32>
    return %4 : tensor<256x2x256x32xf32>
  }
  func.func private @fused_transpose.24(%arg0: tensor<65536x128xf32>, %arg1: tensor<64xf32>) -> tensor<256x2x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
    %1 = stablehlo.slice %arg0 [0:65536, 0:64] : (tensor<65536x128xf32>) -> tensor<65536x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<65536x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<256x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<256x256x2x32xf32>) -> tensor<256x2x32x256xf32>
    return %4 : tensor<256x2x32x256xf32>
  }
  func.func private @fused_transpose.25(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64xf32>) -> tensor<256x2x4096x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x4096x2x32xf32>) -> tensor<256x2x4096x32xf32>
    return %3 : tensor<256x2x4096x32xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<256x64x64x64xf32>, %arg5: tensor<1048576x64xf32>, %arg6: tensor<64xf32>) -> tensor<256x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<1048576x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x64x64xf32>) -> tensor<256x64x4096xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<256x64x4096xf32>) -> tensor<256x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x64x64xf32>, tensor<f32>) -> tensor<256x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x64x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x64x64xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x64xf32>) -> tensor<256x4096xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<256x4096x64xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<256x4096x64xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<256x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x4096xf32>
    %20 = stablehlo.rsqrt %19 : tensor<256x4096xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<256x4096x64xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<256x4096x64xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %26 = stablehlo.add %24, %25 : tensor<256x4096x64xf32>
    %27 = stablehlo.add %5, %26 : tensor<256x4096x64xf32>
    %28 = stablehlo.add %2, %27 : tensor<256x4096x64xf32>
    return %28 : tensor<256x4096x64xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<256x256x64x64xf32>, %arg1: tensor<256xf32>) -> tensor<256x256x4096xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256x64x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x64x64xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<256x256x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<256x256x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x64x64xf32>) -> tensor<256x256x4096xf32>
    %5 = stablehlo.negate %2 : tensor<256x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x64x64xf32>) -> tensor<256x256x4096xf32>
    %9 = stablehlo.abs %8 : tensor<256x256x4096xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<256x256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x256x4096xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x256x4096xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<256x256x4096xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x256x4096xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<256x256x4096xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<256x256x4096xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<256x256x4096xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %25 = stablehlo.add %23, %24 : tensor<256x256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x256x4096xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %28 = stablehlo.add %26, %27 : tensor<256x256x4096xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<256x256x4096xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %31 = stablehlo.add %29, %30 : tensor<256x256x4096xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<256x256x4096xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<256x256x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %37 = stablehlo.negate %13 : tensor<256x256x4096xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
    %40 = stablehlo.exponential %37 : tensor<256x256x4096xf32>
    %41 = stablehlo.divide %10, %9 : tensor<256x256x4096xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<256x256x4096xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %45 = stablehlo.divide %10, %13 : tensor<256x256x4096xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<256x256x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %48 = stablehlo.add %46, %47 : tensor<256x256x4096xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<256x256x4096xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %51 = stablehlo.add %49, %50 : tensor<256x256x4096xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<256x256x4096xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x256x4096xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<256x256x4096xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %57 = stablehlo.add %55, %56 : tensor<256x256x4096xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<256x256x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %60 = stablehlo.add %58, %59 : tensor<256x256x4096xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<256x256x4096xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %63 = stablehlo.add %61, %62 : tensor<256x256x4096xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<256x256x4096xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %66 = stablehlo.add %64, %65 : tensor<256x256x4096xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<256x256x4096xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %69 = stablehlo.add %67, %68 : tensor<256x256x4096xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<256x256x4096xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %73 = stablehlo.add %71, %72 : tensor<256x256x4096xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<256x256x4096xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %76 = stablehlo.add %74, %75 : tensor<256x256x4096xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<256x256x4096xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %79 = stablehlo.add %77, %78 : tensor<256x256x4096xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<256x256x4096xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %82 = stablehlo.add %80, %81 : tensor<256x256x4096xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<256x256x4096xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %85 = stablehlo.add %83, %84 : tensor<256x256x4096xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<256x256x4096xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %88 = stablehlo.add %86, %87 : tensor<256x256x4096xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<256x256x4096xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
    %91 = stablehlo.add %89, %90 : tensor<256x256x4096xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<256x256x4096xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<256x256x4096xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<256x256x4096xf32>
    return %98 : tensor<256x256x4096xf32>
  }
  func.func private @fused_transpose.26(%arg0: tensor<1048576x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x256x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1048576x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x256xf32>) -> tensor<256x4096x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<256x4096x256xf32>) -> tensor<256x256x4096xf32>
    return %3 : tensor<256x256x4096xf32>
  }
  func.func private @fused_transpose.27(%arg0: tensor<65536x128xf32>, %arg1: tensor<64xf32>) -> tensor<256x2x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
    %1 = stablehlo.slice %arg0 [0:65536, 64:128] : (tensor<65536x128xf32>) -> tensor<65536x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<65536x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<256x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<256x256x2x32xf32>) -> tensor<256x2x256x32xf32>
    return %4 : tensor<256x2x256x32xf32>
  }
  func.func private @fused_transpose.28(%arg0: tensor<65536x128xf32>, %arg1: tensor<64xf32>) -> tensor<256x2x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
    %1 = stablehlo.slice %arg0 [0:65536, 0:64] : (tensor<65536x128xf32>) -> tensor<65536x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<65536x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<256x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<256x256x2x32xf32>) -> tensor<256x2x32x256xf32>
    return %4 : tensor<256x2x32x256xf32>
  }
  func.func private @fused_transpose.29(%arg0: tensor<1048576x64xf32>, %arg1: tensor<64xf32>) -> tensor<256x2x4096x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<256x4096x2x32xf32>) -> tensor<256x2x4096x32xf32>
    return %3 : tensor<256x2x4096x32xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<256x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<256x128x16384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<256x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<256x128x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x128x128x128xf32>) -> tensor<256x128x16384xf32>
    %5 = stablehlo.negate %2 : tensor<256x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x128x128x128xf32>) -> tensor<256x128x16384xf32>
    %9 = stablehlo.abs %8 : tensor<256x128x16384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<256x128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x128x16384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x128x16384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<256x128x16384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x128x16384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<256x128x16384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<256x128x16384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<256x128x16384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<256x128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x128x16384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %28 = stablehlo.add %26, %27 : tensor<256x128x16384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<256x128x16384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %31 = stablehlo.add %29, %30 : tensor<256x128x16384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<256x128x16384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<256x128x16384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %37 = stablehlo.negate %13 : tensor<256x128x16384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
    %40 = stablehlo.exponential %37 : tensor<256x128x16384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<256x128x16384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<256x128x16384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<256x128x16384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<256x128x16384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %48 = stablehlo.add %46, %47 : tensor<256x128x16384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<256x128x16384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %51 = stablehlo.add %49, %50 : tensor<256x128x16384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<256x128x16384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x128x16384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<256x128x16384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %57 = stablehlo.add %55, %56 : tensor<256x128x16384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<256x128x16384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %60 = stablehlo.add %58, %59 : tensor<256x128x16384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<256x128x16384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %63 = stablehlo.add %61, %62 : tensor<256x128x16384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<256x128x16384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %66 = stablehlo.add %64, %65 : tensor<256x128x16384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<256x128x16384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %69 = stablehlo.add %67, %68 : tensor<256x128x16384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<256x128x16384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %73 = stablehlo.add %71, %72 : tensor<256x128x16384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<256x128x16384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %76 = stablehlo.add %74, %75 : tensor<256x128x16384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<256x128x16384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %79 = stablehlo.add %77, %78 : tensor<256x128x16384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<256x128x16384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %82 = stablehlo.add %80, %81 : tensor<256x128x16384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<256x128x16384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %85 = stablehlo.add %83, %84 : tensor<256x128x16384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<256x128x16384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %88 = stablehlo.add %86, %87 : tensor<256x128x16384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<256x128x16384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %91 = stablehlo.add %89, %90 : tensor<256x128x16384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<256x128x16384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<256x128x16384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<256x128x16384xf32>
    return %98 : tensor<256x128x16384xf32>
  }
  func.func private @fused_transpose.30(%arg0: tensor<4194304x128xf32>, %arg1: tensor<128xf32>) -> tensor<256x128x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<4194304x128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<4194304x128xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x128xf32>) -> tensor<256x16384x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<256x16384x128xf32>) -> tensor<256x128x16384xf32>
    return %3 : tensor<256x128x16384xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<4194304x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<256x32x128x128xf32>, %arg5: tensor<4194304x32xf32>, %arg6: tensor<32xf32>) -> tensor<256x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<4194304x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<4194304x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x128x128xf32>) -> tensor<256x32x16384xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<256x32x16384xf32>) -> tensor<256x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x128x128xf32>, tensor<f32>) -> tensor<256x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x128x128xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x128x128xf32>) -> tensor<256x16384xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<256x16384x32xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<256x16384x32xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<256x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x16384xf32>
    %20 = stablehlo.rsqrt %19 : tensor<256x16384xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<256x16384x32xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<256x16384x32xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %26 = stablehlo.add %24, %25 : tensor<256x16384x32xf32>
    %27 = stablehlo.add %5, %26 : tensor<256x16384x32xf32>
    %28 = stablehlo.add %2, %27 : tensor<256x16384x32xf32>
    return %28 : tensor<256x16384x32xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<256x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<256x128x16384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<256x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<256x128x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x128x128x128xf32>) -> tensor<256x128x16384xf32>
    %5 = stablehlo.negate %2 : tensor<256x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x128x128x128xf32>) -> tensor<256x128x16384xf32>
    %9 = stablehlo.abs %8 : tensor<256x128x16384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<256x128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<256x128x16384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<256x128x16384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<256x128x16384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<256x128x16384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<256x128x16384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<256x128x16384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<256x128x16384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<256x128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<256x128x16384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %28 = stablehlo.add %26, %27 : tensor<256x128x16384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<256x128x16384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %31 = stablehlo.add %29, %30 : tensor<256x128x16384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<256x128x16384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<256x128x16384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %37 = stablehlo.negate %13 : tensor<256x128x16384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
    %40 = stablehlo.exponential %37 : tensor<256x128x16384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<256x128x16384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<256x128x16384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<256x128x16384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<256x128x16384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %48 = stablehlo.add %46, %47 : tensor<256x128x16384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<256x128x16384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %51 = stablehlo.add %49, %50 : tensor<256x128x16384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<256x128x16384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %54 = stablehlo.add %52, %53 : tensor<256x128x16384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<256x128x16384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %57 = stablehlo.add %55, %56 : tensor<256x128x16384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<256x128x16384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %60 = stablehlo.add %58, %59 : tensor<256x128x16384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<256x128x16384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %63 = stablehlo.add %61, %62 : tensor<256x128x16384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<256x128x16384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %66 = stablehlo.add %64, %65 : tensor<256x128x16384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<256x128x16384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %69 = stablehlo.add %67, %68 : tensor<256x128x16384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<256x128x16384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %73 = stablehlo.add %71, %72 : tensor<256x128x16384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<256x128x16384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %76 = stablehlo.add %74, %75 : tensor<256x128x16384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<256x128x16384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %79 = stablehlo.add %77, %78 : tensor<256x128x16384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<256x128x16384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %82 = stablehlo.add %80, %81 : tensor<256x128x16384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<256x128x16384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %85 = stablehlo.add %83, %84 : tensor<256x128x16384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<256x128x16384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %88 = stablehlo.add %86, %87 : tensor<256x128x16384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<256x128x16384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
    %91 = stablehlo.add %89, %90 : tensor<256x128x16384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<256x128x16384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<256x128x16384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<256x128x16384xf32>
    return %98 : tensor<256x128x16384xf32>
  }
  func.func private @fused_transpose.31(%arg0: tensor<4194304x128xf32>, %arg1: tensor<128xf32>) -> tensor<256x128x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<4194304x128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<4194304x128xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x128xf32>) -> tensor<256x16384x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<256x16384x128xf32>) -> tensor<256x128x16384xf32>
    return %3 : tensor<256x128x16384xf32>
  }
  func.func private @fused_maximum(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256x128x128xf32>, %arg4: tensor<256xf32>) -> tensor<256x256x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<256x256x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x256x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x256x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x256x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<256x256x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x256x128x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256x128x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<256x256x128x128xf32>
    return %9 : tensor<256x256x128x128xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<128x1xf32>, %arg1: tensor<128x1xf32>, %arg2: tensor<128x1xf32>, %arg3: tensor<128x1xf32>, %arg4: tensor<128xf32>, %arg5: tensor<256x256x256xf32>, %arg6: tensor<128xf32>, %arg7: tensor<256x1024x256xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<256x16384x256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<256x4096x256xf32>, %arg13: tensor<256xf32>) -> tensor<256x1024x128x128xf32> {
    %0 = mhlo.bitcast %arg5 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256,16,16]{1,3,2,0}"} : (tensor<256x256x256xf32>) -> tensor<256x256x16x16xf32>
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
    %29 = "stablehlo.gather"(%0, %28) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %30 = mhlo.bitcast %29 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
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
    %41 = "stablehlo.gather"(%0, %40) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %43 = stablehlo.subtract %42, %30 : tensor<256x256x128x128xf32>
    %44 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<256x256x128x128xf32>
    %46 = stablehlo.add %30, %45 : tensor<256x256x128x128xf32>
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
    %58 = "stablehlo.gather"(%0, %57) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %60 = stablehlo.concatenate %55, %38, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %62 = "stablehlo.gather"(%0, %61) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %64 = stablehlo.subtract %63, %59 : tensor<256x256x128x128xf32>
    %65 = stablehlo.multiply %64, %44 : tensor<256x256x128x128xf32>
    %66 = stablehlo.add %59, %65 : tensor<256x256x128x128xf32>
    %67 = stablehlo.subtract %66, %46 : tensor<256x256x128x128xf32>
    %68 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [2] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
    %70 = stablehlo.multiply %67, %69 : tensor<256x256x128x128xf32>
    %71 = stablehlo.add %46, %70 : tensor<256x256x128x128xf32>
    %72 = mhlo.bitcast %arg7 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256,32,32]{1,3,2,0}"} : (tensor<256x1024x256xf32>) -> tensor<256x256x32x32xf32>
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
    %93 = "stablehlo.gather"(%72, %92) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
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
    %104 = "stablehlo.gather"(%72, %103) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %106 = stablehlo.subtract %105, %94 : tensor<256x256x128x128xf32>
    %107 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
    %108 = stablehlo.multiply %106, %107 : tensor<256x256x128x128xf32>
    %109 = stablehlo.add %94, %108 : tensor<256x256x128x128xf32>
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
    %120 = "stablehlo.gather"(%72, %119) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %121 = mhlo.bitcast %120 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %122 = stablehlo.concatenate %117, %101, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %124 = "stablehlo.gather"(%72, %123) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %125 = mhlo.bitcast %124 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %126 = stablehlo.subtract %125, %121 : tensor<256x256x128x128xf32>
    %127 = stablehlo.multiply %126, %107 : tensor<256x256x128x128xf32>
    %128 = stablehlo.add %121, %127 : tensor<256x256x128x128xf32>
    %129 = stablehlo.subtract %128, %109 : tensor<256x256x128x128xf32>
    %130 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [2] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
    %132 = stablehlo.multiply %129, %131 : tensor<256x256x128x128xf32>
    %133 = stablehlo.add %109, %132 : tensor<256x256x128x128xf32>
    %134 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<256xf32>) -> tensor<256x4096x256xf32>
    %135 = stablehlo.add %arg12, %134 : tensor<256x4096x256xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256,64,64]{1,3,2,0}"} : (tensor<256x4096x256xf32>) -> tensor<256x256x64x64xf32>
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
    %156 = "stablehlo.gather"(%136, %155) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
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
    %167 = "stablehlo.gather"(%136, %166) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %168 = mhlo.bitcast %167 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %169 = stablehlo.subtract %168, %157 : tensor<256x256x128x128xf32>
    %170 = stablehlo.broadcast_in_dim %arg8, dims = [3] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
    %171 = stablehlo.multiply %169, %170 : tensor<256x256x128x128xf32>
    %172 = stablehlo.add %157, %171 : tensor<256x256x128x128xf32>
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
    %183 = "stablehlo.gather"(%136, %182) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %185 = stablehlo.concatenate %180, %164, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %187 = "stablehlo.gather"(%136, %186) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %189 = stablehlo.subtract %188, %184 : tensor<256x256x128x128xf32>
    %190 = stablehlo.multiply %189, %170 : tensor<256x256x128x128xf32>
    %191 = stablehlo.add %184, %190 : tensor<256x256x128x128xf32>
    %192 = stablehlo.subtract %191, %172 : tensor<256x256x128x128xf32>
    %193 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [2] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
    %195 = stablehlo.multiply %192, %194 : tensor<256x256x128x128xf32>
    %196 = stablehlo.add %172, %195 : tensor<256x256x128x128xf32>
    %197 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<256xf32>) -> tensor<256x16384x256xf32>
    %198 = stablehlo.add %arg10, %197 : tensor<256x16384x256xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256,128,128]{1,3,2,0}"} : (tensor<256x16384x256xf32>) -> tensor<256x256x128x128xf32>
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
    %217 = "stablehlo.gather"(%199, %216) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
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
    %228 = "stablehlo.gather"(%199, %227) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %230 = stablehlo.subtract %229, %218 : tensor<256x256x128x128xf32>
    %231 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
    %232 = stablehlo.multiply %230, %231 : tensor<256x256x128x128xf32>
    %233 = stablehlo.add %218, %232 : tensor<256x256x128x128xf32>
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
    %244 = "stablehlo.gather"(%199, %243) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %246 = stablehlo.concatenate %241, %225, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %247 = mhlo.bitcast %246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %248 = "stablehlo.gather"(%199, %247) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
    %249 = mhlo.bitcast %248 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
    %250 = stablehlo.subtract %249, %245 : tensor<256x256x128x128xf32>
    %251 = stablehlo.multiply %250, %231 : tensor<256x256x128x128xf32>
    %252 = stablehlo.add %245, %251 : tensor<256x256x128x128xf32>
    %253 = stablehlo.subtract %252, %233 : tensor<256x256x128x128xf32>
    %254 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [2] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
    %256 = stablehlo.multiply %253, %255 : tensor<256x256x128x128xf32>
    %257 = stablehlo.add %233, %256 : tensor<256x256x128x128xf32>
    %258 = stablehlo.concatenate %71, %133, %196, %257, dim = 1 : (tensor<256x256x128x128xf32>, tensor<256x256x128x128xf32>, tensor<256x256x128x128xf32>, tensor<256x256x128x128xf32>) -> tensor<256x1024x128x128xf32>
    return %258 : tensor<256x1024x128x128xf32>
  }
  func.func private @region_0.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.550(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<4194304x32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<256x16384x32xf32>, %arg5: tensor<4194304x32xf32>, %arg6: tensor<32xf32>) -> (tensor<256x32x16384xf32>, tensor<256x16384x32xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<4194304x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<4194304x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<256x16384x32xf32>
    %7 = stablehlo.add %2, %6 : tensor<256x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x16384xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<256x16384x32xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<256x16384x32xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<256x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
    %17 = stablehlo.add %15, %16 : tensor<256x16384xf32>
    %18 = stablehlo.rsqrt %17 : tensor<256x16384xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<256x16384x32xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<256x16384x32xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %24 = stablehlo.add %22, %23 : tensor<256x16384x32xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<256x16384x32xf32>) -> tensor<256x32x16384xf32>
    return %25, %20 : tensor<256x32x16384xf32>, tensor<256x16384x32xf32>
  }
  func.func private @fused_transpose.32(%arg0: tensor<32xf32>, %arg1: tensor<256x16384x32xf32>, %arg2: tensor<32xf32>) -> tensor<256x32x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<256x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x16384x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x16384x32xf32>) -> tensor<256x32x16384xf32>
    return %4 : tensor<256x32x16384xf32>
  }
  func.func private @fused_transpose.33(%arg0: tensor<32xf32>, %arg1: tensor<256x16384x32xf32>, %arg2: tensor<32xf32>) -> tensor<256x32x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<256x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x16384x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x16384x32xf32>) -> tensor<256x32x16384xf32>
    return %4 : tensor<256x32x16384xf32>
  }
  func.func private @fused_add(%arg0: tensor<256x1024x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<256x1024x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256x1024x256xf32>
    return %1 : tensor<256x1024x256xf32>
  }
  func.func private @region_0.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.669(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1048576x64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<256x4096x64xf32>, %arg5: tensor<1048576x64xf32>, %arg6: tensor<64xf32>) -> (tensor<256x64x4096xf32>, tensor<256x4096x64xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1048576x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<1048576x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<256x4096x64xf32>
    %7 = stablehlo.add %2, %6 : tensor<256x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x4096xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<256x4096x64xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<256x4096x64xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<256x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
    %17 = stablehlo.add %15, %16 : tensor<256x4096xf32>
    %18 = stablehlo.rsqrt %17 : tensor<256x4096xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<256x4096x64xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<256x4096x64xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %24 = stablehlo.add %22, %23 : tensor<256x4096x64xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<256x4096x64xf32>) -> tensor<256x64x4096xf32>
    return %25, %20 : tensor<256x64x4096xf32>, tensor<256x4096x64xf32>
  }
  func.func private @fused_transpose.34(%arg0: tensor<64xf32>, %arg1: tensor<256x4096x64xf32>, %arg2: tensor<64xf32>) -> tensor<256x64x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<256x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x4096x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x4096x64xf32>) -> tensor<256x64x4096xf32>
    return %4 : tensor<256x64x4096xf32>
  }
  func.func private @fused_transpose.35(%arg0: tensor<64xf32>, %arg1: tensor<256x4096x64xf32>, %arg2: tensor<64xf32>) -> tensor<256x64x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<256x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x4096x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x4096x64xf32>) -> tensor<256x64x4096xf32>
    return %4 : tensor<256x64x4096xf32>
  }
  func.func private @fused_transpose.36(%arg0: tensor<256x1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>) -> tensor<256x160x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x1024x160xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x1024x160xf32>) -> tensor<256x160x1024xf32>
    return %4 : tensor<256x160x1024xf32>
  }
  func.func private @fused_transpose.37(%arg0: tensor<256x1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>) -> tensor<256x160x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x1024x160xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<256x1024x160xf32>) -> tensor<256x160x1024xf32>
    return %4 : tensor<256x160x1024xf32>
  }
  func.func private @region_0.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.691(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<262144x160xf32>, %arg3: tensor<160xf32>, %arg4: tensor<256x1024x160xf32>, %arg5: tensor<262144x160xf32>, %arg6: tensor<160xf32>) -> (tensor<256x160x1024xf32>, tensor<256x1024x160xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<262144x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<262144x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<256x1024x160xf32>
    %7 = stablehlo.add %2, %6 : tensor<256x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<256x1024xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<256x1024x160xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<256x1024x160xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<256x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %17 = stablehlo.add %15, %16 : tensor<256x1024xf32>
    %18 = stablehlo.rsqrt %17 : tensor<256x1024xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<256x1024x160xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<256x1024x160xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
    %24 = stablehlo.add %22, %23 : tensor<256x1024x160xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<256x1024x160xf32>) -> tensor<256x160x1024xf32>
    return %25, %20 : tensor<256x160x1024xf32>, tensor<256x1024x160xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<256x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256x256x256xf32>
    return %1 : tensor<256x256x256xf32>
  }
  func.func private @region_0.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<256x16x16xf32>, %arg1: tensor<256x160x16x16xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<256x160x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<256x160x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    return %5 : tensor<256x256xf32>
  }
  func.func private @region_0.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<256x16x16xf32>, %arg1: tensor<256x160x16x16xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<256x160x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<256x160x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    return %5 : tensor<256x256xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<256x256xf32>, %arg1: tensor<256x16x16xf32>, %arg2: tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<256x160x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x160x256xf32>
    return %5 : tensor<256x160x256xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<256x256xf32>, %arg1: tensor<256x16x16xf32>, %arg2: tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<256x160x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x160x256xf32>
    return %5 : tensor<256x160x256xf32>
  }
  func.func private @region_0.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<256x16x16xf32>, %arg1: tensor<256x64x16x16xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<256x64x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<256x64x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    return %5 : tensor<256x256xf32>
  }
  func.func private @region_0.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<256x16x16xf32>, %arg1: tensor<256x64x16x16xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<256x64x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<256x64x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    return %5 : tensor<256x256xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<256x256xf32>, %arg1: tensor<256x16x16xf32>, %arg2: tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<256x64x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x64x256xf32>
    return %5 : tensor<256x64x256xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<256x256xf32>, %arg1: tensor<256x16x16xf32>, %arg2: tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<256x64x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x64x256xf32>
    return %5 : tensor<256x64x256xf32>
  }
  func.func private @region_0.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<256x32x16x16xf32>, %arg1: tensor<256x16x16xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x16x16xf32>) -> tensor<256x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<256x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<256x32x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<256x32x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x32x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    return %7 : tensor<256x256xf32>
  }
  func.func private @region_0.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<256x32x16x16xf32>, %arg1: tensor<256x16x16xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x16x16xf32>) -> tensor<256x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<256x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<256x32x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<256x32x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x32x256xf32>, tensor<f32>) -> tensor<256x256xf32>
    return %7 : tensor<256x256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<256x32x16x16xf32>, %arg1: tensor<256x16x16xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x32x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x16x16xf32>) -> tensor<256x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<256x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<256x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %7 = stablehlo.multiply %arg2, %6 : tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<256x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<256x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<256x32x256xf32>
    return %12 : tensor<256x32x256xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<256x32x16x16xf32>, %arg1: tensor<256x16x16xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x32x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x16x16xf32>) -> tensor<256x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<256x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<256x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %7 = stablehlo.multiply %arg2, %6 : tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<256x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<256x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<256x32x256xf32>
    return %12 : tensor<256x32x256xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<256x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x16x16xf32>
    return %1 : tensor<256x16x16xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<256x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x16x16xf32>
    return %1 : tensor<256x16x16xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<256x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x16x16xf32>
    return %1 : tensor<256x16x16xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<256x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x16x16xf32>
    return %1 : tensor<256x16x16xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256x256xf32>
    return %4 : tensor<256x256xf32>
  }
  func.func private @fused_rsqrt.1(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256x256xf32>
    return %4 : tensor<256x256xf32>
  }
  func.func private @fused_rsqrt.2(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256x256xf32>
    return %4 : tensor<256x256xf32>
  }
  func.func private @fused_rsqrt.3(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<256x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256x256xf32>
    return %4 : tensor<256x256xf32>
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
  func.func private @fused_rsqrt.4(%arg0: tensor<256xf32>) -> tensor<256xf32> {
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
  func.func private @fused_reduce.6(%arg0: tensor<256x32x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    return %0 : tensor<256x16x16xf32>
  }
  func.func private @region_0.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<256x160x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    return %0 : tensor<256x16x16xf32>
  }
  func.func private @region_0.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<256x32x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    return %0 : tensor<256x16x16xf32>
  }
  func.func private @region_0.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<256x64x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    return %0 : tensor<256x16x16xf32>
  }
  func.func private @region_0.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<256x64x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    return %0 : tensor<256x16x16xf32>
  }
  func.func private @region_0.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<256x160x16x16xf32>) -> tensor<256x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
    return %0 : tensor<256x16x16xf32>
  }
  func.func private @wrapped_concatenate_computation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
  func.func private @wrapped_slice_computation(%arg0: tensor<65536x64xf32>) -> tensor<65536x32xf32> {
    %0 = stablehlo.slice %arg0 [0:65536, 0:32] : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
    return %0 : tensor<65536x32xf32>
  }
  func.func private @wrapped_slice_computation.1(%arg0: tensor<65536x64xf32>) -> tensor<65536x32xf32> {
    %0 = stablehlo.slice %arg0 [0:65536, 32:64] : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
    return %0 : tensor<65536x32xf32>
  }
  func.func private @wrapped_concatenate_computation.1(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
  func.func private @wrapped_slice_computation.2(%arg0: tensor<65536x64xf32>) -> tensor<65536x32xf32> {
    %0 = stablehlo.slice %arg0 [0:65536, 0:32] : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
    return %0 : tensor<65536x32xf32>
  }
  func.func private @wrapped_slice_computation.3(%arg0: tensor<65536x64xf32>) -> tensor<65536x32xf32> {
    %0 = stablehlo.slice %arg0 [0:65536, 32:64] : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
    return %0 : tensor<65536x32xf32>
  }
  func.func @main(%arg0: tensor<32x3x7x7xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<64x32x3x3xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<160x64x3x3xf32>, %arg9: tensor<160xf32>, %arg10: tensor<160xf32>, %arg11: tensor<160xf32>, %arg12: tensor<256x160x3x3xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32x32xf32>, %arg21: tensor<32xf32>, %arg22: tensor<32x32xf32>, %arg23: tensor<32xf32>, %arg24: tensor<32x32x8x8xf32>, %arg25: tensor<32xf32>, %arg26: tensor<32xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32x32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<32xf32>, %arg31: tensor<32xf32>, %arg32: tensor<128x32xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128x1x3x3xf32>, %arg35: tensor<128xf32>, %arg36: tensor<32x128xf32>, %arg37: tensor<32xf32>, %arg38: tensor<32xf32>, %arg39: tensor<32xf32>, %arg40: tensor<32x32xf32>, %arg41: tensor<32xf32>, %arg42: tensor<32x32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<32x32xf32>, %arg45: tensor<32xf32>, %arg46: tensor<32x32x8x8xf32>, %arg47: tensor<32xf32>, %arg48: tensor<32xf32>, %arg49: tensor<32xf32>, %arg50: tensor<32x32xf32>, %arg51: tensor<32xf32>, %arg52: tensor<32xf32>, %arg53: tensor<32xf32>, %arg54: tensor<128x32xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128x1x3x3xf32>, %arg57: tensor<128xf32>, %arg58: tensor<32x128xf32>, %arg59: tensor<32xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64xf32>, %arg62: tensor<64x64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64x64xf32>, %arg65: tensor<64xf32>, %arg66: tensor<64x64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<64x64x4x4xf32>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64x64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<64xf32>, %arg75: tensor<64xf32>, %arg76: tensor<256x64xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x1x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<64x256xf32>, %arg81: tensor<64xf32>, %arg82: tensor<64xf32>, %arg83: tensor<64xf32>, %arg84: tensor<64x64xf32>, %arg85: tensor<64xf32>, %arg86: tensor<64x64xf32>, %arg87: tensor<64xf32>, %arg88: tensor<64x64xf32>, %arg89: tensor<64xf32>, %arg90: tensor<64x64x4x4xf32>, %arg91: tensor<64xf32>, %arg92: tensor<64xf32>, %arg93: tensor<64xf32>, %arg94: tensor<64x64xf32>, %arg95: tensor<64xf32>, %arg96: tensor<64xf32>, %arg97: tensor<64xf32>, %arg98: tensor<256x64xf32>, %arg99: tensor<256xf32>, %arg100: tensor<256x1x3x3xf32>, %arg101: tensor<256xf32>, %arg102: tensor<64x256xf32>, %arg103: tensor<64xf32>, %arg104: tensor<160xf32>, %arg105: tensor<160xf32>, %arg106: tensor<160x160xf32>, %arg107: tensor<160xf32>, %arg108: tensor<160x160xf32>, %arg109: tensor<160xf32>, %arg110: tensor<160x160xf32>, %arg111: tensor<160xf32>, %arg112: tensor<160x160x2x2xf32>, %arg113: tensor<160xf32>, %arg114: tensor<160xf32>, %arg115: tensor<160xf32>, %arg116: tensor<160x160xf32>, %arg117: tensor<160xf32>, %arg118: tensor<160xf32>, %arg119: tensor<160xf32>, %arg120: tensor<640x160xf32>, %arg121: tensor<640xf32>, %arg122: tensor<640x1x3x3xf32>, %arg123: tensor<640xf32>, %arg124: tensor<160x640xf32>, %arg125: tensor<160xf32>, %arg126: tensor<160xf32>, %arg127: tensor<160xf32>, %arg128: tensor<160x160xf32>, %arg129: tensor<160xf32>, %arg130: tensor<160x160xf32>, %arg131: tensor<160xf32>, %arg132: tensor<160x160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<160x160x2x2xf32>, %arg135: tensor<160xf32>, %arg136: tensor<160xf32>, %arg137: tensor<160xf32>, %arg138: tensor<160x160xf32>, %arg139: tensor<160xf32>, %arg140: tensor<160xf32>, %arg141: tensor<160xf32>, %arg142: tensor<640x160xf32>, %arg143: tensor<640xf32>, %arg144: tensor<640x1x3x3xf32>, %arg145: tensor<640xf32>, %arg146: tensor<160x640xf32>, %arg147: tensor<160xf32>, %arg148: tensor<256xf32>, %arg149: tensor<256xf32>, %arg150: tensor<256x256xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256x256xf32>, %arg153: tensor<256xf32>, %arg154: tensor<256x256xf32>, %arg155: tensor<256xf32>, %arg156: tensor<256x256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<1024x256xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<1024x1x3x3xf32>, %arg163: tensor<1024xf32>, %arg164: tensor<256x1024xf32>, %arg165: tensor<256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<256x256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256x256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x256xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256x256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<1024x256xf32>, %arg179: tensor<1024xf32>, %arg180: tensor<1024x1x3x3xf32>, %arg181: tensor<1024xf32>, %arg182: tensor<256x1024xf32>, %arg183: tensor<256xf32>, %arg184: tensor<32xf32>, %arg185: tensor<32xf32>, %arg186: tensor<64xf32>, %arg187: tensor<64xf32>, %arg188: tensor<160xf32>, %arg189: tensor<160xf32>, %arg190: tensor<256xf32>, %arg191: tensor<256xf32>, %arg192: tensor<256x32xf32>, %arg193: tensor<256xf32>, %arg194: tensor<256x64xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256x160xf32>, %arg197: tensor<256xf32>, %arg198: tensor<256x256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256x1024x1x1xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<150x256x1x1xf32>, %arg204: tensor<150xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<256x3x512x512xf32>) -> tensor<256x150x128x128xf32> {
    %0 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %304 = stablehlo.add %arg208, %303 : tensor<256xf32>
      %305 = stablehlo.rsqrt %304 : tensor<256xf32>
      mhlo.return %305 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.iota dim = 0 : tensor<128xi32>
      %306 = stablehlo.convert %305 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %308 = stablehlo.add %306, %307 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %312 = stablehlo.add %310, %311 : tensor<128xf32>
      %313 = stablehlo.maximum %304, %312 : tensor<128xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %315 = stablehlo.convert %314 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %316 = stablehlo.convert %315 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %317 = stablehlo.subtract %314, %316 : tensor<128x1xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %319 = stablehlo.clamp %303, %317, %318 : tensor<128x1xf32>
      mhlo.return %319 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.iota dim = 0 : tensor<128xi32>
      %306 = stablehlo.convert %305 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %308 = stablehlo.add %306, %307 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %312 = stablehlo.add %310, %311 : tensor<128xf32>
      %313 = stablehlo.maximum %304, %312 : tensor<128xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %315 = stablehlo.convert %314 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %316 = stablehlo.convert %315 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %317 = stablehlo.subtract %314, %316 : tensor<128x1xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %319 = stablehlo.clamp %303, %317, %318 : tensor<128x1xf32>
      mhlo.return %319 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %3 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.iota dim = 0 : tensor<128xi32>
      %306 = stablehlo.convert %305 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %308 = stablehlo.add %306, %307 : tensor<128xf32>
      %309 = stablehlo.multiply %308, %307 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %311 = stablehlo.add %309, %310 : tensor<128xf32>
      %312 = stablehlo.maximum %304, %311 : tensor<128xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %314 = stablehlo.convert %313 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %315 = stablehlo.convert %314 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %316 = stablehlo.subtract %313, %315 : tensor<128x1xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %318 = stablehlo.clamp %303, %316, %317 : tensor<128x1xf32>
      mhlo.return %318 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %4 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.iota dim = 0 : tensor<128xi32>
      %306 = stablehlo.convert %305 : (tensor<128xi32>) -> tensor<128xf32>
      %307 = stablehlo.maximum %304, %306 : tensor<128xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %309 = stablehlo.convert %308 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %310 = stablehlo.convert %309 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %311 = stablehlo.subtract %308, %310 : tensor<128x1xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %313 = stablehlo.clamp %303, %311, %312 : tensor<128x1xf32>
      mhlo.return %313 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %5 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %304 = stablehlo.iota dim = 0 : tensor<128xi32>
      %305 = stablehlo.convert %304 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %307 = stablehlo.add %305, %306 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %311 = stablehlo.add %309, %310 : tensor<128xf32>
      %312 = stablehlo.maximum %303, %311 : tensor<128xf32>
      %313 = stablehlo.convert %312 : (tensor<128xf32>) -> tensor<128xi32>
      %314 = stablehlo.convert %313 : (tensor<128xi32>) -> tensor<128xf32>
      %315 = stablehlo.subtract %312, %314 : tensor<128xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %317 = stablehlo.clamp %303, %315, %316 : tensor<128xf32>
      mhlo.return %317 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %6 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%arg207, %arg0, %arg1) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<256x3x512x512xf32>, tensor<32x3x7x7xf32>, tensor<32xf32>) -> tuple<tensor<256x32x128x128xf32>, tensor<0xui8>>
    %7 = stablehlo.get_tuple_element %6[0] : (tuple<tensor<256x32x128x128xf32>, tensor<0xui8>>) -> tensor<256x32x128x128xf32>
    %8 = "mhlo.fusion"(%arg3, %arg2, %7) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<256x32x128x128xf32>):
      %303 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x128x128xf32>) -> tensor<256x32x16384xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<256x32x16384xf32>) -> tensor<256x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x128x128xf32>, tensor<f32>) -> tensor<256x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<256x128x128xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x128x128xf32>) -> tensor<256x16384xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %310 = stablehlo.subtract %304, %309 : tensor<256x16384x32xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<256x16384x32xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %314 = stablehlo.multiply %312, %313 : tensor<256x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %316 = stablehlo.add %314, %315 : tensor<256x16384xf32>
      %317 = stablehlo.rsqrt %316 : tensor<256x16384xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %319 = stablehlo.multiply %310, %318 : tensor<256x16384x32xf32>
      %320 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<256x16384x32xf32>
      %322 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %323 = stablehlo.add %321, %322 : tensor<256x16384x32xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %325 = stablehlo.multiply %324, %313 : tensor<256x16384xf32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %327 = stablehlo.subtract %323, %326 : tensor<256x16384x32xf32>
      %328 = stablehlo.multiply %327, %327 : tensor<256x16384x32xf32>
      %329 = stablehlo.reduce(%328 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %330 = stablehlo.multiply %329, %313 : tensor<256x16384xf32>
      %331 = stablehlo.add %330, %315 : tensor<256x16384xf32>
      %332 = stablehlo.rsqrt %331 : tensor<256x16384xf32>
      %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %334 = stablehlo.multiply %327, %333 : tensor<256x16384x32xf32>
      mhlo.return %334 : tensor<256x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<256x32x128x128xf32>) -> tensor<256x16384x32xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x32xf32>) -> tensor<4194304x32xf32>
    %10 = "mhlo.fusion"(%9, %arg16, %arg17, %arg18) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4194304x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<4194304x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %306 = stablehlo.add %304, %305 : tensor<4194304x32xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
      mhlo.return %307 : tensor<4194304x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
    %12 = "mhlo.fusion"(%arg17, %8, %arg16) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<256x16384x32xf32>, %arg210: tensor<32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %304 = stablehlo.multiply %arg209, %303 : tensor<256x16384x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x16384x32xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x16384x32xf32>) -> tensor<256x32x16384xf32>
      mhlo.return %307 : tensor<256x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<256x16384x32xf32>, tensor<32xf32>) -> tensor<256x32x16384xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x16384xf32>) -> tensor<256x32x128x128xf32>
    %14 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%13, %arg24, %arg25) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,32,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x32x128x128xf32>, tensor<32x32x8x8xf32>, tensor<32xf32>) -> tuple<tensor<256x32x16x16xf32>, tensor<0xui8>>
    %15 = stablehlo.get_tuple_element %14[0] : (tuple<tensor<256x32x16x16xf32>, tensor<0xui8>>) -> tensor<256x32x16x16xf32>
    %16 = "mhlo.fusion"(%15) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x32x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      mhlo.return %303 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32x16x16xf32>) -> tensor<256x16x16xf32>
    %17 = "mhlo.fusion"(%15, %16) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x32x16x16xf32>, %arg209: tensor<256x16x16xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x16x16xf32>) -> tensor<256x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %305 = stablehlo.multiply %arg209, %304 : tensor<256x16x16xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
      %308 = stablehlo.subtract %303, %307 : tensor<256x32x256xf32>
      %309 = stablehlo.multiply %308, %308 : tensor<256x32x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %310 = stablehlo.reduce(%309 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x32x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      mhlo.return %310 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32x16x16xf32>, tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %18 = "mhlo.fusion"(%15, %16, %17) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32x16x16xf32>, %arg209: tensor<256x16x16xf32>, %arg210: tensor<256x256xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x16x16xf32>) -> tensor<256x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %305 = stablehlo.multiply %arg209, %304 : tensor<256x16x16xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
      %308 = stablehlo.subtract %303, %307 : tensor<256x32x256xf32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %310 = stablehlo.multiply %arg210, %309 : tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %312 = stablehlo.add %310, %311 : tensor<256x256xf32>
      %313 = stablehlo.rsqrt %312 : tensor<256x256xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
      %315 = stablehlo.multiply %308, %314 : tensor<256x32x256xf32>
      mhlo.return %315 : tensor<256x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32x16x16xf32>, tensor<256x16x16xf32>, tensor<256x256xf32>) -> tensor<256x32x256xf32>
    %19 = "mhlo.fusion"(%arg20, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32xf32>, %arg209: tensor<32x32xf32>):
      %303 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
      mhlo.return %303 : tensor<64x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,64]{0,1}"} : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %21 = "mhlo.fusion"(%18, %arg26, %arg27, %20) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x32x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<256x32x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x32x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<256x32x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x32x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x32x256xf32>) -> tensor<256x256x32xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x32xf32>) -> tensor<65536x32xf32>
      %309 = stablehlo.dot %308, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<65536x32xf32>, tensor<32x64xf32>) -> tensor<65536x64xf32>
      mhlo.return %309 : tensor<65536x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32x256xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x64xf32>) -> tensor<65536x64xf32>
    %22 = "mhlo.fusion"(%21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x64xf32>):
      %303 = stablehlo.slice %arg208 [0:65536, 0:32] : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
      mhlo.return %303 : tensor<65536x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x32xf32>) -> tensor<256x256x32xf32>
    %24 = "mhlo.fusion"(%arg19, %11, %arg21, %23) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<256x16384x32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<256x256x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %304 = stablehlo.add %303, %arg209 : tensor<256x16384x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<256x256x32xf32>
      %306 = stablehlo.add %305, %arg211 : tensor<256x256x32xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x256x32xf32>) -> tensor<256x32x256xf32>
      %308 = stablehlo.dot_general %304, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x32xf32>, tensor<256x32x256xf32>) -> tensor<256x16384x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16384x256xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x16384x256xf32>
      mhlo.return %310 : tensor<256x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<256x16384x32xf32>, tensor<32xf32>, tensor<256x256x32xf32>) -> tensor<256x16384x256xf32>
    %25 = "mhlo.fusion"(%24) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x16384x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<256x16384x256xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x256xf32>
      %305 = stablehlo.subtract %arg208, %304 : tensor<256x16384x256xf32>
      %306 = stablehlo.exponential %305 : tensor<256x16384x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x256xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x256xf32>
      %309 = stablehlo.divide %306, %308 : tensor<256x16384x256xf32>
      mhlo.return %309 : tensor<256x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16384x256xf32>) -> tensor<256x16384x256xf32>
    %26 = "mhlo.fusion"(%21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x64xf32>):
      %303 = stablehlo.slice %arg208 [0:65536, 32:64] : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
      mhlo.return %303 : tensor<65536x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
    %27 = mhlo.bitcast %26 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x32xf32>) -> tensor<256x256x32xf32>
    %28 = "mhlo.fusion"(%25, %arg23, %27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x16384x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<256x256x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<256x256x32xf32>
      %304 = stablehlo.add %303, %arg210 : tensor<256x256x32xf32>
      %305 = stablehlo.dot_general %arg208, %304, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x256xf32>, tensor<256x256x32xf32>) -> tensor<256x16384x32xf32>
      mhlo.return %305 : tensor<256x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16384x256xf32>, tensor<32xf32>, tensor<256x256x32xf32>) -> tensor<256x16384x32xf32>
    %29 = mhlo.bitcast %28 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x32xf32>) -> tensor<4194304x32xf32>
    %30 = "mhlo.fusion"(%29, %arg28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4194304x32xf32>, %arg209: tensor<32x32xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
      mhlo.return %303 : tensor<4194304x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
    %31 = "mhlo.fusion"(%arg3, %arg2, %7, %30, %arg29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<256x32x128x128xf32>, %arg211: tensor<4194304x32xf32>, %arg212: tensor<32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %304 = stablehlo.add %303, %arg211 : tensor<4194304x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
      %306 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x128x128xf32>) -> tensor<256x32x16384xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x32x16384xf32>) -> tensor<256x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x128x128xf32>, tensor<f32>) -> tensor<256x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x128x128xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x128x128xf32>) -> tensor<256x16384xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %313 = stablehlo.subtract %307, %312 : tensor<256x16384x32xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<256x16384x32xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %316 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x16384xf32>
      %320 = stablehlo.rsqrt %319 : tensor<256x16384xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %322 = stablehlo.multiply %313, %321 : tensor<256x16384x32xf32>
      %323 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<256x16384x32xf32>
      %325 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %326 = stablehlo.add %324, %325 : tensor<256x16384x32xf32>
      %327 = stablehlo.add %305, %326 : tensor<256x16384x32xf32>
      %328 = stablehlo.reduce(%327 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x16384xf32>
      %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %331 = stablehlo.subtract %327, %330 : tensor<256x16384x32xf32>
      %332 = stablehlo.multiply %331, %331 : tensor<256x16384x32xf32>
      %333 = stablehlo.reduce(%332 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %334 = stablehlo.multiply %333, %316 : tensor<256x16384xf32>
      %335 = stablehlo.add %334, %318 : tensor<256x16384xf32>
      %336 = stablehlo.rsqrt %335 : tensor<256x16384xf32>
      %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %338 = stablehlo.multiply %331, %337 : tensor<256x16384x32xf32>
      mhlo.return %338 : tensor<256x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<256x32x128x128xf32>, tensor<4194304x32xf32>, tensor<32xf32>) -> tensor<256x16384x32xf32>
    %32 = mhlo.bitcast %31 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x32xf32>) -> tensor<4194304x32xf32>
    %33 = "mhlo.fusion"(%32, %arg30, %arg31, %arg32) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4194304x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<4194304x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %306 = stablehlo.add %304, %305 : tensor<4194304x32xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<128x32xf32>) -> tensor<4194304x128xf32>
      mhlo.return %307 : tensor<4194304x128xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32xf32>) -> tensor<4194304x128xf32>
    %34 = "mhlo.fusion"(%33, %arg33) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<4194304x128xf32>, %arg209: tensor<128xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<4194304x128xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<4194304x128xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x128xf32>) -> tensor<256x16384x128xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1] : (tensor<256x16384x128xf32>) -> tensor<256x128x16384xf32>
      mhlo.return %306 : tensor<256x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x128xf32>, tensor<128xf32>) -> tensor<256x128x16384xf32>
    %35 = mhlo.bitcast %34 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x128x16384xf32>) -> tensor<256x128x128x128xf32>
    %36 = stablehlo.custom_call @__cudnn$convForward(%35, %arg34) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<256x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<256x128x128x128xf32>, tensor<0xui8>>
    %37 = stablehlo.get_tuple_element %36[0] : (tuple<tensor<256x128x128x128xf32>, tensor<0xui8>>) -> tensor<256x128x128x128xf32>
    %38 = "mhlo.fusion"(%37, %arg35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x128x128x128xf32>
      %304 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<256x128x128x128xf32>
      %305 = stablehlo.add %arg208, %304 : tensor<256x128x128x128xf32>
      %306 = stablehlo.multiply %303, %305 : tensor<256x128x128x128xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x128x128x128xf32>) -> tensor<256x128x16384xf32>
      %308 = stablehlo.negate %305 : tensor<256x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128x128xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x128x128x128xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x128x128x128xf32>) -> tensor<256x128x16384xf32>
      %312 = stablehlo.abs %311 : tensor<256x128x16384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %314 = stablehlo.compare LT, %312, %313 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %316 = stablehlo.multiply %311, %311 : tensor<256x128x16384xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x128x16384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x128x16384xf32>
      %320 = stablehlo.multiply %319, %316 : tensor<256x128x16384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x128x16384xf32>
      %323 = stablehlo.multiply %322, %316 : tensor<256x128x16384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %325 = stablehlo.add %323, %324 : tensor<256x128x16384xf32>
      %326 = stablehlo.multiply %325, %316 : tensor<256x128x16384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %328 = stablehlo.add %326, %327 : tensor<256x128x16384xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x128x16384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %331 = stablehlo.add %329, %330 : tensor<256x128x16384xf32>
      %332 = stablehlo.multiply %331, %316 : tensor<256x128x16384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %334 = stablehlo.add %332, %333 : tensor<256x128x16384xf32>
      %335 = stablehlo.multiply %311, %334 : tensor<256x128x16384xf32>
      %336 = stablehlo.subtract %313, %335 : tensor<256x128x16384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %338 = stablehlo.compare LT, %311, %337 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %340 = stablehlo.negate %316 : tensor<256x128x16384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %342 = stablehlo.compare LT, %340, %341 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
      %343 = stablehlo.exponential %340 : tensor<256x128x16384xf32>
      %344 = stablehlo.divide %313, %312 : tensor<256x128x16384xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<256x128x16384xf32>
      %346 = stablehlo.compare LT, %312, %339 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %348 = stablehlo.divide %313, %316 : tensor<256x128x16384xf32>
      %349 = stablehlo.multiply %347, %348 : tensor<256x128x16384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %351 = stablehlo.add %349, %350 : tensor<256x128x16384xf32>
      %352 = stablehlo.multiply %351, %348 : tensor<256x128x16384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %354 = stablehlo.add %352, %353 : tensor<256x128x16384xf32>
      %355 = stablehlo.multiply %354, %348 : tensor<256x128x16384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %357 = stablehlo.add %355, %356 : tensor<256x128x16384xf32>
      %358 = stablehlo.multiply %357, %348 : tensor<256x128x16384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %360 = stablehlo.add %358, %359 : tensor<256x128x16384xf32>
      %361 = stablehlo.multiply %360, %348 : tensor<256x128x16384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %363 = stablehlo.add %361, %362 : tensor<256x128x16384xf32>
      %364 = stablehlo.multiply %363, %348 : tensor<256x128x16384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %366 = stablehlo.add %364, %365 : tensor<256x128x16384xf32>
      %367 = stablehlo.multiply %366, %348 : tensor<256x128x16384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %369 = stablehlo.add %367, %368 : tensor<256x128x16384xf32>
      %370 = stablehlo.multiply %369, %348 : tensor<256x128x16384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %372 = stablehlo.add %370, %371 : tensor<256x128x16384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %374 = stablehlo.multiply %373, %348 : tensor<256x128x16384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %376 = stablehlo.add %374, %375 : tensor<256x128x16384xf32>
      %377 = stablehlo.multiply %376, %348 : tensor<256x128x16384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x128x16384xf32>
      %380 = stablehlo.multiply %379, %348 : tensor<256x128x16384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %382 = stablehlo.add %380, %381 : tensor<256x128x16384xf32>
      %383 = stablehlo.multiply %382, %348 : tensor<256x128x16384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %385 = stablehlo.add %383, %384 : tensor<256x128x16384xf32>
      %386 = stablehlo.multiply %385, %348 : tensor<256x128x16384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %388 = stablehlo.add %386, %387 : tensor<256x128x16384xf32>
      %389 = stablehlo.multiply %388, %348 : tensor<256x128x16384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %391 = stablehlo.add %389, %390 : tensor<256x128x16384xf32>
      %392 = stablehlo.multiply %391, %348 : tensor<256x128x16384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %394 = stablehlo.add %392, %393 : tensor<256x128x16384xf32>
      %395 = stablehlo.select %346, %372, %394 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
      %396 = stablehlo.multiply %345, %395 : tensor<256x128x16384xf32>
      %397 = stablehlo.select %342, %337, %396 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
      %398 = stablehlo.subtract %339, %397 : tensor<256x128x16384xf32>
      %399 = stablehlo.select %338, %398, %397 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
      %400 = stablehlo.select %314, %336, %399 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
      %401 = stablehlo.multiply %307, %400 : tensor<256x128x16384xf32>
      mhlo.return %401 : tensor<256x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x128x128xf32>, tensor<128xf32>) -> tensor<256x128x16384xf32>
    %39 = "mhlo.fusion"(%38, %arg36) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x128x16384xf32>, %arg209: tensor<32x128xf32>):
      %303 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<256x128x16384xf32>) -> tensor<256x16384x128xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x128xf32>) -> tensor<4194304x128xf32>
      %305 = stablehlo.dot_general %304, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x128xf32>, tensor<32x128xf32>) -> tensor<4194304x32xf32>
      mhlo.return %305 : tensor<4194304x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x16384xf32>, tensor<32x128xf32>) -> tensor<4194304x32xf32>
    %40 = "mhlo.fusion"(%39, %arg37, %arg3, %arg2, %7, %30, %arg29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4194304x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32xf32>, %arg212: tensor<256x32x128x128xf32>, %arg213: tensor<4194304x32xf32>, %arg214: tensor<32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<4194304x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
      %306 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %307 = stablehlo.add %306, %arg213 : tensor<4194304x32xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
      %309 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x128x128xf32>) -> tensor<256x32x16384xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1] : (tensor<256x32x16384xf32>) -> tensor<256x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x128x128xf32>, tensor<f32>) -> tensor<256x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<256x128x128xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x128x128xf32>) -> tensor<256x16384xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %316 = stablehlo.subtract %310, %315 : tensor<256x16384x32xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<256x16384x32xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<256x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x16384xf32>
      %323 = stablehlo.rsqrt %322 : tensor<256x16384xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %325 = stablehlo.multiply %316, %324 : tensor<256x16384x32xf32>
      %326 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %327 = stablehlo.multiply %325, %326 : tensor<256x16384x32xf32>
      %328 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %329 = stablehlo.add %327, %328 : tensor<256x16384x32xf32>
      %330 = stablehlo.add %308, %329 : tensor<256x16384x32xf32>
      %331 = stablehlo.add %305, %330 : tensor<256x16384x32xf32>
      mhlo.return %331 : tensor<256x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<256x32x128x128xf32>, tensor<4194304x32xf32>, tensor<32xf32>) -> tensor<256x16384x32xf32>
    %41 = "mhlo.fusion"(%40) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x16384x32xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<256x16384xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %307 = stablehlo.subtract %arg208, %306 : tensor<256x16384x32xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<256x16384x32xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %310 = stablehlo.multiply %309, %304 : tensor<256x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %312 = stablehlo.add %310, %311 : tensor<256x16384xf32>
      %313 = stablehlo.rsqrt %312 : tensor<256x16384xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %315 = stablehlo.multiply %307, %314 : tensor<256x16384x32xf32>
      mhlo.return %315 : tensor<256x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16384x32xf32>) -> tensor<256x16384x32xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x32xf32>) -> tensor<4194304x32xf32>
    %43 = "mhlo.fusion"(%42, %arg38, %arg39, %arg40) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4194304x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<4194304x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %306 = stablehlo.add %304, %305 : tensor<4194304x32xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
      mhlo.return %307 : tensor<4194304x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
    %44 = mhlo.bitcast %43 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
    %45 = "mhlo.fusion"(%arg39, %41, %arg38) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<256x16384x32xf32>, %arg210: tensor<32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %304 = stablehlo.multiply %arg209, %303 : tensor<256x16384x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x16384x32xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x16384x32xf32>) -> tensor<256x32x16384xf32>
      mhlo.return %307 : tensor<256x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<256x16384x32xf32>, tensor<32xf32>) -> tensor<256x32x16384xf32>
    %46 = mhlo.bitcast %45 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x16384xf32>) -> tensor<256x32x128x128xf32>
    %47 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%46, %arg46, %arg47) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,32,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x32x128x128xf32>, tensor<32x32x8x8xf32>, tensor<32xf32>) -> tuple<tensor<256x32x16x16xf32>, tensor<0xui8>>
    %48 = stablehlo.get_tuple_element %47[0] : (tuple<tensor<256x32x16x16xf32>, tensor<0xui8>>) -> tensor<256x32x16x16xf32>
    %49 = "mhlo.fusion"(%48) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x32x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x32x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      mhlo.return %303 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32x16x16xf32>) -> tensor<256x16x16xf32>
    %50 = "mhlo.fusion"(%48, %49) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x32x16x16xf32>, %arg209: tensor<256x16x16xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x16x16xf32>) -> tensor<256x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %305 = stablehlo.multiply %arg209, %304 : tensor<256x16x16xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
      %308 = stablehlo.subtract %303, %307 : tensor<256x32x256xf32>
      %309 = stablehlo.multiply %308, %308 : tensor<256x32x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %310 = stablehlo.reduce(%309 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x32x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      mhlo.return %310 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32x16x16xf32>, tensor<256x16x16xf32>) -> tensor<256x256xf32>
    %51 = "mhlo.fusion"(%48, %49, %50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32x16x16xf32>, %arg209: tensor<256x16x16xf32>, %arg210: tensor<256x256xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x32x16x16xf32>) -> tensor<256x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %305 = stablehlo.multiply %arg209, %304 : tensor<256x16x16xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
      %308 = stablehlo.subtract %303, %307 : tensor<256x32x256xf32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %310 = stablehlo.multiply %arg210, %309 : tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %312 = stablehlo.add %310, %311 : tensor<256x256xf32>
      %313 = stablehlo.rsqrt %312 : tensor<256x256xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x32x256xf32>
      %315 = stablehlo.multiply %308, %314 : tensor<256x32x256xf32>
      mhlo.return %315 : tensor<256x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32x16x16xf32>, tensor<256x16x16xf32>, tensor<256x256xf32>) -> tensor<256x32x256xf32>
    %52 = "mhlo.fusion"(%arg42, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32xf32>, %arg209: tensor<32x32xf32>):
      %303 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
      mhlo.return %303 : tensor<64x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %53 = mhlo.bitcast %52 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,64]{0,1}"} : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %54 = "mhlo.fusion"(%51, %arg48, %arg49, %53) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x32x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<256x32x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x32x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<256x32x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x32x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x32x256xf32>) -> tensor<256x256x32xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x32xf32>) -> tensor<65536x32xf32>
      %309 = stablehlo.dot %308, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<65536x32xf32>, tensor<32x64xf32>) -> tensor<65536x64xf32>
      mhlo.return %309 : tensor<65536x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32x256xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x64xf32>) -> tensor<65536x64xf32>
    %55 = "mhlo.fusion"(%54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x64xf32>):
      %303 = stablehlo.slice %arg208 [0:65536, 0:32] : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
      mhlo.return %303 : tensor<65536x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
    %56 = mhlo.bitcast %55 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x32xf32>) -> tensor<256x256x32xf32>
    %57 = "mhlo.fusion"(%arg41, %44, %arg43, %56) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<256x16384x32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<256x256x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %304 = stablehlo.add %303, %arg209 : tensor<256x16384x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<256x256x32xf32>
      %306 = stablehlo.add %305, %arg211 : tensor<256x256x32xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x256x32xf32>) -> tensor<256x32x256xf32>
      %308 = stablehlo.dot_general %304, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x32xf32>, tensor<256x32x256xf32>) -> tensor<256x16384x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16384x256xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x16384x256xf32>
      mhlo.return %310 : tensor<256x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<256x16384x32xf32>, tensor<32xf32>, tensor<256x256x32xf32>) -> tensor<256x16384x256xf32>
    %58 = "mhlo.fusion"(%57) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x16384x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<256x16384x256xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x256xf32>
      %305 = stablehlo.subtract %arg208, %304 : tensor<256x16384x256xf32>
      %306 = stablehlo.exponential %305 : tensor<256x16384x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x256xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x256xf32>
      %309 = stablehlo.divide %306, %308 : tensor<256x16384x256xf32>
      mhlo.return %309 : tensor<256x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16384x256xf32>) -> tensor<256x16384x256xf32>
    %59 = "mhlo.fusion"(%54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x64xf32>):
      %303 = stablehlo.slice %arg208 [0:65536, 32:64] : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
      mhlo.return %303 : tensor<65536x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x64xf32>) -> tensor<65536x32xf32>
    %60 = mhlo.bitcast %59 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x32xf32>) -> tensor<256x256x32xf32>
    %61 = "mhlo.fusion"(%58, %arg45, %60) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x16384x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<256x256x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<256x256x32xf32>
      %304 = stablehlo.add %303, %arg210 : tensor<256x256x32xf32>
      %305 = stablehlo.dot_general %arg208, %304, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x256xf32>, tensor<256x256x32xf32>) -> tensor<256x16384x32xf32>
      mhlo.return %305 : tensor<256x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16384x256xf32>, tensor<32xf32>, tensor<256x256x32xf32>) -> tensor<256x16384x32xf32>
    %62 = mhlo.bitcast %61 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x32xf32>) -> tensor<4194304x32xf32>
    %63 = "mhlo.fusion"(%62, %arg50) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4194304x32xf32>, %arg209: tensor<32x32xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
      mhlo.return %303 : tensor<4194304x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x32xf32>, tensor<32x32xf32>) -> tensor<4194304x32xf32>
    %64 = "mhlo.fusion"(%40, %63, %arg51) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x16384x32xf32>, %arg209: tensor<4194304x32xf32>, %arg210: tensor<32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %304 = stablehlo.add %303, %arg209 : tensor<4194304x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
      %306 = stablehlo.add %305, %arg208 : tensor<256x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<256x16384xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %311 = stablehlo.subtract %306, %310 : tensor<256x16384x32xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<256x16384x32xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<256x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %316 = stablehlo.add %314, %315 : tensor<256x16384xf32>
      %317 = stablehlo.rsqrt %316 : tensor<256x16384xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<256x16384x32xf32>
      mhlo.return %319 : tensor<256x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16384x32xf32>, tensor<4194304x32xf32>, tensor<32xf32>) -> tensor<256x16384x32xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x32xf32>) -> tensor<4194304x32xf32>
    %66 = "mhlo.fusion"(%65, %arg52, %arg53, %arg54) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4194304x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<4194304x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %306 = stablehlo.add %304, %305 : tensor<4194304x32xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x32xf32>, tensor<128x32xf32>) -> tensor<4194304x128xf32>
      mhlo.return %307 : tensor<4194304x128xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32xf32>) -> tensor<4194304x128xf32>
    %67 = "mhlo.fusion"(%66, %arg55) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<4194304x128xf32>, %arg209: tensor<128xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<4194304x128xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<4194304x128xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x128xf32>) -> tensor<256x16384x128xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1] : (tensor<256x16384x128xf32>) -> tensor<256x128x16384xf32>
      mhlo.return %306 : tensor<256x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<4194304x128xf32>, tensor<128xf32>) -> tensor<256x128x16384xf32>
    %68 = mhlo.bitcast %67 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x128x16384xf32>) -> tensor<256x128x128x128xf32>
    %69 = stablehlo.custom_call @__cudnn$convForward(%68, %arg56) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<256x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<256x128x128x128xf32>, tensor<0xui8>>
    %70 = stablehlo.get_tuple_element %69[0] : (tuple<tensor<256x128x128x128xf32>, tensor<0xui8>>) -> tensor<256x128x128x128xf32>
    %71 = "mhlo.fusion"(%70, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x128x128x128xf32>
      %304 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<256x128x128x128xf32>
      %305 = stablehlo.add %arg208, %304 : tensor<256x128x128x128xf32>
      %306 = stablehlo.multiply %303, %305 : tensor<256x128x128x128xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x128x128x128xf32>) -> tensor<256x128x16384xf32>
      %308 = stablehlo.negate %305 : tensor<256x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x128x128x128xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x128x128x128xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x128x128x128xf32>) -> tensor<256x128x16384xf32>
      %312 = stablehlo.abs %311 : tensor<256x128x16384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %314 = stablehlo.compare LT, %312, %313 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %316 = stablehlo.multiply %311, %311 : tensor<256x128x16384xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x128x16384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x128x16384xf32>
      %320 = stablehlo.multiply %319, %316 : tensor<256x128x16384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x128x16384xf32>
      %323 = stablehlo.multiply %322, %316 : tensor<256x128x16384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %325 = stablehlo.add %323, %324 : tensor<256x128x16384xf32>
      %326 = stablehlo.multiply %325, %316 : tensor<256x128x16384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %328 = stablehlo.add %326, %327 : tensor<256x128x16384xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x128x16384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %331 = stablehlo.add %329, %330 : tensor<256x128x16384xf32>
      %332 = stablehlo.multiply %331, %316 : tensor<256x128x16384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %334 = stablehlo.add %332, %333 : tensor<256x128x16384xf32>
      %335 = stablehlo.multiply %311, %334 : tensor<256x128x16384xf32>
      %336 = stablehlo.subtract %313, %335 : tensor<256x128x16384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %338 = stablehlo.compare LT, %311, %337 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %340 = stablehlo.negate %316 : tensor<256x128x16384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %342 = stablehlo.compare LT, %340, %341 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
      %343 = stablehlo.exponential %340 : tensor<256x128x16384xf32>
      %344 = stablehlo.divide %313, %312 : tensor<256x128x16384xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<256x128x16384xf32>
      %346 = stablehlo.compare LT, %312, %339 : (tensor<256x128x16384xf32>, tensor<256x128x16384xf32>) -> tensor<256x128x16384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %348 = stablehlo.divide %313, %316 : tensor<256x128x16384xf32>
      %349 = stablehlo.multiply %347, %348 : tensor<256x128x16384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %351 = stablehlo.add %349, %350 : tensor<256x128x16384xf32>
      %352 = stablehlo.multiply %351, %348 : tensor<256x128x16384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %354 = stablehlo.add %352, %353 : tensor<256x128x16384xf32>
      %355 = stablehlo.multiply %354, %348 : tensor<256x128x16384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %357 = stablehlo.add %355, %356 : tensor<256x128x16384xf32>
      %358 = stablehlo.multiply %357, %348 : tensor<256x128x16384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %360 = stablehlo.add %358, %359 : tensor<256x128x16384xf32>
      %361 = stablehlo.multiply %360, %348 : tensor<256x128x16384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %363 = stablehlo.add %361, %362 : tensor<256x128x16384xf32>
      %364 = stablehlo.multiply %363, %348 : tensor<256x128x16384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %366 = stablehlo.add %364, %365 : tensor<256x128x16384xf32>
      %367 = stablehlo.multiply %366, %348 : tensor<256x128x16384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %369 = stablehlo.add %367, %368 : tensor<256x128x16384xf32>
      %370 = stablehlo.multiply %369, %348 : tensor<256x128x16384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %372 = stablehlo.add %370, %371 : tensor<256x128x16384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %374 = stablehlo.multiply %373, %348 : tensor<256x128x16384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %376 = stablehlo.add %374, %375 : tensor<256x128x16384xf32>
      %377 = stablehlo.multiply %376, %348 : tensor<256x128x16384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x128x16384xf32>
      %380 = stablehlo.multiply %379, %348 : tensor<256x128x16384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %382 = stablehlo.add %380, %381 : tensor<256x128x16384xf32>
      %383 = stablehlo.multiply %382, %348 : tensor<256x128x16384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %385 = stablehlo.add %383, %384 : tensor<256x128x16384xf32>
      %386 = stablehlo.multiply %385, %348 : tensor<256x128x16384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %388 = stablehlo.add %386, %387 : tensor<256x128x16384xf32>
      %389 = stablehlo.multiply %388, %348 : tensor<256x128x16384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %391 = stablehlo.add %389, %390 : tensor<256x128x16384xf32>
      %392 = stablehlo.multiply %391, %348 : tensor<256x128x16384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x128x16384xf32>
      %394 = stablehlo.add %392, %393 : tensor<256x128x16384xf32>
      %395 = stablehlo.select %346, %372, %394 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
      %396 = stablehlo.multiply %345, %395 : tensor<256x128x16384xf32>
      %397 = stablehlo.select %342, %337, %396 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
      %398 = stablehlo.subtract %339, %397 : tensor<256x128x16384xf32>
      %399 = stablehlo.select %338, %398, %397 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
      %400 = stablehlo.select %314, %336, %399 : tensor<256x128x16384xi1>, tensor<256x128x16384xf32>
      %401 = stablehlo.multiply %307, %400 : tensor<256x128x16384xf32>
      mhlo.return %401 : tensor<256x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x128x128xf32>, tensor<128xf32>) -> tensor<256x128x16384xf32>
    %72 = "mhlo.fusion"(%71, %arg58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x128x16384xf32>, %arg209: tensor<32x128xf32>):
      %303 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<256x128x16384xf32>) -> tensor<256x16384x128xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16384x128xf32>) -> tensor<4194304x128xf32>
      %305 = stablehlo.dot_general %304, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4194304x128xf32>, tensor<32x128xf32>) -> tensor<4194304x32xf32>
      mhlo.return %305 : tensor<4194304x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x16384xf32>, tensor<32x128xf32>) -> tensor<4194304x32xf32>
    %73:2 = "mhlo.fusion"(%arg185, %arg184, %72, %arg59, %40, %63, %arg51) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<4194304x32xf32>, %arg211: tensor<32xf32>, %arg212: tensor<256x16384x32xf32>, %arg213: tensor<4194304x32xf32>, %arg214: tensor<32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %304 = stablehlo.add %303, %arg210 : tensor<4194304x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
      %306 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<32xf32>) -> tensor<4194304x32xf32>
      %307 = stablehlo.add %306, %arg213 : tensor<4194304x32xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4194304x32xf32>) -> tensor<256x16384x32xf32>
      %309 = stablehlo.add %308, %arg212 : tensor<256x16384x32xf32>
      %310 = stablehlo.add %305, %309 : tensor<256x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<256x16384xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<256x16384x32xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<256x16384x32xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16384x32xf32>, tensor<f32>) -> tensor<256x16384xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<256x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x16384xf32>
      %320 = stablehlo.add %318, %319 : tensor<256x16384xf32>
      %321 = stablehlo.rsqrt %320 : tensor<256x16384xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<256x16384xf32>) -> tensor<256x16384x32xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<256x16384x32xf32>
      %324 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<256x16384x32xf32>
      %326 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %327 = stablehlo.add %325, %326 : tensor<256x16384x32xf32>
      %328 = stablehlo.transpose %327, dims = [0, 2, 1] : (tensor<256x16384x32xf32>) -> tensor<256x32x16384xf32>
      mhlo.return %328, %323 : tensor<256x32x16384xf32>, tensor<256x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<4194304x32xf32>, tensor<32xf32>, tensor<256x16384x32xf32>, tensor<4194304x32xf32>, tensor<32xf32>) -> (tensor<256x32x16384xf32>, tensor<256x16384x32xf32>)
    %74 = stablehlo.tuple %73#0, %73#1 {xla_shape = "(f32[256,32,16384]{2,1,0}, f32[256,16384,32]{2,1,0})"} : tuple<tensor<256x32x16384xf32>, tensor<256x16384x32xf32>>
    %75 = mhlo.bitcast %73#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x16384xf32>) -> tensor<256x32x128x128xf32>
    %76 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%75, %arg4, %arg5) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,64,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<256x32x128x128xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>) -> tuple<tensor<256x64x64x64xf32>, tensor<0xui8>>
    %77 = stablehlo.get_tuple_element %76[0] : (tuple<tensor<256x64x64x64xf32>, tensor<0xui8>>) -> tensor<256x64x64x64xf32>
    %78 = "mhlo.fusion"(%arg7, %arg6, %77) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<256x64x64x64xf32>):
      %303 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x64x64xf32>) -> tensor<256x64x4096xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<256x64x4096xf32>) -> tensor<256x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x64x64xf32>, tensor<f32>) -> tensor<256x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x64x64xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<256x64x64xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x64xf32>) -> tensor<256x4096xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %310 = stablehlo.subtract %304, %309 : tensor<256x4096x64xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<256x4096x64xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %314 = stablehlo.multiply %312, %313 : tensor<256x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %316 = stablehlo.add %314, %315 : tensor<256x4096xf32>
      %317 = stablehlo.rsqrt %316 : tensor<256x4096xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %319 = stablehlo.multiply %310, %318 : tensor<256x4096x64xf32>
      %320 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<256x4096x64xf32>
      %322 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %323 = stablehlo.add %321, %322 : tensor<256x4096x64xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %325 = stablehlo.multiply %324, %313 : tensor<256x4096xf32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %327 = stablehlo.subtract %323, %326 : tensor<256x4096x64xf32>
      %328 = stablehlo.multiply %327, %327 : tensor<256x4096x64xf32>
      %329 = stablehlo.reduce(%328 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %330 = stablehlo.multiply %329, %313 : tensor<256x4096xf32>
      %331 = stablehlo.add %330, %315 : tensor<256x4096xf32>
      %332 = stablehlo.rsqrt %331 : tensor<256x4096xf32>
      %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %334 = stablehlo.multiply %327, %333 : tensor<256x4096x64xf32>
      mhlo.return %334 : tensor<256x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x64x64x64xf32>) -> tensor<256x4096x64xf32>
    %79 = mhlo.bitcast %78 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x64xf32>) -> tensor<1048576x64xf32>
    %80 = "mhlo.fusion"(%79, %arg60, %arg61, %arg62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<1048576x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<1048576x64xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
      mhlo.return %307 : tensor<1048576x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
    %81 = "mhlo.fusion"(%80, %arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<1048576x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x2x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x4096x2x32xf32>) -> tensor<256x2x4096x32xf32>
      mhlo.return %306 : tensor<256x2x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64xf32>) -> tensor<256x2x4096x32xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x2x4096x32xf32>) -> tensor<512x4096x32xf32>
    %83 = "mhlo.fusion"(%arg61, %78, %arg60) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<256x4096x64xf32>, %arg210: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %304 = stablehlo.multiply %arg209, %303 : tensor<256x4096x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x4096x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x4096x64xf32>) -> tensor<256x64x4096xf32>
      mhlo.return %307 : tensor<256x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<256x4096x64xf32>, tensor<64xf32>) -> tensor<256x64x4096xf32>
    %84 = mhlo.bitcast %83 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x4096xf32>) -> tensor<256x64x64x64xf32>
    %85 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%84, %arg68, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,64,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x64x64x64xf32>, tensor<64x64x4x4xf32>, tensor<64xf32>) -> tuple<tensor<256x64x16x16xf32>, tensor<0xui8>>
    %86 = stablehlo.get_tuple_element %85[0] : (tuple<tensor<256x64x16x16xf32>, tensor<0xui8>>) -> tensor<256x64x16x16xf32>
    %87 = "mhlo.fusion"(%86) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x64x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      mhlo.return %303 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64x16x16xf32>) -> tensor<256x16x16xf32>
    %88 = "mhlo.fusion"(%87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x16x16xf32>):
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x16x16xf32>
      mhlo.return %304 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x16xf32>) -> tensor<256x16x16xf32>
    %89 = "mhlo.fusion"(%88, %86) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x16x16xf32>, %arg209: tensor<256x64x16x16xf32>):
      %303 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
      %304 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
      %306 = stablehlo.subtract %303, %305 : tensor<256x64x256xf32>
      %307 = stablehlo.multiply %306, %306 : tensor<256x64x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      mhlo.return %308 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x16xf32>, tensor<256x64x16x16xf32>) -> tensor<256x256xf32>
    %90 = "mhlo.fusion"(%89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>):
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<256x256xf32>
      mhlo.return %307 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %91 = "mhlo.fusion"(%90, %88, %86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256x16x16xf32>, %arg210: tensor<256x64x16x16xf32>):
      %303 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
      %304 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
      %306 = stablehlo.subtract %303, %305 : tensor<256x64x256xf32>
      %307 = stablehlo.broadcast_in_dim %arg208, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<256x64x256xf32>
      mhlo.return %308 : tensor<256x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256x16x16xf32>, tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
    %92 = "mhlo.fusion"(%91, %arg70, %arg71, %arg64, %arg66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x64x256xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>, %arg212: tensor<64x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<256x64x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x64x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<256x64x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x64x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x64x256xf32>) -> tensor<256x256x64xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x64xf32>) -> tensor<65536x64xf32>
      %309 = stablehlo.concatenate %arg211, %arg212, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
      %311 = stablehlo.dot %308, %310, precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<64x128xf32>) -> tensor<65536x128xf32>
      mhlo.return %311 : tensor<65536x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<65536x128xf32>
    %93 = "mhlo.fusion"(%92, %arg65) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x128xf32>, %arg209: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %304 = stablehlo.slice %arg208 [0:65536, 0:64] : (tensor<65536x128xf32>) -> tensor<65536x64xf32>
      %305 = stablehlo.add %303, %304 : tensor<65536x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<256x256x2x32xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 3, 1] : (tensor<256x256x2x32xf32>) -> tensor<256x2x32x256xf32>
      mhlo.return %307 : tensor<256x2x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x128xf32>, tensor<64xf32>) -> tensor<256x2x32x256xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x2x32x256xf32>) -> tensor<512x32x256xf32>
    %95 = "mhlo.fusion"(%82, %94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x4096x32xf32>, %arg209: tensor<512x32x256xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x4096x32xf32>, tensor<512x32x256xf32>) -> tensor<512x4096x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x4096x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<512x4096x256xf32>
      mhlo.return %305 : tensor<512x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096x32xf32>, tensor<512x32x256xf32>) -> tensor<512x4096x256xf32>
    %96 = "mhlo.fusion"(%95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x4096x256xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x4096x256xf32>) -> tensor<256x2x4096x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %304 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<512x4096x256xf32>, tensor<f32>) -> tensor<512x4096xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x4096xf32>) -> tensor<256x2x4096xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2] : (tensor<256x2x4096xf32>) -> tensor<256x2x4096x256xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<256x2x4096x256xf32>
      %308 = stablehlo.exponential %307 : tensor<256x2x4096x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x2x4096x256xf32>, tensor<f32>) -> tensor<256x2x4096xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<256x2x4096xf32>) -> tensor<256x2x4096x256xf32>
      %311 = stablehlo.divide %308, %310 : tensor<256x2x4096x256xf32>
      mhlo.return %311 : tensor<256x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096x256xf32>) -> tensor<256x2x4096x256xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x2x4096x256xf32>) -> tensor<512x4096x256xf32>
    %98 = "mhlo.fusion"(%92, %arg67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x128xf32>, %arg209: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %304 = stablehlo.slice %arg208 [0:65536, 64:128] : (tensor<65536x128xf32>) -> tensor<65536x64xf32>
      %305 = stablehlo.add %303, %304 : tensor<65536x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<256x256x2x32xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<256x256x2x32xf32>) -> tensor<256x2x256x32xf32>
      mhlo.return %307 : tensor<256x2x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x128xf32>, tensor<64xf32>) -> tensor<256x2x256x32xf32>
    %99 = mhlo.bitcast %98 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x2x256x32xf32>) -> tensor<512x256x32xf32>
    %100 = "mhlo.fusion"(%97, %99) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x4096x256xf32>, %arg209: tensor<512x256x32xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x4096x256xf32>, tensor<512x256x32xf32>) -> tensor<512x4096x32xf32>
      mhlo.return %303 : tensor<512x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096x256xf32>, tensor<512x256x32xf32>) -> tensor<512x4096x32xf32>
    %101 = "mhlo.fusion"(%100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<512x4096x32xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x4096x32xf32>) -> tensor<256x2x4096x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<256x2x4096x32xf32>) -> tensor<256x4096x2x32xf32>
      mhlo.return %304 : tensor<256x4096x2x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096x32xf32>) -> tensor<256x4096x2x32xf32>
    %102 = mhlo.bitcast %101 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x4096x2x32xf32>) -> tensor<1048576x64xf32>
    %103 = "mhlo.fusion"(%102, %arg72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64x64xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
      mhlo.return %303 : tensor<1048576x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
    %104 = "mhlo.fusion"(%arg7, %arg6, %77, %103, %arg73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<256x64x64x64xf32>, %arg211: tensor<1048576x64xf32>, %arg212: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.add %303, %arg211 : tensor<1048576x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
      %306 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x64x64xf32>) -> tensor<256x64x4096xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x64x4096xf32>) -> tensor<256x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x64x64xf32>, tensor<f32>) -> tensor<256x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x64x64xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x64x64xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x64xf32>) -> tensor<256x4096xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %313 = stablehlo.subtract %307, %312 : tensor<256x4096x64xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<256x4096x64xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %316 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x4096xf32>
      %320 = stablehlo.rsqrt %319 : tensor<256x4096xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %322 = stablehlo.multiply %313, %321 : tensor<256x4096x64xf32>
      %323 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<256x4096x64xf32>
      %325 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %326 = stablehlo.add %324, %325 : tensor<256x4096x64xf32>
      %327 = stablehlo.add %305, %326 : tensor<256x4096x64xf32>
      %328 = stablehlo.reduce(%327 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x4096xf32>
      %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %331 = stablehlo.subtract %327, %330 : tensor<256x4096x64xf32>
      %332 = stablehlo.multiply %331, %331 : tensor<256x4096x64xf32>
      %333 = stablehlo.reduce(%332 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %334 = stablehlo.multiply %333, %316 : tensor<256x4096xf32>
      %335 = stablehlo.add %334, %318 : tensor<256x4096xf32>
      %336 = stablehlo.rsqrt %335 : tensor<256x4096xf32>
      %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %338 = stablehlo.multiply %331, %337 : tensor<256x4096x64xf32>
      mhlo.return %338 : tensor<256x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x64x64x64xf32>, tensor<1048576x64xf32>, tensor<64xf32>) -> tensor<256x4096x64xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x64xf32>) -> tensor<1048576x64xf32>
    %106 = "mhlo.fusion"(%105, %arg74, %arg75, %arg76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<1048576x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<1048576x64xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<256x64xf32>) -> tensor<1048576x256xf32>
      mhlo.return %307 : tensor<1048576x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<1048576x256xf32>
    %107 = "mhlo.fusion"(%106, %arg77) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1048576x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<1048576x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<1048576x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x256xf32>) -> tensor<256x4096x256xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1] : (tensor<256x4096x256xf32>) -> tensor<256x256x4096xf32>
      mhlo.return %306 : tensor<256x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x256xf32>, tensor<256xf32>) -> tensor<256x256x4096xf32>
    %108 = mhlo.bitcast %107 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x4096xf32>) -> tensor<256x256x64x64xf32>
    %109 = stablehlo.custom_call @__cudnn$convForward(%108, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<256x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<256x256x64x64xf32>, tensor<0xui8>>
    %110 = stablehlo.get_tuple_element %109[0] : (tuple<tensor<256x256x64x64xf32>, tensor<0xui8>>) -> tensor<256x256x64x64xf32>
    %111 = "mhlo.fusion"(%110, %arg79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256x64x64xf32>, %arg209: tensor<256xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256x64x64xf32>
      %304 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256x64x64xf32>
      %305 = stablehlo.add %arg208, %304 : tensor<256x256x64x64xf32>
      %306 = stablehlo.multiply %303, %305 : tensor<256x256x64x64xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x64x64xf32>) -> tensor<256x256x4096xf32>
      %308 = stablehlo.negate %305 : tensor<256x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256x64x64xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x256x64x64xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x64x64xf32>) -> tensor<256x256x4096xf32>
      %312 = stablehlo.abs %311 : tensor<256x256x4096xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %314 = stablehlo.compare LT, %312, %313 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %316 = stablehlo.multiply %311, %311 : tensor<256x256x4096xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x256x4096xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x256x4096xf32>
      %320 = stablehlo.multiply %319, %316 : tensor<256x256x4096xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x256x4096xf32>
      %323 = stablehlo.multiply %322, %316 : tensor<256x256x4096xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %325 = stablehlo.add %323, %324 : tensor<256x256x4096xf32>
      %326 = stablehlo.multiply %325, %316 : tensor<256x256x4096xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %328 = stablehlo.add %326, %327 : tensor<256x256x4096xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x256x4096xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %331 = stablehlo.add %329, %330 : tensor<256x256x4096xf32>
      %332 = stablehlo.multiply %331, %316 : tensor<256x256x4096xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %334 = stablehlo.add %332, %333 : tensor<256x256x4096xf32>
      %335 = stablehlo.multiply %311, %334 : tensor<256x256x4096xf32>
      %336 = stablehlo.subtract %313, %335 : tensor<256x256x4096xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %338 = stablehlo.compare LT, %311, %337 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %340 = stablehlo.negate %316 : tensor<256x256x4096xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %342 = stablehlo.compare LT, %340, %341 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
      %343 = stablehlo.exponential %340 : tensor<256x256x4096xf32>
      %344 = stablehlo.divide %313, %312 : tensor<256x256x4096xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<256x256x4096xf32>
      %346 = stablehlo.compare LT, %312, %339 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %348 = stablehlo.divide %313, %316 : tensor<256x256x4096xf32>
      %349 = stablehlo.multiply %347, %348 : tensor<256x256x4096xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %351 = stablehlo.add %349, %350 : tensor<256x256x4096xf32>
      %352 = stablehlo.multiply %351, %348 : tensor<256x256x4096xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %354 = stablehlo.add %352, %353 : tensor<256x256x4096xf32>
      %355 = stablehlo.multiply %354, %348 : tensor<256x256x4096xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %357 = stablehlo.add %355, %356 : tensor<256x256x4096xf32>
      %358 = stablehlo.multiply %357, %348 : tensor<256x256x4096xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %360 = stablehlo.add %358, %359 : tensor<256x256x4096xf32>
      %361 = stablehlo.multiply %360, %348 : tensor<256x256x4096xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %363 = stablehlo.add %361, %362 : tensor<256x256x4096xf32>
      %364 = stablehlo.multiply %363, %348 : tensor<256x256x4096xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %366 = stablehlo.add %364, %365 : tensor<256x256x4096xf32>
      %367 = stablehlo.multiply %366, %348 : tensor<256x256x4096xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %369 = stablehlo.add %367, %368 : tensor<256x256x4096xf32>
      %370 = stablehlo.multiply %369, %348 : tensor<256x256x4096xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %372 = stablehlo.add %370, %371 : tensor<256x256x4096xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %374 = stablehlo.multiply %373, %348 : tensor<256x256x4096xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %376 = stablehlo.add %374, %375 : tensor<256x256x4096xf32>
      %377 = stablehlo.multiply %376, %348 : tensor<256x256x4096xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x256x4096xf32>
      %380 = stablehlo.multiply %379, %348 : tensor<256x256x4096xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %382 = stablehlo.add %380, %381 : tensor<256x256x4096xf32>
      %383 = stablehlo.multiply %382, %348 : tensor<256x256x4096xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %385 = stablehlo.add %383, %384 : tensor<256x256x4096xf32>
      %386 = stablehlo.multiply %385, %348 : tensor<256x256x4096xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %388 = stablehlo.add %386, %387 : tensor<256x256x4096xf32>
      %389 = stablehlo.multiply %388, %348 : tensor<256x256x4096xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %391 = stablehlo.add %389, %390 : tensor<256x256x4096xf32>
      %392 = stablehlo.multiply %391, %348 : tensor<256x256x4096xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %394 = stablehlo.add %392, %393 : tensor<256x256x4096xf32>
      %395 = stablehlo.select %346, %372, %394 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
      %396 = stablehlo.multiply %345, %395 : tensor<256x256x4096xf32>
      %397 = stablehlo.select %342, %337, %396 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
      %398 = stablehlo.subtract %339, %397 : tensor<256x256x4096xf32>
      %399 = stablehlo.select %338, %398, %397 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
      %400 = stablehlo.select %314, %336, %399 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
      %401 = stablehlo.multiply %307, %400 : tensor<256x256x4096xf32>
      mhlo.return %401 : tensor<256x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x64x64xf32>, tensor<256xf32>) -> tensor<256x256x4096xf32>
    %112 = "mhlo.fusion"(%111, %arg80) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256x4096xf32>, %arg209: tensor<64x256xf32>):
      %303 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<256x256x4096xf32>) -> tensor<256x4096x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x256xf32>) -> tensor<1048576x256xf32>
      %305 = stablehlo.dot_general %304, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x256xf32>, tensor<64x256xf32>) -> tensor<1048576x64xf32>
      mhlo.return %305 : tensor<1048576x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x4096xf32>, tensor<64x256xf32>) -> tensor<1048576x64xf32>
    %113 = "mhlo.fusion"(%112, %arg81, %arg7, %arg6, %77, %103, %arg73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64xf32>, %arg212: tensor<256x64x64x64xf32>, %arg213: tensor<1048576x64xf32>, %arg214: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<1048576x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
      %306 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %307 = stablehlo.add %306, %arg213 : tensor<1048576x64xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
      %309 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x64x64xf32>) -> tensor<256x64x4096xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1] : (tensor<256x64x4096xf32>) -> tensor<256x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x64x64xf32>, tensor<f32>) -> tensor<256x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x64x64xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<256x64x64xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x64xf32>) -> tensor<256x4096xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %316 = stablehlo.subtract %310, %315 : tensor<256x4096x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<256x4096x64xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<256x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x4096xf32>
      %323 = stablehlo.rsqrt %322 : tensor<256x4096xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %325 = stablehlo.multiply %316, %324 : tensor<256x4096x64xf32>
      %326 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %327 = stablehlo.multiply %325, %326 : tensor<256x4096x64xf32>
      %328 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %329 = stablehlo.add %327, %328 : tensor<256x4096x64xf32>
      %330 = stablehlo.add %308, %329 : tensor<256x4096x64xf32>
      %331 = stablehlo.add %305, %330 : tensor<256x4096x64xf32>
      mhlo.return %331 : tensor<256x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x64x64xf32>, tensor<1048576x64xf32>, tensor<64xf32>) -> tensor<256x4096x64xf32>
    %114 = "mhlo.fusion"(%113) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x64xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<256x4096xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %307 = stablehlo.subtract %arg208, %306 : tensor<256x4096x64xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<256x4096x64xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %310 = stablehlo.multiply %309, %304 : tensor<256x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %312 = stablehlo.add %310, %311 : tensor<256x4096xf32>
      %313 = stablehlo.rsqrt %312 : tensor<256x4096xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %315 = stablehlo.multiply %307, %314 : tensor<256x4096x64xf32>
      mhlo.return %315 : tensor<256x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x64xf32>) -> tensor<256x4096x64xf32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x64xf32>) -> tensor<1048576x64xf32>
    %116 = "mhlo.fusion"(%115, %arg82, %arg83, %arg84) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<1048576x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<1048576x64xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
      mhlo.return %307 : tensor<1048576x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
    %117 = "mhlo.fusion"(%116, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<1048576x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x2x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x4096x2x32xf32>) -> tensor<256x2x4096x32xf32>
      mhlo.return %306 : tensor<256x2x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64xf32>) -> tensor<256x2x4096x32xf32>
    %118 = mhlo.bitcast %117 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x2x4096x32xf32>) -> tensor<512x4096x32xf32>
    %119 = "mhlo.fusion"(%arg83, %114, %arg82) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<256x4096x64xf32>, %arg210: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %304 = stablehlo.multiply %arg209, %303 : tensor<256x4096x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x4096x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x4096x64xf32>) -> tensor<256x64x4096xf32>
      mhlo.return %307 : tensor<256x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<256x4096x64xf32>, tensor<64xf32>) -> tensor<256x64x4096xf32>
    %120 = mhlo.bitcast %119 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x4096xf32>) -> tensor<256x64x64x64xf32>
    %121 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%120, %arg90, %arg91) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,64,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x64x64x64xf32>, tensor<64x64x4x4xf32>, tensor<64xf32>) -> tuple<tensor<256x64x16x16xf32>, tensor<0xui8>>
    %122 = stablehlo.get_tuple_element %121[0] : (tuple<tensor<256x64x16x16xf32>, tensor<0xui8>>) -> tensor<256x64x16x16xf32>
    %123 = "mhlo.fusion"(%122) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x64x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      mhlo.return %303 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64x16x16xf32>) -> tensor<256x16x16xf32>
    %124 = "mhlo.fusion"(%123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x16x16xf32>):
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x16x16xf32>
      mhlo.return %304 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x16xf32>) -> tensor<256x16x16xf32>
    %125 = "mhlo.fusion"(%124, %122) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x16x16xf32>, %arg209: tensor<256x64x16x16xf32>):
      %303 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
      %304 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
      %306 = stablehlo.subtract %303, %305 : tensor<256x64x256xf32>
      %307 = stablehlo.multiply %306, %306 : tensor<256x64x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x64x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      mhlo.return %308 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x16xf32>, tensor<256x64x16x16xf32>) -> tensor<256x256xf32>
    %126 = "mhlo.fusion"(%125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>):
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<256x256xf32>
      mhlo.return %307 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %127 = "mhlo.fusion"(%126, %124, %122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256x16x16xf32>, %arg210: tensor<256x64x16x16xf32>):
      %303 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
      %304 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
      %306 = stablehlo.subtract %303, %305 : tensor<256x64x256xf32>
      %307 = stablehlo.broadcast_in_dim %arg208, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x64x256xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<256x64x256xf32>
      mhlo.return %308 : tensor<256x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256x16x16xf32>, tensor<256x64x16x16xf32>) -> tensor<256x64x256xf32>
    %128 = "mhlo.fusion"(%127, %arg92, %arg93, %arg86, %arg88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x64x256xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>, %arg212: tensor<64x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<256x64x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x64x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<256x64x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x64x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x64x256xf32>) -> tensor<256x256x64xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x64xf32>) -> tensor<65536x64xf32>
      %309 = stablehlo.concatenate %arg211, %arg212, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
      %311 = stablehlo.dot %308, %310, precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<64x128xf32>) -> tensor<65536x128xf32>
      mhlo.return %311 : tensor<65536x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<65536x128xf32>
    %129 = "mhlo.fusion"(%128, %arg87) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x128xf32>, %arg209: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %304 = stablehlo.slice %arg208 [0:65536, 0:64] : (tensor<65536x128xf32>) -> tensor<65536x64xf32>
      %305 = stablehlo.add %303, %304 : tensor<65536x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<256x256x2x32xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 3, 1] : (tensor<256x256x2x32xf32>) -> tensor<256x2x32x256xf32>
      mhlo.return %307 : tensor<256x2x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x128xf32>, tensor<64xf32>) -> tensor<256x2x32x256xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x2x32x256xf32>) -> tensor<512x32x256xf32>
    %131 = "mhlo.fusion"(%118, %130) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x4096x32xf32>, %arg209: tensor<512x32x256xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x4096x32xf32>, tensor<512x32x256xf32>) -> tensor<512x4096x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x4096x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<512x4096x256xf32>
      mhlo.return %305 : tensor<512x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096x32xf32>, tensor<512x32x256xf32>) -> tensor<512x4096x256xf32>
    %132 = "mhlo.fusion"(%131) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x4096x256xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x4096x256xf32>) -> tensor<256x2x4096x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %304 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<512x4096x256xf32>, tensor<f32>) -> tensor<512x4096xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x4096xf32>) -> tensor<256x2x4096xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2] : (tensor<256x2x4096xf32>) -> tensor<256x2x4096x256xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<256x2x4096x256xf32>
      %308 = stablehlo.exponential %307 : tensor<256x2x4096x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x2x4096x256xf32>, tensor<f32>) -> tensor<256x2x4096xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<256x2x4096xf32>) -> tensor<256x2x4096x256xf32>
      %311 = stablehlo.divide %308, %310 : tensor<256x2x4096x256xf32>
      mhlo.return %311 : tensor<256x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096x256xf32>) -> tensor<256x2x4096x256xf32>
    %133 = mhlo.bitcast %132 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x2x4096x256xf32>) -> tensor<512x4096x256xf32>
    %134 = "mhlo.fusion"(%128, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x128xf32>, %arg209: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %304 = stablehlo.slice %arg208 [0:65536, 64:128] : (tensor<65536x128xf32>) -> tensor<65536x64xf32>
      %305 = stablehlo.add %303, %304 : tensor<65536x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<256x256x2x32xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<256x256x2x32xf32>) -> tensor<256x2x256x32xf32>
      mhlo.return %307 : tensor<256x2x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x128xf32>, tensor<64xf32>) -> tensor<256x2x256x32xf32>
    %135 = mhlo.bitcast %134 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x2x256x32xf32>) -> tensor<512x256x32xf32>
    %136 = "mhlo.fusion"(%133, %135) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<512x4096x256xf32>, %arg209: tensor<512x256x32xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x4096x256xf32>, tensor<512x256x32xf32>) -> tensor<512x4096x32xf32>
      mhlo.return %303 : tensor<512x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096x256xf32>, tensor<512x256x32xf32>) -> tensor<512x4096x32xf32>
    %137 = "mhlo.fusion"(%136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<512x4096x32xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x4096x32xf32>) -> tensor<256x2x4096x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<256x2x4096x32xf32>) -> tensor<256x4096x2x32xf32>
      mhlo.return %304 : tensor<256x4096x2x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096x32xf32>) -> tensor<256x4096x2x32xf32>
    %138 = mhlo.bitcast %137 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x4096x2x32xf32>) -> tensor<1048576x64xf32>
    %139 = "mhlo.fusion"(%138, %arg94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64x64xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
      mhlo.return %303 : tensor<1048576x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64x64xf32>) -> tensor<1048576x64xf32>
    %140 = "mhlo.fusion"(%113, %139, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x64xf32>, %arg209: tensor<1048576x64xf32>, %arg210: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.add %303, %arg209 : tensor<1048576x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
      %306 = stablehlo.add %305, %arg208 : tensor<256x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<256x4096xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %311 = stablehlo.subtract %306, %310 : tensor<256x4096x64xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<256x4096x64xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<256x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %316 = stablehlo.add %314, %315 : tensor<256x4096xf32>
      %317 = stablehlo.rsqrt %316 : tensor<256x4096xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<256x4096x64xf32>
      mhlo.return %319 : tensor<256x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x64xf32>, tensor<1048576x64xf32>, tensor<64xf32>) -> tensor<256x4096x64xf32>
    %141 = mhlo.bitcast %140 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x64xf32>) -> tensor<1048576x64xf32>
    %142 = "mhlo.fusion"(%141, %arg96, %arg97, %arg98) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1048576x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<1048576x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<1048576x64xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x64xf32>, tensor<256x64xf32>) -> tensor<1048576x256xf32>
      mhlo.return %307 : tensor<1048576x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<1048576x256xf32>
    %143 = "mhlo.fusion"(%142, %arg99) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1048576x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<1048576x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<1048576x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x256xf32>) -> tensor<256x4096x256xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1] : (tensor<256x4096x256xf32>) -> tensor<256x256x4096xf32>
      mhlo.return %306 : tensor<256x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x256xf32>, tensor<256xf32>) -> tensor<256x256x4096xf32>
    %144 = mhlo.bitcast %143 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x4096xf32>) -> tensor<256x256x64x64xf32>
    %145 = stablehlo.custom_call @__cudnn$convForward(%144, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<256x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<256x256x64x64xf32>, tensor<0xui8>>
    %146 = stablehlo.get_tuple_element %145[0] : (tuple<tensor<256x256x64x64xf32>, tensor<0xui8>>) -> tensor<256x256x64x64xf32>
    %147 = "mhlo.fusion"(%146, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256x64x64xf32>, %arg209: tensor<256xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256x64x64xf32>
      %304 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256x64x64xf32>
      %305 = stablehlo.add %arg208, %304 : tensor<256x256x64x64xf32>
      %306 = stablehlo.multiply %303, %305 : tensor<256x256x64x64xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x64x64xf32>) -> tensor<256x256x4096xf32>
      %308 = stablehlo.negate %305 : tensor<256x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256x64x64xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x256x64x64xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x64x64xf32>) -> tensor<256x256x4096xf32>
      %312 = stablehlo.abs %311 : tensor<256x256x4096xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %314 = stablehlo.compare LT, %312, %313 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %316 = stablehlo.multiply %311, %311 : tensor<256x256x4096xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x256x4096xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x256x4096xf32>
      %320 = stablehlo.multiply %319, %316 : tensor<256x256x4096xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x256x4096xf32>
      %323 = stablehlo.multiply %322, %316 : tensor<256x256x4096xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %325 = stablehlo.add %323, %324 : tensor<256x256x4096xf32>
      %326 = stablehlo.multiply %325, %316 : tensor<256x256x4096xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %328 = stablehlo.add %326, %327 : tensor<256x256x4096xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x256x4096xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %331 = stablehlo.add %329, %330 : tensor<256x256x4096xf32>
      %332 = stablehlo.multiply %331, %316 : tensor<256x256x4096xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %334 = stablehlo.add %332, %333 : tensor<256x256x4096xf32>
      %335 = stablehlo.multiply %311, %334 : tensor<256x256x4096xf32>
      %336 = stablehlo.subtract %313, %335 : tensor<256x256x4096xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %338 = stablehlo.compare LT, %311, %337 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %340 = stablehlo.negate %316 : tensor<256x256x4096xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %342 = stablehlo.compare LT, %340, %341 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
      %343 = stablehlo.exponential %340 : tensor<256x256x4096xf32>
      %344 = stablehlo.divide %313, %312 : tensor<256x256x4096xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<256x256x4096xf32>
      %346 = stablehlo.compare LT, %312, %339 : (tensor<256x256x4096xf32>, tensor<256x256x4096xf32>) -> tensor<256x256x4096xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %348 = stablehlo.divide %313, %316 : tensor<256x256x4096xf32>
      %349 = stablehlo.multiply %347, %348 : tensor<256x256x4096xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %351 = stablehlo.add %349, %350 : tensor<256x256x4096xf32>
      %352 = stablehlo.multiply %351, %348 : tensor<256x256x4096xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %354 = stablehlo.add %352, %353 : tensor<256x256x4096xf32>
      %355 = stablehlo.multiply %354, %348 : tensor<256x256x4096xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %357 = stablehlo.add %355, %356 : tensor<256x256x4096xf32>
      %358 = stablehlo.multiply %357, %348 : tensor<256x256x4096xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %360 = stablehlo.add %358, %359 : tensor<256x256x4096xf32>
      %361 = stablehlo.multiply %360, %348 : tensor<256x256x4096xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %363 = stablehlo.add %361, %362 : tensor<256x256x4096xf32>
      %364 = stablehlo.multiply %363, %348 : tensor<256x256x4096xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %366 = stablehlo.add %364, %365 : tensor<256x256x4096xf32>
      %367 = stablehlo.multiply %366, %348 : tensor<256x256x4096xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %369 = stablehlo.add %367, %368 : tensor<256x256x4096xf32>
      %370 = stablehlo.multiply %369, %348 : tensor<256x256x4096xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %372 = stablehlo.add %370, %371 : tensor<256x256x4096xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %374 = stablehlo.multiply %373, %348 : tensor<256x256x4096xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %376 = stablehlo.add %374, %375 : tensor<256x256x4096xf32>
      %377 = stablehlo.multiply %376, %348 : tensor<256x256x4096xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x256x4096xf32>
      %380 = stablehlo.multiply %379, %348 : tensor<256x256x4096xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %382 = stablehlo.add %380, %381 : tensor<256x256x4096xf32>
      %383 = stablehlo.multiply %382, %348 : tensor<256x256x4096xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %385 = stablehlo.add %383, %384 : tensor<256x256x4096xf32>
      %386 = stablehlo.multiply %385, %348 : tensor<256x256x4096xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %388 = stablehlo.add %386, %387 : tensor<256x256x4096xf32>
      %389 = stablehlo.multiply %388, %348 : tensor<256x256x4096xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %391 = stablehlo.add %389, %390 : tensor<256x256x4096xf32>
      %392 = stablehlo.multiply %391, %348 : tensor<256x256x4096xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x256x4096xf32>
      %394 = stablehlo.add %392, %393 : tensor<256x256x4096xf32>
      %395 = stablehlo.select %346, %372, %394 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
      %396 = stablehlo.multiply %345, %395 : tensor<256x256x4096xf32>
      %397 = stablehlo.select %342, %337, %396 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
      %398 = stablehlo.subtract %339, %397 : tensor<256x256x4096xf32>
      %399 = stablehlo.select %338, %398, %397 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
      %400 = stablehlo.select %314, %336, %399 : tensor<256x256x4096xi1>, tensor<256x256x4096xf32>
      %401 = stablehlo.multiply %307, %400 : tensor<256x256x4096xf32>
      mhlo.return %401 : tensor<256x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x64x64xf32>, tensor<256xf32>) -> tensor<256x256x4096xf32>
    %148 = "mhlo.fusion"(%147, %arg102) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256x4096xf32>, %arg209: tensor<64x256xf32>):
      %303 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<256x256x4096xf32>) -> tensor<256x4096x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x256xf32>) -> tensor<1048576x256xf32>
      %305 = stablehlo.dot_general %304, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x256xf32>, tensor<64x256xf32>) -> tensor<1048576x64xf32>
      mhlo.return %305 : tensor<1048576x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x4096xf32>, tensor<64x256xf32>) -> tensor<1048576x64xf32>
    %149:2 = "mhlo.fusion"(%arg187, %arg186, %148, %arg103, %113, %139, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1048576x64xf32>, %arg211: tensor<64xf32>, %arg212: tensor<256x4096x64xf32>, %arg213: tensor<1048576x64xf32>, %arg214: tensor<64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %304 = stablehlo.add %303, %arg210 : tensor<1048576x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
      %306 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<64xf32>) -> tensor<1048576x64xf32>
      %307 = stablehlo.add %306, %arg213 : tensor<1048576x64xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x64xf32>) -> tensor<256x4096x64xf32>
      %309 = stablehlo.add %308, %arg212 : tensor<256x4096x64xf32>
      %310 = stablehlo.add %305, %309 : tensor<256x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<256x4096xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<256x4096x64xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<256x4096x64xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4096x64xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<256x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x4096xf32>
      %320 = stablehlo.add %318, %319 : tensor<256x4096xf32>
      %321 = stablehlo.rsqrt %320 : tensor<256x4096xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<256x4096xf32>) -> tensor<256x4096x64xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<256x4096x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<256x4096x64xf32>
      %326 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %327 = stablehlo.add %325, %326 : tensor<256x4096x64xf32>
      %328 = stablehlo.transpose %327, dims = [0, 2, 1] : (tensor<256x4096x64xf32>) -> tensor<256x64x4096xf32>
      mhlo.return %328, %323 : tensor<256x64x4096xf32>, tensor<256x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1048576x64xf32>, tensor<64xf32>, tensor<256x4096x64xf32>, tensor<1048576x64xf32>, tensor<64xf32>) -> (tensor<256x64x4096xf32>, tensor<256x4096x64xf32>)
    %150 = stablehlo.tuple %149#0, %149#1 {xla_shape = "(f32[256,64,4096]{2,1,0}, f32[256,4096,64]{2,1,0})"} : tuple<tensor<256x64x4096xf32>, tensor<256x4096x64xf32>>
    %151 = mhlo.bitcast %149#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x64x4096xf32>) -> tensor<256x64x64x64xf32>
    %152 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%151, %arg8, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,160,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<256x64x64x64xf32>, tensor<160x64x3x3xf32>, tensor<160xf32>) -> tuple<tensor<256x160x32x32xf32>, tensor<0xui8>>
    %153 = stablehlo.get_tuple_element %152[0] : (tuple<tensor<256x160x32x32xf32>, tensor<0xui8>>) -> tensor<256x160x32x32xf32>
    %154 = "mhlo.fusion"(%arg11, %arg10, %153) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<256x160x32x32xf32>):
      %303 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x32x32xf32>) -> tensor<256x160x1024xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<256x160x1024xf32>) -> tensor<256x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x32x32xf32>, tensor<f32>) -> tensor<256x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x32x32xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<256x32x32xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x32xf32>) -> tensor<256x1024xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %310 = stablehlo.subtract %304, %309 : tensor<256x1024x160xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<256x1024x160xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %314 = stablehlo.multiply %312, %313 : tensor<256x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %316 = stablehlo.add %314, %315 : tensor<256x1024xf32>
      %317 = stablehlo.rsqrt %316 : tensor<256x1024xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %319 = stablehlo.multiply %310, %318 : tensor<256x1024x160xf32>
      %320 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<256x1024x160xf32>
      %322 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %323 = stablehlo.add %321, %322 : tensor<256x1024x160xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %325 = stablehlo.multiply %324, %313 : tensor<256x1024xf32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %327 = stablehlo.subtract %323, %326 : tensor<256x1024x160xf32>
      %328 = stablehlo.multiply %327, %327 : tensor<256x1024x160xf32>
      %329 = stablehlo.reduce(%328 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %330 = stablehlo.multiply %329, %313 : tensor<256x1024xf32>
      %331 = stablehlo.add %330, %315 : tensor<256x1024xf32>
      %332 = stablehlo.rsqrt %331 : tensor<256x1024xf32>
      %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %334 = stablehlo.multiply %327, %333 : tensor<256x1024x160xf32>
      mhlo.return %334 : tensor<256x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<256x160x32x32xf32>) -> tensor<256x1024x160xf32>
    %155 = mhlo.bitcast %154 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x160xf32>) -> tensor<262144x160xf32>
    %156 = "mhlo.fusion"(%155, %arg104, %arg105, %arg106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<262144x160xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %306 = stablehlo.add %304, %305 : tensor<262144x160xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
      mhlo.return %307 : tensor<262144x160xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
    %157 = "mhlo.fusion"(%156, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<262144x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x5x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x1024x5x32xf32>) -> tensor<256x5x1024x32xf32>
      mhlo.return %306 : tensor<256x5x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160xf32>) -> tensor<256x5x1024x32xf32>
    %158 = mhlo.bitcast %157 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x5x1024x32xf32>) -> tensor<1280x1024x32xf32>
    %159 = "mhlo.fusion"(%154, %arg105, %arg104) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x1024x160xf32>
      %305 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x1024x160xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x1024x160xf32>) -> tensor<256x160x1024xf32>
      mhlo.return %307 : tensor<256x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<256x160x1024xf32>
    %160 = mhlo.bitcast %159 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x160x1024xf32>) -> tensor<256x160x32x32xf32>
    %161 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%160, %arg112, %arg113) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,160,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x160x32x32xf32>, tensor<160x160x2x2xf32>, tensor<160xf32>) -> tuple<tensor<256x160x16x16xf32>, tensor<0xui8>>
    %162 = stablehlo.get_tuple_element %161[0] : (tuple<tensor<256x160x16x16xf32>, tensor<0xui8>>) -> tensor<256x160x16x16xf32>
    %163 = "mhlo.fusion"(%162) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x160x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      mhlo.return %303 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x16x16xf32>) -> tensor<256x16x16xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x16x16xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x16x16xf32>
      mhlo.return %304 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x16xf32>) -> tensor<256x16x16xf32>
    %165 = "mhlo.fusion"(%164, %162) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x16x16xf32>, %arg209: tensor<256x160x16x16xf32>):
      %303 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
      %304 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
      %306 = stablehlo.subtract %303, %305 : tensor<256x160x256xf32>
      %307 = stablehlo.multiply %306, %306 : tensor<256x160x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      mhlo.return %308 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x16xf32>, tensor<256x160x16x16xf32>) -> tensor<256x256xf32>
    %166 = "mhlo.fusion"(%165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<256x256xf32>
      mhlo.return %307 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %167 = "mhlo.fusion"(%166, %164, %162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256x16x16xf32>, %arg210: tensor<256x160x16x16xf32>):
      %303 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
      %304 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
      %306 = stablehlo.subtract %303, %305 : tensor<256x160x256xf32>
      %307 = stablehlo.broadcast_in_dim %arg208, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<256x160x256xf32>
      mhlo.return %308 : tensor<256x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256x16x16xf32>, tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
    %168 = "mhlo.fusion"(%167, %arg114, %arg115, %arg108) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x160x256xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x160x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x160x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x160x256xf32>) -> tensor<256x256x160xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x160xf32>) -> tensor<65536x160xf32>
      %309 = stablehlo.dot_general %308, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %309 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x256xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %169 = "mhlo.fusion"(%168, %arg109) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<256x256x5x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 3, 1] : (tensor<256x256x5x32xf32>) -> tensor<256x5x32x256xf32>
      mhlo.return %306 : tensor<256x5x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>) -> tensor<256x5x32x256xf32>
    %170 = mhlo.bitcast %169 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x5x32x256xf32>) -> tensor<1280x32x256xf32>
    %171 = "mhlo.fusion"(%158, %170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1280x1024x32xf32>, %arg209: tensor<1280x32x256xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x1024x32xf32>, tensor<1280x32x256xf32>) -> tensor<1280x1024x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280x1024x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1280x1024x256xf32>
      mhlo.return %305 : tensor<1280x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x1024x32xf32>, tensor<1280x32x256xf32>) -> tensor<1280x1024x256xf32>
    %172 = "mhlo.fusion"(%171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1280x1024x256xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1280x1024x256xf32>) -> tensor<256x5x1024x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %304 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1280x1024x256xf32>, tensor<f32>) -> tensor<1280x1024xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1280x1024xf32>) -> tensor<256x5x1024xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2] : (tensor<256x5x1024xf32>) -> tensor<256x5x1024x256xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<256x5x1024x256xf32>
      %308 = stablehlo.exponential %307 : tensor<256x5x1024x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x5x1024x256xf32>, tensor<f32>) -> tensor<256x5x1024xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<256x5x1024xf32>) -> tensor<256x5x1024x256xf32>
      %311 = stablehlo.divide %308, %310 : tensor<256x5x1024x256xf32>
      mhlo.return %311 : tensor<256x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x1024x256xf32>) -> tensor<256x5x1024x256xf32>
    %173 = mhlo.bitcast %172 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x5x1024x256xf32>) -> tensor<1280x1024x256xf32>
    %174 = "mhlo.fusion"(%167, %arg114, %arg115, %arg110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x160x256xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x160x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x160x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x160x256xf32>) -> tensor<256x256x160xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x160xf32>) -> tensor<65536x160xf32>
      %309 = stablehlo.dot_general %308, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %309 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x256xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %175 = "mhlo.fusion"(%174, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<256x256x5x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x256x5x32xf32>) -> tensor<256x5x256x32xf32>
      mhlo.return %306 : tensor<256x5x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>) -> tensor<256x5x256x32xf32>
    %176 = mhlo.bitcast %175 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x5x256x32xf32>) -> tensor<1280x256x32xf32>
    %177 = "mhlo.fusion"(%173, %176) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1280x1024x256xf32>, %arg209: tensor<1280x256x32xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x1024x256xf32>, tensor<1280x256x32xf32>) -> tensor<1280x1024x32xf32>
      mhlo.return %303 : tensor<1280x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x1024x256xf32>, tensor<1280x256x32xf32>) -> tensor<1280x1024x32xf32>
    %178 = "mhlo.fusion"(%177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1280x1024x32xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1280x1024x32xf32>) -> tensor<256x5x1024x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<256x5x1024x32xf32>) -> tensor<256x1024x5x32xf32>
      mhlo.return %304 : tensor<256x1024x5x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x1024x32xf32>) -> tensor<256x1024x5x32xf32>
    %179 = mhlo.bitcast %178 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x5x32xf32>) -> tensor<262144x160xf32>
    %180 = "mhlo.fusion"(%179, %arg116) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160x160xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
      mhlo.return %303 : tensor<262144x160xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
    %181 = "mhlo.fusion"(%arg11, %arg10, %153, %180, %arg117) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<256x160x32x32xf32>, %arg211: tensor<262144x160xf32>, %arg212: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.add %303, %arg211 : tensor<262144x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
      %306 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x32x32xf32>) -> tensor<256x160x1024xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x160x1024xf32>) -> tensor<256x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x32x32xf32>, tensor<f32>) -> tensor<256x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x32x32xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x32x32xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x32xf32>) -> tensor<256x1024xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %313 = stablehlo.subtract %307, %312 : tensor<256x1024x160xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<256x1024x160xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %316 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x1024xf32>
      %320 = stablehlo.rsqrt %319 : tensor<256x1024xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %322 = stablehlo.multiply %313, %321 : tensor<256x1024x160xf32>
      %323 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<256x1024x160xf32>
      %325 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %326 = stablehlo.add %324, %325 : tensor<256x1024x160xf32>
      %327 = stablehlo.add %305, %326 : tensor<256x1024x160xf32>
      %328 = stablehlo.reduce(%327 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x1024xf32>
      %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %331 = stablehlo.subtract %327, %330 : tensor<256x1024x160xf32>
      %332 = stablehlo.multiply %331, %331 : tensor<256x1024x160xf32>
      %333 = stablehlo.reduce(%332 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %334 = stablehlo.multiply %333, %316 : tensor<256x1024xf32>
      %335 = stablehlo.add %334, %318 : tensor<256x1024xf32>
      %336 = stablehlo.rsqrt %335 : tensor<256x1024xf32>
      %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %338 = stablehlo.multiply %331, %337 : tensor<256x1024x160xf32>
      mhlo.return %338 : tensor<256x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<256x160x32x32xf32>, tensor<262144x160xf32>, tensor<160xf32>) -> tensor<256x1024x160xf32>
    %182 = mhlo.bitcast %181 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x160xf32>) -> tensor<262144x160xf32>
    %183 = "mhlo.fusion"(%182, %arg118, %arg119, %arg120) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<640x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<262144x160xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %306 = stablehlo.add %304, %305 : tensor<262144x160xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<640x160xf32>) -> tensor<262144x640xf32>
      mhlo.return %307 : tensor<262144x640xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<640x160xf32>) -> tensor<262144x640xf32>
    %184 = "mhlo.fusion"(%183, %arg121) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<262144x640xf32>, %arg209: tensor<640xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<262144x640xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<262144x640xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x640xf32>) -> tensor<256x1024x640xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1] : (tensor<256x1024x640xf32>) -> tensor<256x640x1024xf32>
      mhlo.return %306 : tensor<256x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x640xf32>, tensor<640xf32>) -> tensor<256x640x1024xf32>
    %185 = mhlo.bitcast %184 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x640x1024xf32>) -> tensor<256x640x32x32xf32>
    %186 = stablehlo.custom_call @__cudnn$convForward(%185, %arg122) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,640,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<256x640x32x32xf32>, tensor<640x1x3x3xf32>) -> tuple<tensor<256x640x32x32xf32>, tensor<0xui8>>
    %187 = stablehlo.get_tuple_element %186[0] : (tuple<tensor<256x640x32x32xf32>, tensor<0xui8>>) -> tensor<256x640x32x32xf32>
    %188 = "mhlo.fusion"(%187, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x640x32x32xf32>, %arg209: tensor<640xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x640x32x32xf32>
      %304 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<256x640x32x32xf32>
      %305 = stablehlo.add %arg208, %304 : tensor<256x640x32x32xf32>
      %306 = stablehlo.multiply %303, %305 : tensor<256x640x32x32xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x640x32x32xf32>) -> tensor<256x640x1024xf32>
      %308 = stablehlo.negate %305 : tensor<256x640x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x640x32x32xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x640x32x32xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x640x32x32xf32>) -> tensor<256x640x1024xf32>
      %312 = stablehlo.abs %311 : tensor<256x640x1024xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %314 = stablehlo.compare LT, %312, %313 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %316 = stablehlo.multiply %311, %311 : tensor<256x640x1024xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x640x1024xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x640x1024xf32>
      %320 = stablehlo.multiply %319, %316 : tensor<256x640x1024xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x640x1024xf32>
      %323 = stablehlo.multiply %322, %316 : tensor<256x640x1024xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %325 = stablehlo.add %323, %324 : tensor<256x640x1024xf32>
      %326 = stablehlo.multiply %325, %316 : tensor<256x640x1024xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<256x640x1024xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x640x1024xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %331 = stablehlo.add %329, %330 : tensor<256x640x1024xf32>
      %332 = stablehlo.multiply %331, %316 : tensor<256x640x1024xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %334 = stablehlo.add %332, %333 : tensor<256x640x1024xf32>
      %335 = stablehlo.multiply %311, %334 : tensor<256x640x1024xf32>
      %336 = stablehlo.subtract %313, %335 : tensor<256x640x1024xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %338 = stablehlo.compare LT, %311, %337 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %340 = stablehlo.negate %316 : tensor<256x640x1024xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %342 = stablehlo.compare LT, %340, %341 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
      %343 = stablehlo.exponential %340 : tensor<256x640x1024xf32>
      %344 = stablehlo.divide %313, %312 : tensor<256x640x1024xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<256x640x1024xf32>
      %346 = stablehlo.compare LT, %312, %339 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %348 = stablehlo.divide %313, %316 : tensor<256x640x1024xf32>
      %349 = stablehlo.multiply %347, %348 : tensor<256x640x1024xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %351 = stablehlo.add %349, %350 : tensor<256x640x1024xf32>
      %352 = stablehlo.multiply %351, %348 : tensor<256x640x1024xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %354 = stablehlo.add %352, %353 : tensor<256x640x1024xf32>
      %355 = stablehlo.multiply %354, %348 : tensor<256x640x1024xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %357 = stablehlo.add %355, %356 : tensor<256x640x1024xf32>
      %358 = stablehlo.multiply %357, %348 : tensor<256x640x1024xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %360 = stablehlo.add %358, %359 : tensor<256x640x1024xf32>
      %361 = stablehlo.multiply %360, %348 : tensor<256x640x1024xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %363 = stablehlo.add %361, %362 : tensor<256x640x1024xf32>
      %364 = stablehlo.multiply %363, %348 : tensor<256x640x1024xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %366 = stablehlo.add %364, %365 : tensor<256x640x1024xf32>
      %367 = stablehlo.multiply %366, %348 : tensor<256x640x1024xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %369 = stablehlo.add %367, %368 : tensor<256x640x1024xf32>
      %370 = stablehlo.multiply %369, %348 : tensor<256x640x1024xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %372 = stablehlo.add %370, %371 : tensor<256x640x1024xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %374 = stablehlo.multiply %373, %348 : tensor<256x640x1024xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %376 = stablehlo.add %374, %375 : tensor<256x640x1024xf32>
      %377 = stablehlo.multiply %376, %348 : tensor<256x640x1024xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x640x1024xf32>
      %380 = stablehlo.multiply %379, %348 : tensor<256x640x1024xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %382 = stablehlo.add %380, %381 : tensor<256x640x1024xf32>
      %383 = stablehlo.multiply %382, %348 : tensor<256x640x1024xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %385 = stablehlo.add %383, %384 : tensor<256x640x1024xf32>
      %386 = stablehlo.multiply %385, %348 : tensor<256x640x1024xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %388 = stablehlo.add %386, %387 : tensor<256x640x1024xf32>
      %389 = stablehlo.multiply %388, %348 : tensor<256x640x1024xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %391 = stablehlo.add %389, %390 : tensor<256x640x1024xf32>
      %392 = stablehlo.multiply %391, %348 : tensor<256x640x1024xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %394 = stablehlo.add %392, %393 : tensor<256x640x1024xf32>
      %395 = stablehlo.select %346, %372, %394 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
      %396 = stablehlo.multiply %345, %395 : tensor<256x640x1024xf32>
      %397 = stablehlo.select %342, %337, %396 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
      %398 = stablehlo.subtract %339, %397 : tensor<256x640x1024xf32>
      %399 = stablehlo.select %338, %398, %397 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
      %400 = stablehlo.select %314, %336, %399 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
      %401 = stablehlo.multiply %307, %400 : tensor<256x640x1024xf32>
      mhlo.return %401 : tensor<256x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<256x640x32x32xf32>, tensor<640xf32>) -> tensor<256x640x1024xf32>
    %189 = "mhlo.fusion"(%188, %arg124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x640x1024xf32>, %arg209: tensor<160x640xf32>):
      %303 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<256x640x1024xf32>) -> tensor<256x1024x640xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x640xf32>) -> tensor<262144x640xf32>
      %305 = stablehlo.dot_general %304, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x640xf32>, tensor<160x640xf32>) -> tensor<262144x160xf32>
      mhlo.return %305 : tensor<262144x160xf32>
    }) {output_operand_aliasing = []} : (tensor<256x640x1024xf32>, tensor<160x640xf32>) -> tensor<262144x160xf32>
    %190 = "mhlo.fusion"(%189, %arg125, %arg11, %arg10, %153, %180, %arg117) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160xf32>, %arg212: tensor<256x160x32x32xf32>, %arg213: tensor<262144x160xf32>, %arg214: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<262144x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
      %306 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %307 = stablehlo.add %306, %arg213 : tensor<262144x160xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
      %309 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x32x32xf32>) -> tensor<256x160x1024xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1] : (tensor<256x160x1024xf32>) -> tensor<256x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x32x32xf32>, tensor<f32>) -> tensor<256x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x32x32xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<256x32x32xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x32x32xf32>) -> tensor<256x1024xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %316 = stablehlo.subtract %310, %315 : tensor<256x1024x160xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<256x1024x160xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<256x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x1024xf32>
      %323 = stablehlo.rsqrt %322 : tensor<256x1024xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %325 = stablehlo.multiply %316, %324 : tensor<256x1024x160xf32>
      %326 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %327 = stablehlo.multiply %325, %326 : tensor<256x1024x160xf32>
      %328 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %329 = stablehlo.add %327, %328 : tensor<256x1024x160xf32>
      %330 = stablehlo.add %308, %329 : tensor<256x1024x160xf32>
      %331 = stablehlo.add %305, %330 : tensor<256x1024x160xf32>
      mhlo.return %331 : tensor<256x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<256x160x32x32xf32>, tensor<262144x160xf32>, tensor<160xf32>) -> tensor<256x1024x160xf32>
    %191 = "mhlo.fusion"(%190) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x1024x160xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<256x1024xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %307 = stablehlo.subtract %arg208, %306 : tensor<256x1024x160xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<256x1024x160xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %310 = stablehlo.multiply %309, %304 : tensor<256x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %312 = stablehlo.add %310, %311 : tensor<256x1024xf32>
      %313 = stablehlo.rsqrt %312 : tensor<256x1024xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %315 = stablehlo.multiply %307, %314 : tensor<256x1024x160xf32>
      mhlo.return %315 : tensor<256x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x160xf32>) -> tensor<256x1024x160xf32>
    %192 = mhlo.bitcast %191 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x160xf32>) -> tensor<262144x160xf32>
    %193 = "mhlo.fusion"(%192, %arg126, %arg127, %arg128) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<262144x160xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %306 = stablehlo.add %304, %305 : tensor<262144x160xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
      mhlo.return %307 : tensor<262144x160xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
    %194 = "mhlo.fusion"(%193, %arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<262144x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x5x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x1024x5x32xf32>) -> tensor<256x5x1024x32xf32>
      mhlo.return %306 : tensor<256x5x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160xf32>) -> tensor<256x5x1024x32xf32>
    %195 = mhlo.bitcast %194 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x5x1024x32xf32>) -> tensor<1280x1024x32xf32>
    %196 = "mhlo.fusion"(%191, %arg127, %arg126) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x1024x160xf32>
      %305 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x1024x160xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x1024x160xf32>) -> tensor<256x160x1024xf32>
      mhlo.return %307 : tensor<256x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<256x160x1024xf32>
    %197 = mhlo.bitcast %196 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x160x1024xf32>) -> tensor<256x160x32x32xf32>
    %198 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%197, %arg134, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,160,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x160x32x32xf32>, tensor<160x160x2x2xf32>, tensor<160xf32>) -> tuple<tensor<256x160x16x16xf32>, tensor<0xui8>>
    %199 = stablehlo.get_tuple_element %198[0] : (tuple<tensor<256x160x16x16xf32>, tensor<0xui8>>) -> tensor<256x160x16x16xf32>
    %200 = "mhlo.fusion"(%199) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x160x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      mhlo.return %303 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x16x16xf32>) -> tensor<256x16x16xf32>
    %201 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x16x16xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x16x16xf32>
      mhlo.return %304 : tensor<256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x16xf32>) -> tensor<256x16x16xf32>
    %202 = "mhlo.fusion"(%201, %199) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<256x16x16xf32>, %arg209: tensor<256x160x16x16xf32>):
      %303 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
      %304 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
      %306 = stablehlo.subtract %303, %305 : tensor<256x160x256xf32>
      %307 = stablehlo.multiply %306, %306 : tensor<256x160x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x160x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      mhlo.return %308 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16x16xf32>, tensor<256x160x16x16xf32>) -> tensor<256x256xf32>
    %203 = "mhlo.fusion"(%202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x256xf32>
      %307 = stablehlo.rsqrt %306 : tensor<256x256xf32>
      mhlo.return %307 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %204 = "mhlo.fusion"(%203, %201, %199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256x16x16xf32>, %arg210: tensor<256x160x16x16xf32>):
      %303 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
      %304 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
      %306 = stablehlo.subtract %303, %305 : tensor<256x160x256xf32>
      %307 = stablehlo.broadcast_in_dim %arg208, dims = [0, 2] : (tensor<256x256xf32>) -> tensor<256x160x256xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<256x160x256xf32>
      mhlo.return %308 : tensor<256x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256x16x16xf32>, tensor<256x160x16x16xf32>) -> tensor<256x160x256xf32>
    %205 = "mhlo.fusion"(%204, %arg136, %arg137, %arg130) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x160x256xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x160x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x160x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x160x256xf32>) -> tensor<256x256x160xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x160xf32>) -> tensor<65536x160xf32>
      %309 = stablehlo.dot_general %308, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %309 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x256xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %206 = "mhlo.fusion"(%205, %arg131) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<256x256x5x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 3, 1] : (tensor<256x256x5x32xf32>) -> tensor<256x5x32x256xf32>
      mhlo.return %306 : tensor<256x5x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>) -> tensor<256x5x32x256xf32>
    %207 = mhlo.bitcast %206 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x5x32x256xf32>) -> tensor<1280x32x256xf32>
    %208 = "mhlo.fusion"(%195, %207) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1280x1024x32xf32>, %arg209: tensor<1280x32x256xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x1024x32xf32>, tensor<1280x32x256xf32>) -> tensor<1280x1024x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280x1024x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1280x1024x256xf32>
      mhlo.return %305 : tensor<1280x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x1024x32xf32>, tensor<1280x32x256xf32>) -> tensor<1280x1024x256xf32>
    %209 = "mhlo.fusion"(%208) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1280x1024x256xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1280x1024x256xf32>) -> tensor<256x5x1024x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %304 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1280x1024x256xf32>, tensor<f32>) -> tensor<1280x1024xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1280x1024xf32>) -> tensor<256x5x1024xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2] : (tensor<256x5x1024xf32>) -> tensor<256x5x1024x256xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<256x5x1024x256xf32>
      %308 = stablehlo.exponential %307 : tensor<256x5x1024x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x5x1024x256xf32>, tensor<f32>) -> tensor<256x5x1024xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<256x5x1024xf32>) -> tensor<256x5x1024x256xf32>
      %311 = stablehlo.divide %308, %310 : tensor<256x5x1024x256xf32>
      mhlo.return %311 : tensor<256x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x1024x256xf32>) -> tensor<256x5x1024x256xf32>
    %210 = mhlo.bitcast %209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x5x1024x256xf32>) -> tensor<1280x1024x256xf32>
    %211 = "mhlo.fusion"(%204, %arg136, %arg137, %arg132) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x160x256xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x160x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<256x160x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x160x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x160x256xf32>) -> tensor<256x256x160xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x160xf32>) -> tensor<65536x160xf32>
      %309 = stablehlo.dot_general %308, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %309 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x256xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %212 = "mhlo.fusion"(%211, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<256x256x5x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x256x5x32xf32>) -> tensor<256x5x256x32xf32>
      mhlo.return %306 : tensor<256x5x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>) -> tensor<256x5x256x32xf32>
    %213 = mhlo.bitcast %212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x5x256x32xf32>) -> tensor<1280x256x32xf32>
    %214 = "mhlo.fusion"(%210, %213) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1280x1024x256xf32>, %arg209: tensor<1280x256x32xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1280x1024x256xf32>, tensor<1280x256x32xf32>) -> tensor<1280x1024x32xf32>
      mhlo.return %303 : tensor<1280x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x1024x256xf32>, tensor<1280x256x32xf32>) -> tensor<1280x1024x32xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1280x1024x32xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1280x1024x32xf32>) -> tensor<256x5x1024x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<256x5x1024x32xf32>) -> tensor<256x1024x5x32xf32>
      mhlo.return %304 : tensor<256x1024x5x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1280x1024x32xf32>) -> tensor<256x1024x5x32xf32>
    %216 = mhlo.bitcast %215 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x5x32xf32>) -> tensor<262144x160xf32>
    %217 = "mhlo.fusion"(%216, %arg138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160x160xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
      mhlo.return %303 : tensor<262144x160xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160x160xf32>) -> tensor<262144x160xf32>
    %218 = "mhlo.fusion"(%190, %217, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x1024x160xf32>, %arg209: tensor<262144x160xf32>, %arg210: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.add %303, %arg209 : tensor<262144x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
      %306 = stablehlo.add %305, %arg208 : tensor<256x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<256x1024xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %311 = stablehlo.subtract %306, %310 : tensor<256x1024x160xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<256x1024x160xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<256x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %316 = stablehlo.add %314, %315 : tensor<256x1024xf32>
      %317 = stablehlo.rsqrt %316 : tensor<256x1024xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<256x1024x160xf32>
      mhlo.return %319 : tensor<256x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x160xf32>, tensor<262144x160xf32>, tensor<160xf32>) -> tensor<256x1024x160xf32>
    %219 = mhlo.bitcast %218 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x160xf32>) -> tensor<262144x160xf32>
    %220 = "mhlo.fusion"(%219, %arg140, %arg141, %arg142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<262144x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<640x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<262144x160xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %306 = stablehlo.add %304, %305 : tensor<262144x160xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x160xf32>, tensor<640x160xf32>) -> tensor<262144x640xf32>
      mhlo.return %307 : tensor<262144x640xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<640x160xf32>) -> tensor<262144x640xf32>
    %221 = "mhlo.fusion"(%220, %arg143) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<262144x640xf32>, %arg209: tensor<640xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<262144x640xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<262144x640xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x640xf32>) -> tensor<256x1024x640xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1] : (tensor<256x1024x640xf32>) -> tensor<256x640x1024xf32>
      mhlo.return %306 : tensor<256x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x640xf32>, tensor<640xf32>) -> tensor<256x640x1024xf32>
    %222 = mhlo.bitcast %221 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x640x1024xf32>) -> tensor<256x640x32x32xf32>
    %223 = stablehlo.custom_call @__cudnn$convForward(%222, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,640,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<256x640x32x32xf32>, tensor<640x1x3x3xf32>) -> tuple<tensor<256x640x32x32xf32>, tensor<0xui8>>
    %224 = stablehlo.get_tuple_element %223[0] : (tuple<tensor<256x640x32x32xf32>, tensor<0xui8>>) -> tensor<256x640x32x32xf32>
    %225 = "mhlo.fusion"(%224, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x640x32x32xf32>, %arg209: tensor<640xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x640x32x32xf32>
      %304 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<256x640x32x32xf32>
      %305 = stablehlo.add %arg208, %304 : tensor<256x640x32x32xf32>
      %306 = stablehlo.multiply %303, %305 : tensor<256x640x32x32xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x640x32x32xf32>) -> tensor<256x640x1024xf32>
      %308 = stablehlo.negate %305 : tensor<256x640x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x640x32x32xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x640x32x32xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x640x32x32xf32>) -> tensor<256x640x1024xf32>
      %312 = stablehlo.abs %311 : tensor<256x640x1024xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %314 = stablehlo.compare LT, %312, %313 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %316 = stablehlo.multiply %311, %311 : tensor<256x640x1024xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x640x1024xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x640x1024xf32>
      %320 = stablehlo.multiply %319, %316 : tensor<256x640x1024xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x640x1024xf32>
      %323 = stablehlo.multiply %322, %316 : tensor<256x640x1024xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %325 = stablehlo.add %323, %324 : tensor<256x640x1024xf32>
      %326 = stablehlo.multiply %325, %316 : tensor<256x640x1024xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %328 = stablehlo.add %326, %327 : tensor<256x640x1024xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x640x1024xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %331 = stablehlo.add %329, %330 : tensor<256x640x1024xf32>
      %332 = stablehlo.multiply %331, %316 : tensor<256x640x1024xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %334 = stablehlo.add %332, %333 : tensor<256x640x1024xf32>
      %335 = stablehlo.multiply %311, %334 : tensor<256x640x1024xf32>
      %336 = stablehlo.subtract %313, %335 : tensor<256x640x1024xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %338 = stablehlo.compare LT, %311, %337 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %340 = stablehlo.negate %316 : tensor<256x640x1024xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %342 = stablehlo.compare LT, %340, %341 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
      %343 = stablehlo.exponential %340 : tensor<256x640x1024xf32>
      %344 = stablehlo.divide %313, %312 : tensor<256x640x1024xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<256x640x1024xf32>
      %346 = stablehlo.compare LT, %312, %339 : (tensor<256x640x1024xf32>, tensor<256x640x1024xf32>) -> tensor<256x640x1024xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %348 = stablehlo.divide %313, %316 : tensor<256x640x1024xf32>
      %349 = stablehlo.multiply %347, %348 : tensor<256x640x1024xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %351 = stablehlo.add %349, %350 : tensor<256x640x1024xf32>
      %352 = stablehlo.multiply %351, %348 : tensor<256x640x1024xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %354 = stablehlo.add %352, %353 : tensor<256x640x1024xf32>
      %355 = stablehlo.multiply %354, %348 : tensor<256x640x1024xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %357 = stablehlo.add %355, %356 : tensor<256x640x1024xf32>
      %358 = stablehlo.multiply %357, %348 : tensor<256x640x1024xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %360 = stablehlo.add %358, %359 : tensor<256x640x1024xf32>
      %361 = stablehlo.multiply %360, %348 : tensor<256x640x1024xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %363 = stablehlo.add %361, %362 : tensor<256x640x1024xf32>
      %364 = stablehlo.multiply %363, %348 : tensor<256x640x1024xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %366 = stablehlo.add %364, %365 : tensor<256x640x1024xf32>
      %367 = stablehlo.multiply %366, %348 : tensor<256x640x1024xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %369 = stablehlo.add %367, %368 : tensor<256x640x1024xf32>
      %370 = stablehlo.multiply %369, %348 : tensor<256x640x1024xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %372 = stablehlo.add %370, %371 : tensor<256x640x1024xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %374 = stablehlo.multiply %373, %348 : tensor<256x640x1024xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %376 = stablehlo.add %374, %375 : tensor<256x640x1024xf32>
      %377 = stablehlo.multiply %376, %348 : tensor<256x640x1024xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x640x1024xf32>
      %380 = stablehlo.multiply %379, %348 : tensor<256x640x1024xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %382 = stablehlo.add %380, %381 : tensor<256x640x1024xf32>
      %383 = stablehlo.multiply %382, %348 : tensor<256x640x1024xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %385 = stablehlo.add %383, %384 : tensor<256x640x1024xf32>
      %386 = stablehlo.multiply %385, %348 : tensor<256x640x1024xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %388 = stablehlo.add %386, %387 : tensor<256x640x1024xf32>
      %389 = stablehlo.multiply %388, %348 : tensor<256x640x1024xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %391 = stablehlo.add %389, %390 : tensor<256x640x1024xf32>
      %392 = stablehlo.multiply %391, %348 : tensor<256x640x1024xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x640x1024xf32>
      %394 = stablehlo.add %392, %393 : tensor<256x640x1024xf32>
      %395 = stablehlo.select %346, %372, %394 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
      %396 = stablehlo.multiply %345, %395 : tensor<256x640x1024xf32>
      %397 = stablehlo.select %342, %337, %396 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
      %398 = stablehlo.subtract %339, %397 : tensor<256x640x1024xf32>
      %399 = stablehlo.select %338, %398, %397 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
      %400 = stablehlo.select %314, %336, %399 : tensor<256x640x1024xi1>, tensor<256x640x1024xf32>
      %401 = stablehlo.multiply %307, %400 : tensor<256x640x1024xf32>
      mhlo.return %401 : tensor<256x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<256x640x32x32xf32>, tensor<640xf32>) -> tensor<256x640x1024xf32>
    %226 = "mhlo.fusion"(%225, %arg146) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x640x1024xf32>, %arg209: tensor<160x640xf32>):
      %303 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<256x640x1024xf32>) -> tensor<256x1024x640xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x640xf32>) -> tensor<262144x640xf32>
      %305 = stablehlo.dot_general %304, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x640xf32>, tensor<160x640xf32>) -> tensor<262144x160xf32>
      mhlo.return %305 : tensor<262144x160xf32>
    }) {output_operand_aliasing = []} : (tensor<256x640x1024xf32>, tensor<160x640xf32>) -> tensor<262144x160xf32>
    %227:2 = "mhlo.fusion"(%arg189, %arg188, %226, %arg147, %190, %217, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<262144x160xf32>, %arg211: tensor<160xf32>, %arg212: tensor<256x1024x160xf32>, %arg213: tensor<262144x160xf32>, %arg214: tensor<160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %304 = stablehlo.add %303, %arg210 : tensor<262144x160xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
      %306 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<160xf32>) -> tensor<262144x160xf32>
      %307 = stablehlo.add %306, %arg213 : tensor<262144x160xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x160xf32>) -> tensor<256x1024x160xf32>
      %309 = stablehlo.add %308, %arg212 : tensor<256x1024x160xf32>
      %310 = stablehlo.add %305, %309 : tensor<256x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<256x1024xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<256x1024x160xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<256x1024x160xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x160xf32>, tensor<f32>) -> tensor<256x1024xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<256x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<256x1024xf32>
      %321 = stablehlo.rsqrt %320 : tensor<256x1024xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<256x1024xf32>) -> tensor<256x1024x160xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<256x1024x160xf32>
      %324 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<256x1024x160xf32>
      %326 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %327 = stablehlo.add %325, %326 : tensor<256x1024x160xf32>
      %328 = stablehlo.transpose %327, dims = [0, 2, 1] : (tensor<256x1024x160xf32>) -> tensor<256x160x1024xf32>
      mhlo.return %328, %323 : tensor<256x160x1024xf32>, tensor<256x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<262144x160xf32>, tensor<160xf32>, tensor<256x1024x160xf32>, tensor<262144x160xf32>, tensor<160xf32>) -> (tensor<256x160x1024xf32>, tensor<256x1024x160xf32>)
    %228 = stablehlo.tuple %227#0, %227#1 {xla_shape = "(f32[256,160,1024]{2,1,0}, f32[256,1024,160]{2,1,0})"} : tuple<tensor<256x160x1024xf32>, tensor<256x1024x160xf32>>
    %229 = mhlo.bitcast %227#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x160x1024xf32>) -> tensor<256x160x32x32xf32>
    %230 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%229, %arg12, %arg13) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,256,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x160x32x32xf32>, tensor<256x160x3x3xf32>, tensor<256xf32>) -> tuple<tensor<256x256x16x16xf32>, tensor<0xui8>>
    %231 = stablehlo.get_tuple_element %230[0] : (tuple<tensor<256x256x16x16xf32>, tensor<0xui8>>) -> tensor<256x256x16x16xf32>
    %232 = "mhlo.fusion"(%arg15, %arg14, %231) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256x256x16x16xf32>):
      %303 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x16x16xf32>) -> tensor<256x256x256xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x256x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<256x16x16xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %310 = stablehlo.subtract %304, %309 : tensor<256x256x256xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<256x256x256xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %314 = stablehlo.multiply %312, %313 : tensor<256x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %316 = stablehlo.add %314, %315 : tensor<256x256xf32>
      %317 = stablehlo.rsqrt %316 : tensor<256x256xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %319 = stablehlo.multiply %310, %318 : tensor<256x256x256xf32>
      %320 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<256x256x256xf32>
      %322 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %323 = stablehlo.add %321, %322 : tensor<256x256x256xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %325 = stablehlo.multiply %324, %313 : tensor<256x256xf32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %327 = stablehlo.subtract %323, %326 : tensor<256x256x256xf32>
      %328 = stablehlo.multiply %327, %327 : tensor<256x256x256xf32>
      %329 = stablehlo.reduce(%328 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %330 = stablehlo.multiply %329, %313 : tensor<256x256xf32>
      %331 = stablehlo.add %330, %315 : tensor<256x256xf32>
      %332 = stablehlo.rsqrt %331 : tensor<256x256xf32>
      %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %334 = stablehlo.multiply %327, %333 : tensor<256x256x256xf32>
      mhlo.return %334 : tensor<256x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x256x16x16xf32>) -> tensor<256x256x256xf32>
    %233 = mhlo.bitcast %232 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x256xf32>) -> tensor<65536x256xf32>
    %234 = "mhlo.fusion"(%233, %arg148, %arg149, %arg150) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<65536x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<65536x256xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %307 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    %235 = "mhlo.fusion"(%234, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x256x8x32xf32>) -> tensor<256x8x256x32xf32>
      mhlo.return %306 : tensor<256x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x8x256x32xf32>
    %236 = mhlo.bitcast %235 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x8x256x32xf32>) -> tensor<2048x256x32xf32>
    %237 = "mhlo.fusion"(%233, %arg148, %arg149, %arg152) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<65536x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<65536x256xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %307 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    %238 = "mhlo.fusion"(%237, %arg153) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 3, 1] : (tensor<256x256x8x32xf32>) -> tensor<256x8x32x256xf32>
      mhlo.return %306 : tensor<256x8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x8x32x256xf32>
    %239 = mhlo.bitcast %238 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x8x32x256xf32>) -> tensor<2048x32x256xf32>
    %240 = "mhlo.fusion"(%236, %239) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2048x256x32xf32>, %arg209: tensor<2048x32x256xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256x32xf32>, tensor<2048x32x256xf32>) -> tensor<2048x256x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048x256x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<2048x256x256xf32>
      mhlo.return %305 : tensor<2048x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256x32xf32>, tensor<2048x32x256xf32>) -> tensor<2048x256x256xf32>
    %241 = "mhlo.fusion"(%240) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2048x256x256xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2048x256x256xf32>) -> tensor<256x8x256x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %304 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<2048x256x256xf32>, tensor<f32>) -> tensor<2048x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<256x8x256xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2] : (tensor<256x8x256xf32>) -> tensor<256x8x256x256xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<256x8x256x256xf32>
      %308 = stablehlo.exponential %307 : tensor<256x8x256x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x8x256x256xf32>, tensor<f32>) -> tensor<256x8x256xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<256x8x256xf32>) -> tensor<256x8x256x256xf32>
      %311 = stablehlo.divide %308, %310 : tensor<256x8x256x256xf32>
      mhlo.return %311 : tensor<256x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256x256xf32>) -> tensor<256x8x256x256xf32>
    %242 = mhlo.bitcast %241 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x8x256x256xf32>) -> tensor<2048x256x256xf32>
    %243 = "mhlo.fusion"(%233, %arg148, %arg149, %arg154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<65536x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<65536x256xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %307 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    %244 = "mhlo.fusion"(%243, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x256x8x32xf32>) -> tensor<256x8x256x32xf32>
      mhlo.return %306 : tensor<256x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x8x256x32xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x8x256x32xf32>) -> tensor<2048x256x32xf32>
    %246 = "mhlo.fusion"(%242, %245) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2048x256x256xf32>, %arg209: tensor<2048x256x32xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256x256xf32>, tensor<2048x256x32xf32>) -> tensor<2048x256x32xf32>
      mhlo.return %303 : tensor<2048x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256x256xf32>, tensor<2048x256x32xf32>) -> tensor<2048x256x32xf32>
    %247 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256x32xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2048x256x32xf32>) -> tensor<256x8x256x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<256x8x256x32xf32>) -> tensor<256x256x8x32xf32>
      mhlo.return %304 : tensor<256x256x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256x32xf32>) -> tensor<256x256x8x32xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x8x32xf32>) -> tensor<65536x256xf32>
    %249 = "mhlo.fusion"(%248, %arg156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256x256xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %303 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    %250 = "mhlo.fusion"(%arg15, %arg14, %231, %249, %arg157) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256x256x16x16xf32>, %arg211: tensor<65536x256xf32>, %arg212: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg211 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
      %306 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x16x16xf32>) -> tensor<256x256x256xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x256x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x16x16xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %313 = stablehlo.subtract %307, %312 : tensor<256x256x256xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<256x256x256xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %316 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x256xf32>
      %320 = stablehlo.rsqrt %319 : tensor<256x256xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %322 = stablehlo.multiply %313, %321 : tensor<256x256x256xf32>
      %323 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<256x256x256xf32>
      %325 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %326 = stablehlo.add %324, %325 : tensor<256x256x256xf32>
      %327 = stablehlo.add %305, %326 : tensor<256x256x256xf32>
      %328 = stablehlo.reduce(%327 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x256xf32>
      %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %331 = stablehlo.subtract %327, %330 : tensor<256x256x256xf32>
      %332 = stablehlo.multiply %331, %331 : tensor<256x256x256xf32>
      %333 = stablehlo.reduce(%332 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %334 = stablehlo.multiply %333, %316 : tensor<256x256xf32>
      %335 = stablehlo.add %334, %318 : tensor<256x256xf32>
      %336 = stablehlo.rsqrt %335 : tensor<256x256xf32>
      %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %338 = stablehlo.multiply %331, %337 : tensor<256x256x256xf32>
      mhlo.return %338 : tensor<256x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x256x16x16xf32>, tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x256x256xf32>
    %251 = mhlo.bitcast %250 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x256xf32>) -> tensor<65536x256xf32>
    %252 = "mhlo.fusion"(%251, %arg158, %arg159, %arg160) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1024x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<65536x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<65536x256xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<1024x256xf32>) -> tensor<65536x1024xf32>
      mhlo.return %307 : tensor<65536x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256xf32>) -> tensor<65536x1024xf32>
    %253 = "mhlo.fusion"(%252, %arg161) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x1024xf32>, %arg209: tensor<1024xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<65536x1024xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x1024xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x1024xf32>) -> tensor<256x256x1024xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1] : (tensor<256x256x1024xf32>) -> tensor<256x1024x256xf32>
      mhlo.return %306 : tensor<256x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x1024xf32>, tensor<1024xf32>) -> tensor<256x1024x256xf32>
    %254 = mhlo.bitcast %253 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x256xf32>) -> tensor<256x1024x16x16xf32>
    %255 = stablehlo.custom_call @__cudnn$convForward(%254, %arg162) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,1024,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x1024x16x16xf32>, tensor<1024x1x3x3xf32>) -> tuple<tensor<256x1024x16x16xf32>, tensor<0xui8>>
    %256 = stablehlo.get_tuple_element %255[0] : (tuple<tensor<256x1024x16x16xf32>, tensor<0xui8>>) -> tensor<256x1024x16x16xf32>
    %257 = "mhlo.fusion"(%256, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024x16x16xf32>, %arg209: tensor<1024xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x1024x16x16xf32>
      %304 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x16x16xf32>
      %305 = stablehlo.add %arg208, %304 : tensor<256x1024x16x16xf32>
      %306 = stablehlo.multiply %303, %305 : tensor<256x1024x16x16xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x16x16xf32>) -> tensor<256x1024x256xf32>
      %308 = stablehlo.negate %305 : tensor<256x1024x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024x16x16xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x1024x16x16xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x16x16xf32>) -> tensor<256x1024x256xf32>
      %312 = stablehlo.abs %311 : tensor<256x1024x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %314 = stablehlo.compare LT, %312, %313 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %316 = stablehlo.multiply %311, %311 : tensor<256x1024x256xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x1024x256xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x1024x256xf32>
      %320 = stablehlo.multiply %319, %316 : tensor<256x1024x256xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x1024x256xf32>
      %323 = stablehlo.multiply %322, %316 : tensor<256x1024x256xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %325 = stablehlo.add %323, %324 : tensor<256x1024x256xf32>
      %326 = stablehlo.multiply %325, %316 : tensor<256x1024x256xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %328 = stablehlo.add %326, %327 : tensor<256x1024x256xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x1024x256xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %331 = stablehlo.add %329, %330 : tensor<256x1024x256xf32>
      %332 = stablehlo.multiply %331, %316 : tensor<256x1024x256xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %334 = stablehlo.add %332, %333 : tensor<256x1024x256xf32>
      %335 = stablehlo.multiply %311, %334 : tensor<256x1024x256xf32>
      %336 = stablehlo.subtract %313, %335 : tensor<256x1024x256xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %338 = stablehlo.compare LT, %311, %337 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %340 = stablehlo.negate %316 : tensor<256x1024x256xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %342 = stablehlo.compare LT, %340, %341 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
      %343 = stablehlo.exponential %340 : tensor<256x1024x256xf32>
      %344 = stablehlo.divide %313, %312 : tensor<256x1024x256xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<256x1024x256xf32>
      %346 = stablehlo.compare LT, %312, %339 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %348 = stablehlo.divide %313, %316 : tensor<256x1024x256xf32>
      %349 = stablehlo.multiply %347, %348 : tensor<256x1024x256xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %351 = stablehlo.add %349, %350 : tensor<256x1024x256xf32>
      %352 = stablehlo.multiply %351, %348 : tensor<256x1024x256xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %354 = stablehlo.add %352, %353 : tensor<256x1024x256xf32>
      %355 = stablehlo.multiply %354, %348 : tensor<256x1024x256xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %357 = stablehlo.add %355, %356 : tensor<256x1024x256xf32>
      %358 = stablehlo.multiply %357, %348 : tensor<256x1024x256xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %360 = stablehlo.add %358, %359 : tensor<256x1024x256xf32>
      %361 = stablehlo.multiply %360, %348 : tensor<256x1024x256xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %363 = stablehlo.add %361, %362 : tensor<256x1024x256xf32>
      %364 = stablehlo.multiply %363, %348 : tensor<256x1024x256xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %366 = stablehlo.add %364, %365 : tensor<256x1024x256xf32>
      %367 = stablehlo.multiply %366, %348 : tensor<256x1024x256xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %369 = stablehlo.add %367, %368 : tensor<256x1024x256xf32>
      %370 = stablehlo.multiply %369, %348 : tensor<256x1024x256xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %372 = stablehlo.add %370, %371 : tensor<256x1024x256xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %374 = stablehlo.multiply %373, %348 : tensor<256x1024x256xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %376 = stablehlo.add %374, %375 : tensor<256x1024x256xf32>
      %377 = stablehlo.multiply %376, %348 : tensor<256x1024x256xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x1024x256xf32>
      %380 = stablehlo.multiply %379, %348 : tensor<256x1024x256xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %382 = stablehlo.add %380, %381 : tensor<256x1024x256xf32>
      %383 = stablehlo.multiply %382, %348 : tensor<256x1024x256xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %385 = stablehlo.add %383, %384 : tensor<256x1024x256xf32>
      %386 = stablehlo.multiply %385, %348 : tensor<256x1024x256xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %388 = stablehlo.add %386, %387 : tensor<256x1024x256xf32>
      %389 = stablehlo.multiply %388, %348 : tensor<256x1024x256xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %391 = stablehlo.add %389, %390 : tensor<256x1024x256xf32>
      %392 = stablehlo.multiply %391, %348 : tensor<256x1024x256xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %394 = stablehlo.add %392, %393 : tensor<256x1024x256xf32>
      %395 = stablehlo.select %346, %372, %394 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
      %396 = stablehlo.multiply %345, %395 : tensor<256x1024x256xf32>
      %397 = stablehlo.select %342, %337, %396 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
      %398 = stablehlo.subtract %339, %397 : tensor<256x1024x256xf32>
      %399 = stablehlo.select %338, %398, %397 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
      %400 = stablehlo.select %314, %336, %399 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
      %401 = stablehlo.multiply %307, %400 : tensor<256x1024x256xf32>
      mhlo.return %401 : tensor<256x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x16x16xf32>, tensor<1024xf32>) -> tensor<256x1024x256xf32>
    %258 = "mhlo.fusion"(%257, %arg164) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x1024x256xf32>, %arg209: tensor<256x1024xf32>):
      %303 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<256x1024x256xf32>) -> tensor<256x256x1024xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x1024xf32>) -> tensor<65536x1024xf32>
      %305 = stablehlo.dot_general %304, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x1024xf32>, tensor<256x1024xf32>) -> tensor<65536x256xf32>
      mhlo.return %305 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x256xf32>, tensor<256x1024xf32>) -> tensor<65536x256xf32>
    %259 = "mhlo.fusion"(%258, %arg165, %arg15, %arg14, %231, %249, %arg157) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<256x256x16x16xf32>, %arg213: tensor<65536x256xf32>, %arg214: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
      %306 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %307 = stablehlo.add %306, %arg213 : tensor<65536x256xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
      %309 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x16x16xf32>) -> tensor<256x256x256xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1] : (tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x256x16x16xf32>, tensor<f32>) -> tensor<256x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x16x16xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<256x16x16xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x16x16xf32>) -> tensor<256x256xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %316 = stablehlo.subtract %310, %315 : tensor<256x256x256xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<256x256x256xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<256x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x256xf32>
      %323 = stablehlo.rsqrt %322 : tensor<256x256xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %325 = stablehlo.multiply %316, %324 : tensor<256x256x256xf32>
      %326 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %327 = stablehlo.multiply %325, %326 : tensor<256x256x256xf32>
      %328 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %329 = stablehlo.add %327, %328 : tensor<256x256x256xf32>
      %330 = stablehlo.add %308, %329 : tensor<256x256x256xf32>
      %331 = stablehlo.add %305, %330 : tensor<256x256x256xf32>
      mhlo.return %331 : tensor<256x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x16x16xf32>, tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x256x256xf32>
    %260 = "mhlo.fusion"(%259) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<256x256xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %307 = stablehlo.subtract %arg208, %306 : tensor<256x256x256xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<256x256x256xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %310 = stablehlo.multiply %309, %304 : tensor<256x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %312 = stablehlo.add %310, %311 : tensor<256x256xf32>
      %313 = stablehlo.rsqrt %312 : tensor<256x256xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %315 = stablehlo.multiply %307, %314 : tensor<256x256x256xf32>
      mhlo.return %315 : tensor<256x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
    %261 = mhlo.bitcast %260 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x256xf32>) -> tensor<65536x256xf32>
    %262 = "mhlo.fusion"(%261, %arg166, %arg167, %arg168) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<65536x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<65536x256xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %307 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    %263 = "mhlo.fusion"(%262, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x256x8x32xf32>) -> tensor<256x8x256x32xf32>
      mhlo.return %306 : tensor<256x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x8x256x32xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x8x256x32xf32>) -> tensor<2048x256x32xf32>
    %265 = "mhlo.fusion"(%261, %arg166, %arg167, %arg170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<65536x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<65536x256xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %307 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    %266 = "mhlo.fusion"(%265, %arg171) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 3, 1] : (tensor<256x256x8x32xf32>) -> tensor<256x8x32x256xf32>
      mhlo.return %306 : tensor<256x8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x8x32x256xf32>
    %267 = mhlo.bitcast %266 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x8x32x256xf32>) -> tensor<2048x32x256xf32>
    %268 = "mhlo.fusion"(%264, %267) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2048x256x32xf32>, %arg209: tensor<2048x32x256xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256x32xf32>, tensor<2048x32x256xf32>) -> tensor<2048x256x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048x256x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<2048x256x256xf32>
      mhlo.return %305 : tensor<2048x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256x32xf32>, tensor<2048x32x256xf32>) -> tensor<2048x256x256xf32>
    %269 = "mhlo.fusion"(%268) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2048x256x256xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2048x256x256xf32>) -> tensor<256x8x256x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %304 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<2048x256x256xf32>, tensor<f32>) -> tensor<2048x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<256x8x256xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2] : (tensor<256x8x256xf32>) -> tensor<256x8x256x256xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<256x8x256x256xf32>
      %308 = stablehlo.exponential %307 : tensor<256x8x256x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<256x8x256x256xf32>, tensor<f32>) -> tensor<256x8x256xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<256x8x256xf32>) -> tensor<256x8x256x256xf32>
      %311 = stablehlo.divide %308, %310 : tensor<256x8x256x256xf32>
      mhlo.return %311 : tensor<256x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256x256xf32>) -> tensor<256x8x256x256xf32>
    %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x8x256x256xf32>) -> tensor<2048x256x256xf32>
    %271 = "mhlo.fusion"(%261, %arg166, %arg167, %arg172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<65536x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<65536x256xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %307 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    %272 = "mhlo.fusion"(%271, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x8x32xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1, 3] : (tensor<256x256x8x32xf32>) -> tensor<256x8x256x32xf32>
      mhlo.return %306 : tensor<256x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x8x256x32xf32>
    %273 = mhlo.bitcast %272 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x8x256x32xf32>) -> tensor<2048x256x32xf32>
    %274 = "mhlo.fusion"(%270, %273) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2048x256x256xf32>, %arg209: tensor<2048x256x32xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256x256xf32>, tensor<2048x256x32xf32>) -> tensor<2048x256x32xf32>
      mhlo.return %303 : tensor<2048x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256x256xf32>, tensor<2048x256x32xf32>) -> tensor<2048x256x32xf32>
    %275 = "mhlo.fusion"(%274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256x32xf32>):
      %303 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2048x256x32xf32>) -> tensor<256x8x256x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<256x8x256x32xf32>) -> tensor<256x256x8x32xf32>
      mhlo.return %304 : tensor<256x256x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256x32xf32>) -> tensor<256x256x8x32xf32>
    %276 = mhlo.bitcast %275 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x256x8x32xf32>) -> tensor<65536x256xf32>
    %277 = "mhlo.fusion"(%276, %arg174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256x256xf32>):
      %303 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %303 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256x256xf32>) -> tensor<65536x256xf32>
    %278 = "mhlo.fusion"(%259, %277, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256x256xf32>, %arg209: tensor<65536x256xf32>, %arg210: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg209 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
      %306 = stablehlo.add %305, %arg208 : tensor<256x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<256x256xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %311 = stablehlo.subtract %306, %310 : tensor<256x256x256xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<256x256x256xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<256x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %316 = stablehlo.add %314, %315 : tensor<256x256xf32>
      %317 = stablehlo.rsqrt %316 : tensor<256x256xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<256x256x256xf32>
      mhlo.return %319 : tensor<256x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x256xf32>, tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x256x256xf32>
    %279 = mhlo.bitcast %278 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x256xf32>) -> tensor<65536x256xf32>
    %280 = "mhlo.fusion"(%279, %arg176, %arg177, %arg178) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1024x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<65536x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<65536x256xf32>
      %307 = stablehlo.dot_general %306, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<1024x256xf32>) -> tensor<65536x1024xf32>
      mhlo.return %307 : tensor<65536x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256xf32>) -> tensor<65536x1024xf32>
    %281 = "mhlo.fusion"(%280, %arg179) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<65536x1024xf32>, %arg209: tensor<1024xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<65536x1024xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x1024xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x1024xf32>) -> tensor<256x256x1024xf32>
      %306 = stablehlo.transpose %305, dims = [0, 2, 1] : (tensor<256x256x1024xf32>) -> tensor<256x1024x256xf32>
      mhlo.return %306 : tensor<256x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x1024xf32>, tensor<1024xf32>) -> tensor<256x1024x256xf32>
    %282 = mhlo.bitcast %281 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x1024x256xf32>) -> tensor<256x1024x16x16xf32>
    %283 = stablehlo.custom_call @__cudnn$convForward(%282, %arg180) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,1024,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<256x1024x16x16xf32>, tensor<1024x1x3x3xf32>) -> tuple<tensor<256x1024x16x16xf32>, tensor<0xui8>>
    %284 = stablehlo.get_tuple_element %283[0] : (tuple<tensor<256x1024x16x16xf32>, tensor<0xui8>>) -> tensor<256x1024x16x16xf32>
    %285 = "mhlo.fusion"(%284, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024x16x16xf32>, %arg209: tensor<1024xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x1024x16x16xf32>
      %304 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x16x16xf32>
      %305 = stablehlo.add %arg208, %304 : tensor<256x1024x16x16xf32>
      %306 = stablehlo.multiply %303, %305 : tensor<256x1024x16x16xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x16x16xf32>) -> tensor<256x1024x256xf32>
      %308 = stablehlo.negate %305 : tensor<256x1024x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1024x16x16xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<256x1024x16x16xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x1024x16x16xf32>) -> tensor<256x1024x256xf32>
      %312 = stablehlo.abs %311 : tensor<256x1024x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %314 = stablehlo.compare LT, %312, %313 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %316 = stablehlo.multiply %311, %311 : tensor<256x1024x256xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<256x1024x256xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %319 = stablehlo.add %317, %318 : tensor<256x1024x256xf32>
      %320 = stablehlo.multiply %319, %316 : tensor<256x1024x256xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %322 = stablehlo.add %320, %321 : tensor<256x1024x256xf32>
      %323 = stablehlo.multiply %322, %316 : tensor<256x1024x256xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %325 = stablehlo.add %323, %324 : tensor<256x1024x256xf32>
      %326 = stablehlo.multiply %325, %316 : tensor<256x1024x256xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %328 = stablehlo.add %326, %327 : tensor<256x1024x256xf32>
      %329 = stablehlo.multiply %328, %316 : tensor<256x1024x256xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %331 = stablehlo.add %329, %330 : tensor<256x1024x256xf32>
      %332 = stablehlo.multiply %331, %316 : tensor<256x1024x256xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %334 = stablehlo.add %332, %333 : tensor<256x1024x256xf32>
      %335 = stablehlo.multiply %311, %334 : tensor<256x1024x256xf32>
      %336 = stablehlo.subtract %313, %335 : tensor<256x1024x256xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %338 = stablehlo.compare LT, %311, %337 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %340 = stablehlo.negate %316 : tensor<256x1024x256xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %342 = stablehlo.compare LT, %340, %341 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
      %343 = stablehlo.exponential %340 : tensor<256x1024x256xf32>
      %344 = stablehlo.divide %313, %312 : tensor<256x1024x256xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<256x1024x256xf32>
      %346 = stablehlo.compare LT, %312, %339 : (tensor<256x1024x256xf32>, tensor<256x1024x256xf32>) -> tensor<256x1024x256xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %348 = stablehlo.divide %313, %316 : tensor<256x1024x256xf32>
      %349 = stablehlo.multiply %347, %348 : tensor<256x1024x256xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %351 = stablehlo.add %349, %350 : tensor<256x1024x256xf32>
      %352 = stablehlo.multiply %351, %348 : tensor<256x1024x256xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %354 = stablehlo.add %352, %353 : tensor<256x1024x256xf32>
      %355 = stablehlo.multiply %354, %348 : tensor<256x1024x256xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %357 = stablehlo.add %355, %356 : tensor<256x1024x256xf32>
      %358 = stablehlo.multiply %357, %348 : tensor<256x1024x256xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %360 = stablehlo.add %358, %359 : tensor<256x1024x256xf32>
      %361 = stablehlo.multiply %360, %348 : tensor<256x1024x256xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %363 = stablehlo.add %361, %362 : tensor<256x1024x256xf32>
      %364 = stablehlo.multiply %363, %348 : tensor<256x1024x256xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %366 = stablehlo.add %364, %365 : tensor<256x1024x256xf32>
      %367 = stablehlo.multiply %366, %348 : tensor<256x1024x256xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %369 = stablehlo.add %367, %368 : tensor<256x1024x256xf32>
      %370 = stablehlo.multiply %369, %348 : tensor<256x1024x256xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %372 = stablehlo.add %370, %371 : tensor<256x1024x256xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %374 = stablehlo.multiply %373, %348 : tensor<256x1024x256xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %375 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %376 = stablehlo.add %374, %375 : tensor<256x1024x256xf32>
      %377 = stablehlo.multiply %376, %348 : tensor<256x1024x256xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x1024x256xf32>
      %380 = stablehlo.multiply %379, %348 : tensor<256x1024x256xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %382 = stablehlo.add %380, %381 : tensor<256x1024x256xf32>
      %383 = stablehlo.multiply %382, %348 : tensor<256x1024x256xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %385 = stablehlo.add %383, %384 : tensor<256x1024x256xf32>
      %386 = stablehlo.multiply %385, %348 : tensor<256x1024x256xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %388 = stablehlo.add %386, %387 : tensor<256x1024x256xf32>
      %389 = stablehlo.multiply %388, %348 : tensor<256x1024x256xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %391 = stablehlo.add %389, %390 : tensor<256x1024x256xf32>
      %392 = stablehlo.multiply %391, %348 : tensor<256x1024x256xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<256x1024x256xf32>
      %394 = stablehlo.add %392, %393 : tensor<256x1024x256xf32>
      %395 = stablehlo.select %346, %372, %394 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
      %396 = stablehlo.multiply %345, %395 : tensor<256x1024x256xf32>
      %397 = stablehlo.select %342, %337, %396 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
      %398 = stablehlo.subtract %339, %397 : tensor<256x1024x256xf32>
      %399 = stablehlo.select %338, %398, %397 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
      %400 = stablehlo.select %314, %336, %399 : tensor<256x1024x256xi1>, tensor<256x1024x256xf32>
      %401 = stablehlo.multiply %307, %400 : tensor<256x1024x256xf32>
      mhlo.return %401 : tensor<256x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x16x16xf32>, tensor<1024xf32>) -> tensor<256x1024x256xf32>
    %286 = "mhlo.fusion"(%285, %arg182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x1024x256xf32>, %arg209: tensor<256x1024xf32>):
      %303 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<256x1024x256xf32>) -> tensor<256x256x1024xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x256x1024xf32>) -> tensor<65536x1024xf32>
      %305 = stablehlo.dot_general %304, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x1024xf32>, tensor<256x1024xf32>) -> tensor<65536x256xf32>
      mhlo.return %305 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x256xf32>, tensor<256x1024xf32>) -> tensor<65536x256xf32>
    %287 = "mhlo.fusion"(%286, %arg183, %259, %277, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256x256x256xf32>, %arg211: tensor<65536x256xf32>, %arg212: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<65536x256xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
      %306 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %307 = stablehlo.add %306, %arg211 : tensor<65536x256xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<256x256x256xf32>
      %309 = stablehlo.add %308, %arg210 : tensor<256x256x256xf32>
      %310 = stablehlo.add %305, %309 : tensor<256x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<256x256xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<256x256x256xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<256x256x256xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x256x256xf32>, tensor<f32>) -> tensor<256x256xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<256x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<256x256xf32>
      %321 = stablehlo.rsqrt %320 : tensor<256x256xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<256x256x256xf32>
      mhlo.return %323 : tensor<256x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>, tensor<256x256x256xf32>, tensor<65536x256xf32>, tensor<256xf32>) -> tensor<256x256x256xf32>
    %288 = "mhlo.fusion"(%287, %arg190, %arg191, %arg198) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x256x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x256x256xf32>
      %307 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<256x256x256xf32>
      %309 = stablehlo.dot_general %306, %308, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256x256xf32>, tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
      mhlo.return %309 : tensor<256x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<256x256x256xf32>
    %289 = "mhlo.fusion"(%288, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<256x256x256xf32>
      %304 = stablehlo.add %arg208, %303 : tensor<256x256x256xf32>
      mhlo.return %304 : tensor<256x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x256xf32>, tensor<256xf32>) -> tensor<256x256x256xf32>
    %290 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %304 = stablehlo.iota dim = 0 : tensor<128xi32>
      %305 = stablehlo.convert %304 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %307 = stablehlo.add %305, %306 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %311 = stablehlo.add %309, %310 : tensor<128xf32>
      %312 = stablehlo.maximum %303, %311 : tensor<128xf32>
      %313 = stablehlo.convert %312 : (tensor<128xf32>) -> tensor<128xi32>
      %314 = stablehlo.convert %313 : (tensor<128xi32>) -> tensor<128xf32>
      %315 = stablehlo.subtract %312, %314 : tensor<128xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %317 = stablehlo.clamp %303, %315, %316 : tensor<128xf32>
      mhlo.return %317 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %291 = "mhlo.fusion"(%227#1, %arg188, %arg189, %arg196) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<256x160xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x1024x160xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<256x1024x160xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x1024x160xf32>
      %307 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x160xf32>) -> tensor<160x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<256x160x256xf32>
      %309 = stablehlo.dot_general %306, %308, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1024x160xf32>, tensor<256x160x256xf32>) -> tensor<256x1024x256xf32>
      mhlo.return %309 : tensor<256x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<256x160xf32>) -> tensor<256x1024x256xf32>
    %292 = "mhlo.fusion"(%291, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024x256xf32>, %arg209: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<256x1024x256xf32>
      %304 = stablehlo.add %arg208, %303 : tensor<256x1024x256xf32>
      mhlo.return %304 : tensor<256x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x256xf32>, tensor<256xf32>) -> tensor<256x1024x256xf32>
    %293 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %304 = stablehlo.iota dim = 0 : tensor<128xi32>
      %305 = stablehlo.convert %304 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %307 = stablehlo.add %305, %306 : tensor<128xf32>
      %308 = stablehlo.multiply %307, %306 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %310 = stablehlo.add %308, %309 : tensor<128xf32>
      %311 = stablehlo.maximum %303, %310 : tensor<128xf32>
      %312 = stablehlo.convert %311 : (tensor<128xf32>) -> tensor<128xi32>
      %313 = stablehlo.convert %312 : (tensor<128xi32>) -> tensor<128xf32>
      %314 = stablehlo.subtract %311, %313 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %316 = stablehlo.clamp %303, %314, %315 : tensor<128xf32>
      mhlo.return %316 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %294 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %304 = stablehlo.iota dim = 0 : tensor<128xi32>
      %305 = stablehlo.convert %304 : (tensor<128xi32>) -> tensor<128xf32>
      %306 = stablehlo.maximum %303, %305 : tensor<128xf32>
      %307 = stablehlo.convert %306 : (tensor<128xf32>) -> tensor<128xi32>
      %308 = stablehlo.convert %307 : (tensor<128xi32>) -> tensor<128xf32>
      %309 = stablehlo.subtract %306, %308 : tensor<128xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %311 = stablehlo.clamp %303, %309, %310 : tensor<128xf32>
      mhlo.return %311 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %295 = "mhlo.fusion"(%73#1, %arg184, %arg185, %arg192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<256x32xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x16384x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<256x16384x32xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x16384x32xf32>
      %307 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x32xf32>) -> tensor<32x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<256x32x256xf32>
      %309 = stablehlo.dot_general %306, %308, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x16384x32xf32>, tensor<256x32x256xf32>) -> tensor<256x16384x256xf32>
      mhlo.return %309 : tensor<256x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<256x32xf32>) -> tensor<256x16384x256xf32>
    %296 = "mhlo.fusion"(%149#1, %arg186, %arg187, %arg194) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %303 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %304 = stablehlo.multiply %arg208, %303 : tensor<256x4096x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<256x4096x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<256x4096x64xf32>
      %307 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<256x64x256xf32>
      %309 = stablehlo.dot_general %306, %308, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x64xf32>, tensor<256x64x256xf32>) -> tensor<256x4096x256xf32>
      mhlo.return %309 : tensor<256x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<256x4096x256xf32>
    %297 = "mhlo.fusion"(%1, %2, %3, %4, %5, %289, %290, %292, %293, %294, %295, %arg193, %296, %arg195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x1xf32>, %arg209: tensor<128x1xf32>, %arg210: tensor<128x1xf32>, %arg211: tensor<128x1xf32>, %arg212: tensor<128xf32>, %arg213: tensor<256x256x256xf32>, %arg214: tensor<128xf32>, %arg215: tensor<256x1024x256xf32>, %arg216: tensor<128xf32>, %arg217: tensor<128xf32>, %arg218: tensor<256x16384x256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256x4096x256xf32>, %arg221: tensor<256xf32>):
      %303 = mhlo.bitcast %arg213 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256,16,16]{1,3,2,0}"} : (tensor<256x256x256xf32>) -> tensor<256x256x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.iota dim = 0 : tensor<128xi32>
      %306 = stablehlo.convert %305 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %308 = stablehlo.add %306, %307 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %312 = stablehlo.add %310, %311 : tensor<128xf32>
      %313 = stablehlo.maximum %304, %312 : tensor<128xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %315 = stablehlo.convert %314 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %316 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_3 = stablehlo.constant dense<16> : tensor<i32>
      %318 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %319 = stablehlo.add %315, %318 : tensor<128x1xi32>
      %320 = stablehlo.select %317, %319, %315 : tensor<128x1xi1>, tensor<128x1xi32>
      %321 = mhlo.bitcast %320 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %323 = stablehlo.convert %313 : (tensor<128xf32>) -> tensor<128xi32>
      %324 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %325 = stablehlo.compare LT, %323, %324 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %326 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %327 = stablehlo.add %323, %326 : tensor<128xi32>
      %328 = stablehlo.select %325, %327, %323 : tensor<128xi1>, tensor<128xi32>
      %329 = stablehlo.broadcast_in_dim %328, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %330 = stablehlo.concatenate %322, %329, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %331 = mhlo.bitcast %330 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %332 = "stablehlo.gather"(%303, %331) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %333 = mhlo.bitcast %332 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %c_4 = stablehlo.constant dense<15> : tensor<i32>
      %334 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %c_5 = stablehlo.constant dense<1> : tensor<i32>
      %335 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %336 = stablehlo.add %323, %335 : tensor<128xi32>
      %337 = stablehlo.minimum %334, %336 : tensor<128xi32>
      %338 = stablehlo.compare LT, %337, %324 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %339 = stablehlo.add %337, %326 : tensor<128xi32>
      %340 = stablehlo.select %338, %339, %337 : tensor<128xi1>, tensor<128xi32>
      %341 = stablehlo.broadcast_in_dim %340, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %342 = stablehlo.concatenate %322, %341, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %343 = mhlo.bitcast %342 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %344 = "stablehlo.gather"(%303, %343) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %345 = mhlo.bitcast %344 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %346 = stablehlo.subtract %345, %333 : tensor<256x256x128x128xf32>
      %347 = stablehlo.broadcast_in_dim %arg212, dims = [3] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<256x256x128x128xf32>
      %349 = stablehlo.add %333, %348 : tensor<256x256x128x128xf32>
      %350 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %351 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %352 = stablehlo.add %315, %351 : tensor<128x1xi32>
      %353 = stablehlo.minimum %350, %352 : tensor<128x1xi32>
      %354 = stablehlo.compare LT, %353, %316 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %355 = stablehlo.add %353, %318 : tensor<128x1xi32>
      %356 = stablehlo.select %354, %355, %353 : tensor<128x1xi1>, tensor<128x1xi32>
      %357 = mhlo.bitcast %356 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %358 = stablehlo.broadcast_in_dim %357, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %359 = stablehlo.concatenate %358, %329, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %360 = mhlo.bitcast %359 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %361 = "stablehlo.gather"(%303, %360) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %362 = mhlo.bitcast %361 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %363 = stablehlo.concatenate %358, %341, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %364 = mhlo.bitcast %363 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %365 = "stablehlo.gather"(%303, %364) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %366 = mhlo.bitcast %365 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %367 = stablehlo.subtract %366, %362 : tensor<256x256x128x128xf32>
      %368 = stablehlo.multiply %367, %347 : tensor<256x256x128x128xf32>
      %369 = stablehlo.add %362, %368 : tensor<256x256x128x128xf32>
      %370 = stablehlo.subtract %369, %349 : tensor<256x256x128x128xf32>
      %371 = mhlo.bitcast %arg208 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %372 = stablehlo.broadcast_in_dim %371, dims = [2] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
      %373 = stablehlo.multiply %370, %372 : tensor<256x256x128x128xf32>
      %374 = stablehlo.add %349, %373 : tensor<256x256x128x128xf32>
      %375 = mhlo.bitcast %arg215 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256,32,32]{1,3,2,0}"} : (tensor<256x1024x256xf32>) -> tensor<256x256x32x32xf32>
      %cst_6 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %377 = stablehlo.multiply %308, %376 : tensor<128xf32>
      %378 = stablehlo.add %377, %311 : tensor<128xf32>
      %379 = stablehlo.maximum %304, %378 : tensor<128xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %381 = stablehlo.convert %380 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %382 = stablehlo.compare LT, %381, %316 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<32> : tensor<i32>
      %383 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %384 = stablehlo.add %381, %383 : tensor<128x1xi32>
      %385 = stablehlo.select %382, %384, %381 : tensor<128x1xi1>, tensor<128x1xi32>
      %386 = mhlo.bitcast %385 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %388 = stablehlo.convert %379 : (tensor<128xf32>) -> tensor<128xi32>
      %389 = stablehlo.compare LT, %388, %324 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %390 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %391 = stablehlo.add %388, %390 : tensor<128xi32>
      %392 = stablehlo.select %389, %391, %388 : tensor<128xi1>, tensor<128xi32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %394 = stablehlo.concatenate %387, %393, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %396 = "stablehlo.gather"(%375, %395) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %c_8 = stablehlo.constant dense<31> : tensor<i32>
      %398 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %399 = stablehlo.add %388, %335 : tensor<128xi32>
      %400 = stablehlo.minimum %398, %399 : tensor<128xi32>
      %401 = stablehlo.compare LT, %400, %324 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %402 = stablehlo.add %400, %390 : tensor<128xi32>
      %403 = stablehlo.select %401, %402, %400 : tensor<128xi1>, tensor<128xi32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %405 = stablehlo.concatenate %387, %404, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %407 = "stablehlo.gather"(%375, %406) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %409 = stablehlo.subtract %408, %397 : tensor<256x256x128x128xf32>
      %410 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
      %411 = stablehlo.multiply %409, %410 : tensor<256x256x128x128xf32>
      %412 = stablehlo.add %397, %411 : tensor<256x256x128x128xf32>
      %413 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %414 = stablehlo.add %381, %351 : tensor<128x1xi32>
      %415 = stablehlo.minimum %413, %414 : tensor<128x1xi32>
      %416 = stablehlo.compare LT, %415, %316 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %417 = stablehlo.add %415, %383 : tensor<128x1xi32>
      %418 = stablehlo.select %416, %417, %415 : tensor<128x1xi1>, tensor<128x1xi32>
      %419 = mhlo.bitcast %418 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %420 = stablehlo.broadcast_in_dim %419, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %421 = stablehlo.concatenate %420, %393, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %423 = "stablehlo.gather"(%375, %422) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %425 = stablehlo.concatenate %420, %404, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %426 = mhlo.bitcast %425 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %427 = "stablehlo.gather"(%375, %426) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %429 = stablehlo.subtract %428, %424 : tensor<256x256x128x128xf32>
      %430 = stablehlo.multiply %429, %410 : tensor<256x256x128x128xf32>
      %431 = stablehlo.add %424, %430 : tensor<256x256x128x128xf32>
      %432 = stablehlo.subtract %431, %412 : tensor<256x256x128x128xf32>
      %433 = mhlo.bitcast %arg209 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %434 = stablehlo.broadcast_in_dim %433, dims = [2] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
      %435 = stablehlo.multiply %432, %434 : tensor<256x256x128x128xf32>
      %436 = stablehlo.add %412, %435 : tensor<256x256x128x128xf32>
      %437 = stablehlo.broadcast_in_dim %arg221, dims = [2] : (tensor<256xf32>) -> tensor<256x4096x256xf32>
      %438 = stablehlo.add %arg220, %437 : tensor<256x4096x256xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256,64,64]{1,3,2,0}"} : (tensor<256x4096x256xf32>) -> tensor<256x256x64x64xf32>
      %440 = stablehlo.multiply %308, %307 : tensor<128xf32>
      %441 = stablehlo.add %440, %311 : tensor<128xf32>
      %442 = stablehlo.maximum %304, %441 : tensor<128xf32>
      %443 = mhlo.bitcast %442 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %444 = stablehlo.convert %443 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %445 = stablehlo.compare LT, %444, %316 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<64> : tensor<i32>
      %446 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %447 = stablehlo.add %444, %446 : tensor<128x1xi32>
      %448 = stablehlo.select %445, %447, %444 : tensor<128x1xi1>, tensor<128x1xi32>
      %449 = mhlo.bitcast %448 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %450 = stablehlo.broadcast_in_dim %449, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %451 = stablehlo.convert %442 : (tensor<128xf32>) -> tensor<128xi32>
      %452 = stablehlo.compare LT, %451, %324 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %453 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %454 = stablehlo.add %451, %453 : tensor<128xi32>
      %455 = stablehlo.select %452, %454, %451 : tensor<128xi1>, tensor<128xi32>
      %456 = stablehlo.broadcast_in_dim %455, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %457 = stablehlo.concatenate %450, %456, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %458 = mhlo.bitcast %457 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %459 = "stablehlo.gather"(%439, %458) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %c_10 = stablehlo.constant dense<63> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %462 = stablehlo.add %451, %335 : tensor<128xi32>
      %463 = stablehlo.minimum %461, %462 : tensor<128xi32>
      %464 = stablehlo.compare LT, %463, %324 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %465 = stablehlo.add %463, %453 : tensor<128xi32>
      %466 = stablehlo.select %464, %465, %463 : tensor<128xi1>, tensor<128xi32>
      %467 = stablehlo.broadcast_in_dim %466, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %468 = stablehlo.concatenate %450, %467, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %469 = mhlo.bitcast %468 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %470 = "stablehlo.gather"(%439, %469) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %472 = stablehlo.subtract %471, %460 : tensor<256x256x128x128xf32>
      %473 = stablehlo.broadcast_in_dim %arg216, dims = [3] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
      %474 = stablehlo.multiply %472, %473 : tensor<256x256x128x128xf32>
      %475 = stablehlo.add %460, %474 : tensor<256x256x128x128xf32>
      %476 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %477 = stablehlo.add %444, %351 : tensor<128x1xi32>
      %478 = stablehlo.minimum %476, %477 : tensor<128x1xi32>
      %479 = stablehlo.compare LT, %478, %316 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %480 = stablehlo.add %478, %446 : tensor<128x1xi32>
      %481 = stablehlo.select %479, %480, %478 : tensor<128x1xi1>, tensor<128x1xi32>
      %482 = mhlo.bitcast %481 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %483 = stablehlo.broadcast_in_dim %482, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %484 = stablehlo.concatenate %483, %456, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %485 = mhlo.bitcast %484 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %486 = "stablehlo.gather"(%439, %485) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %487 = mhlo.bitcast %486 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %488 = stablehlo.concatenate %483, %467, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %489 = mhlo.bitcast %488 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %490 = "stablehlo.gather"(%439, %489) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %491 = mhlo.bitcast %490 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %492 = stablehlo.subtract %491, %487 : tensor<256x256x128x128xf32>
      %493 = stablehlo.multiply %492, %473 : tensor<256x256x128x128xf32>
      %494 = stablehlo.add %487, %493 : tensor<256x256x128x128xf32>
      %495 = stablehlo.subtract %494, %475 : tensor<256x256x128x128xf32>
      %496 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %497 = stablehlo.broadcast_in_dim %496, dims = [2] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
      %498 = stablehlo.multiply %495, %497 : tensor<256x256x128x128xf32>
      %499 = stablehlo.add %475, %498 : tensor<256x256x128x128xf32>
      %500 = stablehlo.broadcast_in_dim %arg219, dims = [2] : (tensor<256xf32>) -> tensor<256x16384x256xf32>
      %501 = stablehlo.add %arg218, %500 : tensor<256x16384x256xf32>
      %502 = mhlo.bitcast %501 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256,128,128]{1,3,2,0}"} : (tensor<256x16384x256xf32>) -> tensor<256x256x128x128xf32>
      %503 = stablehlo.maximum %304, %306 : tensor<128xf32>
      %504 = mhlo.bitcast %503 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %505 = stablehlo.convert %504 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %506 = stablehlo.compare LT, %505, %316 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_11 = stablehlo.constant dense<128> : tensor<i32>
      %507 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %508 = stablehlo.add %505, %507 : tensor<128x1xi32>
      %509 = stablehlo.select %506, %508, %505 : tensor<128x1xi1>, tensor<128x1xi32>
      %510 = mhlo.bitcast %509 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %511 = stablehlo.broadcast_in_dim %510, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %512 = stablehlo.convert %503 : (tensor<128xf32>) -> tensor<128xi32>
      %513 = stablehlo.compare LT, %512, %324 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %514 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %515 = stablehlo.add %512, %514 : tensor<128xi32>
      %516 = stablehlo.select %513, %515, %512 : tensor<128xi1>, tensor<128xi32>
      %517 = stablehlo.broadcast_in_dim %516, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %518 = stablehlo.concatenate %511, %517, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %520 = "stablehlo.gather"(%502, %519) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %521 = mhlo.bitcast %520 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %c_12 = stablehlo.constant dense<127> : tensor<i32>
      %522 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %523 = stablehlo.add %512, %335 : tensor<128xi32>
      %524 = stablehlo.minimum %522, %523 : tensor<128xi32>
      %525 = stablehlo.compare LT, %524, %324 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %526 = stablehlo.add %524, %514 : tensor<128xi32>
      %527 = stablehlo.select %525, %526, %524 : tensor<128xi1>, tensor<128xi32>
      %528 = stablehlo.broadcast_in_dim %527, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %529 = stablehlo.concatenate %511, %528, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %530 = mhlo.bitcast %529 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %531 = "stablehlo.gather"(%502, %530) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %532 = mhlo.bitcast %531 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %533 = stablehlo.subtract %532, %521 : tensor<256x256x128x128xf32>
      %534 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
      %535 = stablehlo.multiply %533, %534 : tensor<256x256x128x128xf32>
      %536 = stablehlo.add %521, %535 : tensor<256x256x128x128xf32>
      %537 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %538 = stablehlo.add %505, %351 : tensor<128x1xi32>
      %539 = stablehlo.minimum %537, %538 : tensor<128x1xi32>
      %540 = stablehlo.compare LT, %539, %316 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %541 = stablehlo.add %539, %507 : tensor<128x1xi32>
      %542 = stablehlo.select %540, %541, %539 : tensor<128x1xi1>, tensor<128x1xi32>
      %543 = mhlo.bitcast %542 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %544 = stablehlo.broadcast_in_dim %543, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %545 = stablehlo.concatenate %544, %517, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %546 = mhlo.bitcast %545 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %547 = "stablehlo.gather"(%502, %546) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %548 = mhlo.bitcast %547 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %549 = stablehlo.concatenate %544, %528, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %550 = mhlo.bitcast %549 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %551 = "stablehlo.gather"(%502, %550) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,256,256,1,1]{0,2,1,4,3}"} : (tensor<256x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x256x256x1x1xf32>
      %552 = mhlo.bitcast %551 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x256x256x1x1xf32>) -> tensor<256x256x128x128xf32>
      %553 = stablehlo.subtract %552, %548 : tensor<256x256x128x128xf32>
      %554 = stablehlo.multiply %553, %534 : tensor<256x256x128x128xf32>
      %555 = stablehlo.add %548, %554 : tensor<256x256x128x128xf32>
      %556 = stablehlo.subtract %555, %536 : tensor<256x256x128x128xf32>
      %557 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %558 = stablehlo.broadcast_in_dim %557, dims = [2] : (tensor<128xf32>) -> tensor<256x256x128x128xf32>
      %559 = stablehlo.multiply %556, %558 : tensor<256x256x128x128xf32>
      %560 = stablehlo.add %536, %559 : tensor<256x256x128x128xf32>
      %561 = stablehlo.concatenate %374, %436, %499, %560, dim = 1 : (tensor<256x256x128x128xf32>, tensor<256x256x128x128xf32>, tensor<256x256x128x128xf32>, tensor<256x256x128x128xf32>) -> tensor<256x1024x128x128xf32>
      mhlo.return %561 : tensor<256x1024x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>, tensor<128xf32>, tensor<256x256x256xf32>, tensor<128xf32>, tensor<256x1024x256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x16384x256xf32>, tensor<256xf32>, tensor<256x4096x256xf32>, tensor<256xf32>) -> tensor<256x1024x128x128xf32>
    %298 = stablehlo.custom_call @__cudnn$convForward(%297, %arg200) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,256,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<256x1024x128x128xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<256x256x128x128xf32>, tensor<0xui8>>
    %299 = stablehlo.get_tuple_element %298[0] : (tuple<tensor<256x256x128x128xf32>, tensor<0xui8>>) -> tensor<256x256x128x128xf32>
    %300 = "mhlo.fusion"(%arg202, %arg201, %0, %299, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256x128x128xf32>, %arg212: tensor<256xf32>):
      %303 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<256x256x128x128xf32>
      %304 = stablehlo.subtract %arg211, %303 : tensor<256x256x128x128xf32>
      %305 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<256x256x128x128xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<256x256x128x128xf32>
      %307 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256x128x128xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<256x256x128x128xf32>
      %309 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<256xf32>) -> tensor<256x256x128x128xf32>
      %310 = stablehlo.add %308, %309 : tensor<256x256x128x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256x128x128xf32>
      %312 = stablehlo.maximum %310, %311 : tensor<256x256x128x128xf32>
      mhlo.return %312 : tensor<256x256x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x128x128xf32>, tensor<256xf32>) -> tensor<256x256x128x128xf32>
    %301 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%300, %arg203, %arg204) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[256,150,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<256x256x128x128xf32>, tensor<150x256x1x1xf32>, tensor<150xf32>) -> tuple<tensor<256x150x128x128xf32>, tensor<0xui8>>
    %302 = stablehlo.get_tuple_element %301[0] : (tuple<tensor<256x150x128x128xf32>, tensor<0xui8>>) -> tensor<256x150x128x128xf32>
    return %302 : tensor<256x150x128x128xf32>
  }
}
