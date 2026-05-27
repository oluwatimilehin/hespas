module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.101_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.102_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.3_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.109_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.110_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.5_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.116_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.117_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.118_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.7_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.124_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.125_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.9_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.136_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.11_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.140_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.141_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.142_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.143_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.144_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.13_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.148_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.149_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.150_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.151_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.152_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.15_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.156_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.157_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.158_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.159_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.160_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.17_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.164_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.165_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.166_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.167_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.168_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.19_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.172_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.173_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.174_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.175_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.176_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.21_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.180_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.181_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.182_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.183_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.184_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot.23_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3200x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
    return %6 : tensor<3200x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.188_computation(%arg0: tensor<768x50x64xf32>, %arg1: tensor<768x64x50xf32>) -> tensor<768x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    return %0 : tensor<768x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.189_computation(%arg0: tensor<768x50x50xf32>, %arg1: tensor<768x50x64xf32>) -> tensor<768x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
    return %0 : tensor<768x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.190_computation(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<64x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
    return %2 : tensor<64x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.191_computation(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    return %4 : tensor<3200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.192_computation(%arg0: tensor<3200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    return %0 : tensor<3200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.193_computation(%arg0: tensor<64x768xf32>, %arg1: tensor<2x768xf32>) -> tensor<64x2xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x768xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x768xf32>, tensor<2x768xf32>) -> tensor<64x2xf32>
    return %2 : tensor<64x2xf32>
  }
  func.func private @region_1.4.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.62(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.65(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.67(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.70(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.72(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.75(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.77(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.80(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.82(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.85(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.87(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.90(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.92(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.95(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.97(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.100(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.102(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.105(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.107(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.110(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.112(%arg0: tensor<64x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<64x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<64x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<64x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<64x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<64x50x768xf32>
    return %16 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.115(%arg0: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    return %12 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.117(%arg0: tensor<768xf32>, %arg1: tensor<64x768x50xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg4, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    %13 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x50x768xf32>
    %15 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x50x768xf32>
    %17 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %18 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x50x768xf32>
    %20 = stablehlo.add %16, %19 : tensor<64x50x768xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %22 = stablehlo.multiply %21, %1 : tensor<64x50xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<64x50x768xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<64x50x768xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %27 = stablehlo.multiply %26, %1 : tensor<64x50xf32>
    %28 = stablehlo.add %27, %8 : tensor<64x50xf32>
    %29 = stablehlo.rsqrt %28 : tensor<64x50xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<64x50x768xf32>
    return %31 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.120(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x50x768xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x50x768xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %18 = stablehlo.multiply %17, %1 : tensor<64x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<64x50x768xf32>
    %21 = stablehlo.multiply %20, %20 : tensor<64x50x768xf32>
    %22 = stablehlo.reduce(%21 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %23 = stablehlo.multiply %22, %1 : tensor<64x50xf32>
    %24 = stablehlo.add %23, %8 : tensor<64x50xf32>
    %25 = stablehlo.rsqrt %24 : tensor<64x50xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %27 = stablehlo.multiply %20, %26 : tensor<64x50x768xf32>
    return %27 : tensor<64x50x768xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.6(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.7(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.8(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.9(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.10(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_transpose.11(%arg0: tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
    return %1 : tensor<64x50x12x64xf32>
  }
  func.func private @fused_select(%arg0: tensor<50x768xf32>, %arg1: tensor<1x50xi32>) -> tensor<1x50x768xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x50xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<1x50xi1>, tensor<1x50xi32>
    %5 = stablehlo.compare GE, %4, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_1 = stablehlo.constant dense<49> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %7 = stablehlo.compare LE, %4, %6 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %8 = stablehlo.and %5, %7 : tensor<1x50xi1>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
    %11 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
    %12 = "stablehlo.gather"(%arg0, %11) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[50,1,768]{2,0,1}"} : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 0, 1]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
    %15 = stablehlo.select %10, %13, %14 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
    return %15 : tensor<1x50x768xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.12(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.13(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.14(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.15(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.16(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.17(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.18(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.19(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.20(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.21(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.22(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.23(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.24(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.25(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.26(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.27(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.28(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.29(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.30(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.31(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.32(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.33(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.34(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.35(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.36(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.37(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.38(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.39(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.40(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.41(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    return %7 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.42(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.43(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.44(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<64x768x50xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<64x50x768xf32>) -> tensor<64x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg6 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<64x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.subtract %arg6, %3 : tensor<64x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<64x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<64x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<64x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<64x50x768xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x50x768xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x50x768xf32>
    %17 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %18 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x50x768xf32>
    %20 = stablehlo.add %16, %19 : tensor<64x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %22 = stablehlo.add %21, %arg0 : tensor<3200x768xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %24 = stablehlo.add %20, %23 : tensor<64x50x768xf32>
    return %24 : tensor<64x50x768xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<3200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<64x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<64x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<64x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<64x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<64x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<64x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<64x50x3072xf32>
    return %10 : tensor<64x50x3072xf32>
  }
  func.func private @fused_transpose.45(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %4 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.46(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
    return %6 : tensor<64x12x64x50xf32>
  }
  func.func private @fused_transpose.47(%arg0: tensor<3200x2304xf32>, %arg1: tensor<768xf32>) -> tensor<64x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %1 = stablehlo.slice %arg0 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3200x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
    return %6 : tensor<64x12x50x64xf32>
  }
  func.func private @fused_transpose.48(%arg0: tensor<64x768x7x7xf32>) -> tensor<64x49x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x768x7x7xf32>) -> tensor<64x768x49xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<64x768x49xf32>) -> tensor<64x49x768xf32>
    return %1 : tensor<64x49x768xf32>
  }
  func.func private @region_1.4.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.185(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.186(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.1(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.187(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.2(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.188(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.3(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.189(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.4(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.190(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.5(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.191(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.6(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.192(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.7(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.193(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.8(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.194(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.9(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.195(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @region_1.4.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.10.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.10(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.196(%arg0: tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<64x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
    %5 = stablehlo.not %4 : tensor<64x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<64x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<64x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<64x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
    return %16 : tensor<64x12x50x50xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<64x49x768xf32>, %arg1: tensor<1x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<64x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<64x1x768xf32>
    %1 = stablehlo.concatenate %0, %arg0, dim = 1 : (tensor<64x1x768xf32>, tensor<64x49x768xf32>) -> tensor<64x50x768xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<64x50x768xf32>
    %4 = stablehlo.add %1, %3 : tensor<64x50x768xf32>
    return %4 : tensor<64x50x768xf32>
  }
  func.func private @region_1.4.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<3200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<64x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<64x768x50xf32>) -> tensor<64x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<64x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<3200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<64x50x768xf32>
    %8 = stablehlo.slice %7 [0:64, 1:50, 0:768] : (tensor<64x50x768xf32>) -> tensor<64x49x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x768xf32>
    return %9 : tensor<64x768xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<64x2xf32>, %arg1: tensor<2xf32>) -> tensor<64x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<64x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<64x2xf32>
    return %1 : tensor<64x2xf32>
  }
  func.func @main(%arg0: tensor<768xf32>, %arg1: tensor<768x3x32x32xf32>, %arg2: tensor<50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768x768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<3072x768xf32>, %arg16: tensor<3072xf32>, %arg17: tensor<768x3072xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768x768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<3072x768xf32>, %arg32: tensor<3072xf32>, %arg33: tensor<768x3072xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<3072x768xf32>, %arg48: tensor<3072xf32>, %arg49: tensor<768x3072xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768x768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<3072x768xf32>, %arg64: tensor<3072xf32>, %arg65: tensor<768x3072xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768x768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<3072x768xf32>, %arg80: tensor<3072xf32>, %arg81: tensor<768x3072xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<3072x768xf32>, %arg96: tensor<3072xf32>, %arg97: tensor<768x3072xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768x768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<3072x768xf32>, %arg112: tensor<3072xf32>, %arg113: tensor<768x3072xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768x768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<3072x768xf32>, %arg128: tensor<3072xf32>, %arg129: tensor<768x3072xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768xf32>, %arg133: tensor<768x768xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<3072x768xf32>, %arg144: tensor<3072xf32>, %arg145: tensor<768x3072xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768xf32>, %arg149: tensor<768x768xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768x768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<3072x768xf32>, %arg160: tensor<3072xf32>, %arg161: tensor<768x3072xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768xf32>, %arg165: tensor<768x768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<3072x768xf32>, %arg176: tensor<3072xf32>, %arg177: tensor<768x3072xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768xf32>, %arg181: tensor<768x768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768x768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<3072x768xf32>, %arg192: tensor<3072xf32>, %arg193: tensor<768x3072xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<2x768xf32>, %arg198: tensor<2xf32>, %arg199: tensor<1x50xi32>, %arg200: tensor<64x3x224x224xf32>) -> tensor<64x2xf32> {
    %0 = stablehlo.custom_call @__cudnn$convForward(%arg200, %arg1) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,768,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<64x3x224x224xf32>, tensor<768x3x32x32xf32>) -> tuple<tensor<64x768x7x7xf32>, tensor<0xui8>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<64x768x7x7xf32>, tensor<0xui8>>) -> tensor<64x768x7x7xf32>
    %2 = "mhlo.fusion"(%1) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<64x768x7x7xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x768x7x7xf32>) -> tensor<64x768x49xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1] : (tensor<64x768x49xf32>) -> tensor<64x49x768xf32>
      mhlo.return %284 : tensor<64x49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x768x7x7xf32>) -> tensor<64x49x768xf32>
    %3 = "mhlo.fusion"(%arg2, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<1x50xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %283 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %284 = stablehlo.compare LT, %arg202, %283 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_0 = stablehlo.constant dense<50> : tensor<i32>
      %285 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %286 = stablehlo.add %arg202, %285 : tensor<1x50xi32>
      %287 = stablehlo.select %284, %286, %arg202 : tensor<1x50xi1>, tensor<1x50xi32>
      %288 = stablehlo.compare GE, %287, %283 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %290 = stablehlo.compare LE, %287, %289 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %291 = stablehlo.and %288, %290 : tensor<1x50xi1>
      %292 = mhlo.bitcast %291 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %293 = stablehlo.broadcast_in_dim %292, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %294 = mhlo.bitcast %287 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
      %295 = "stablehlo.gather"(%arg201, %294) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[50,1,768]{2,0,1}"} : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 0, 1]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %297 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %298 = stablehlo.select %293, %296, %297 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      mhlo.return %298 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<1x50xi32>) -> tensor<1x50x768xf32>
    %4 = "mhlo.fusion"(%2, %3, %arg0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<64x49x768xf32>, %arg202: tensor<1x50x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<64x1x768xf32>
      %284 = stablehlo.concatenate %283, %arg201, dim = 1 : (tensor<64x1x768xf32>, tensor<64x49x768xf32>) -> tensor<64x50x768xf32>
      %285 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.add %284, %286 : tensor<64x50x768xf32>
      mhlo.return %287 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x49x768xf32>, tensor<1x50x768xf32>, tensor<768xf32>) -> tensor<64x50x768xf32>
    %5 = "mhlo.fusion"(%arg4, %arg3, %4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg203 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg203, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      %296 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<64x50x768xf32>
      %298 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.add %297, %298 : tensor<64x50x768xf32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %301 = stablehlo.multiply %300, %284 : tensor<64x50xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %303 = stablehlo.subtract %299, %302 : tensor<64x50x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<64x50x768xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %306 = stablehlo.multiply %305, %284 : tensor<64x50xf32>
      %307 = stablehlo.add %306, %291 : tensor<64x50xf32>
      %308 = stablehlo.rsqrt %307 : tensor<64x50xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %310 = stablehlo.multiply %303, %309 : tensor<64x50x768xf32>
      mhlo.return %310 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %7 = "mhlo.fusion"(%6, %arg13, %arg14, %arg9, %arg5, %arg7) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %8 = "mhlo.fusion"(%7, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %10 = "mhlo.fusion"(%7, %arg6) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %12 = "mhlo.fusion"(%9, %11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %13 = "mhlo.fusion"(%12) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %15 = "mhlo.fusion"(%7, %arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %17 = "mhlo.fusion"(%14, %16) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %18 = "mhlo.fusion"(%17) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %20 = "mhlo.fusion"(%19, %arg11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %21 = "mhlo.fusion"(%arg12, %20, %arg4, %arg3, %4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<64x768x50xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg205 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg205, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      %296 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<64x50x768xf32>
      %298 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.add %297, %298 : tensor<64x50x768xf32>
      %300 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %301 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %302 = stablehlo.add %300, %301 : tensor<64x50x768xf32>
      %303 = stablehlo.add %299, %302 : tensor<64x50x768xf32>
      %304 = stablehlo.reduce(%303 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %305 = stablehlo.multiply %304, %284 : tensor<64x50xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<64x50x768xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<64x50x768xf32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %310 = stablehlo.multiply %309, %284 : tensor<64x50xf32>
      %311 = stablehlo.add %310, %291 : tensor<64x50xf32>
      %312 = stablehlo.rsqrt %311 : tensor<64x50xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %314 = stablehlo.multiply %307, %313 : tensor<64x50x768xf32>
      mhlo.return %314 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<64x768x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %23 = "mhlo.fusion"(%22, %arg19, %arg20, %arg15) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %24 = "mhlo.fusion"(%23, %arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %25 = mhlo.bitcast %24 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %26 = "mhlo.fusion"(%25, %arg17) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %27 = "mhlo.fusion"(%26, %arg18, %arg12, %20, %arg4, %arg3, %4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<64x768x50xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg207 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg207, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      %296 = stablehlo.broadcast_in_dim %arg206, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<64x50x768xf32>
      %298 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.add %297, %298 : tensor<64x50x768xf32>
      %300 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %301 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %302 = stablehlo.add %300, %301 : tensor<64x50x768xf32>
      %303 = stablehlo.add %299, %302 : tensor<64x50x768xf32>
      %304 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %305 = stablehlo.add %304, %arg201 : tensor<3200x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %307 = stablehlo.add %303, %306 : tensor<64x50x768xf32>
      mhlo.return %307 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<64x768x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %28 = "mhlo.fusion"(%27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %29 = mhlo.bitcast %28 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %30 = "mhlo.fusion"(%29, %arg29, %arg30, %arg25, %arg21, %arg23) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %31 = "mhlo.fusion"(%30, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %32 = mhlo.bitcast %31 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %33 = "mhlo.fusion"(%30, %arg22) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %34 = mhlo.bitcast %33 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %35 = "mhlo.fusion"(%32, %34) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %36 = "mhlo.fusion"(%35) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %37 = mhlo.bitcast %36 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %38 = "mhlo.fusion"(%30, %arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %39 = mhlo.bitcast %38 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %40 = "mhlo.fusion"(%37, %39) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %41 = "mhlo.fusion"(%40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %43 = "mhlo.fusion"(%42, %arg27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %44 = "mhlo.fusion"(%27, %arg28, %43) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %45 = mhlo.bitcast %44 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %46 = "mhlo.fusion"(%45, %arg35, %arg36, %arg31) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %47 = "mhlo.fusion"(%46, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %48 = mhlo.bitcast %47 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %49 = "mhlo.fusion"(%48, %arg33) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %50 = "mhlo.fusion"(%49, %arg34, %27, %arg28, %43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %51 = "mhlo.fusion"(%50) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %52 = mhlo.bitcast %51 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %53 = "mhlo.fusion"(%52, %arg45, %arg46, %arg41, %arg37, %arg39) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %54 = "mhlo.fusion"(%53, %arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %55 = mhlo.bitcast %54 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %56 = "mhlo.fusion"(%53, %arg38) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %57 = mhlo.bitcast %56 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %58 = "mhlo.fusion"(%55, %57) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %59 = "mhlo.fusion"(%58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %60 = mhlo.bitcast %59 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %61 = "mhlo.fusion"(%53, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %62 = mhlo.bitcast %61 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %63 = "mhlo.fusion"(%60, %62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %64 = "mhlo.fusion"(%63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %66 = "mhlo.fusion"(%65, %arg43) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %67 = "mhlo.fusion"(%50, %arg44, %66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %68 = mhlo.bitcast %67 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %69 = "mhlo.fusion"(%68, %arg51, %arg52, %arg47) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %70 = "mhlo.fusion"(%69, %arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %71 = mhlo.bitcast %70 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %72 = "mhlo.fusion"(%71, %arg49) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %73 = "mhlo.fusion"(%72, %arg50, %50, %arg44, %66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %74 = "mhlo.fusion"(%73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %75 = mhlo.bitcast %74 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %76 = "mhlo.fusion"(%75, %arg61, %arg62, %arg57, %arg53, %arg55) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %77 = "mhlo.fusion"(%76, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %78 = mhlo.bitcast %77 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %79 = "mhlo.fusion"(%76, %arg54) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %81 = "mhlo.fusion"(%78, %80) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %82 = "mhlo.fusion"(%81) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %83 = mhlo.bitcast %82 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %84 = "mhlo.fusion"(%76, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %85 = mhlo.bitcast %84 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %86 = "mhlo.fusion"(%83, %85) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %87 = "mhlo.fusion"(%86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %88 = mhlo.bitcast %87 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %89 = "mhlo.fusion"(%88, %arg59) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %90 = "mhlo.fusion"(%73, %arg60, %89) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %91 = mhlo.bitcast %90 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %92 = "mhlo.fusion"(%91, %arg67, %arg68, %arg63) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %93 = "mhlo.fusion"(%92, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %95 = "mhlo.fusion"(%94, %arg65) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %96 = "mhlo.fusion"(%95, %arg66, %73, %arg60, %89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %97 = "mhlo.fusion"(%96) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %98 = mhlo.bitcast %97 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %99 = "mhlo.fusion"(%98, %arg77, %arg78, %arg73, %arg69, %arg71) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %100 = "mhlo.fusion"(%99, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %101 = mhlo.bitcast %100 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %102 = "mhlo.fusion"(%99, %arg70) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %103 = mhlo.bitcast %102 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %104 = "mhlo.fusion"(%101, %103) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %105 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %106 = mhlo.bitcast %105 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %107 = "mhlo.fusion"(%99, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %108 = mhlo.bitcast %107 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %109 = "mhlo.fusion"(%106, %108) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %110 = "mhlo.fusion"(%109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %111 = mhlo.bitcast %110 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %112 = "mhlo.fusion"(%111, %arg75) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %113 = "mhlo.fusion"(%96, %arg76, %112) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %114 = mhlo.bitcast %113 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %115 = "mhlo.fusion"(%114, %arg83, %arg84, %arg79) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %116 = "mhlo.fusion"(%115, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %117 = mhlo.bitcast %116 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %118 = "mhlo.fusion"(%117, %arg81) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %119 = "mhlo.fusion"(%118, %arg82, %96, %arg76, %112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %120 = "mhlo.fusion"(%119) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %121 = mhlo.bitcast %120 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %122 = "mhlo.fusion"(%121, %arg93, %arg94, %arg89, %arg85, %arg87) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %123 = "mhlo.fusion"(%122, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %124 = mhlo.bitcast %123 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %125 = "mhlo.fusion"(%122, %arg86) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %126 = mhlo.bitcast %125 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %127 = "mhlo.fusion"(%124, %126) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %128 = "mhlo.fusion"(%127) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %129 = mhlo.bitcast %128 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %130 = "mhlo.fusion"(%122, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %131 = mhlo.bitcast %130 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %132 = "mhlo.fusion"(%129, %131) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %133 = "mhlo.fusion"(%132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %134 = mhlo.bitcast %133 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %135 = "mhlo.fusion"(%134, %arg91) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %136 = "mhlo.fusion"(%119, %arg92, %135) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %137 = mhlo.bitcast %136 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %138 = "mhlo.fusion"(%137, %arg99, %arg100, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %139 = "mhlo.fusion"(%138, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %140 = mhlo.bitcast %139 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %141 = "mhlo.fusion"(%140, %arg97) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %142 = "mhlo.fusion"(%141, %arg98, %119, %arg92, %135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %144 = mhlo.bitcast %143 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %145 = "mhlo.fusion"(%144, %arg109, %arg110, %arg105, %arg101, %arg103) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %146 = "mhlo.fusion"(%145, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %147 = mhlo.bitcast %146 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %148 = "mhlo.fusion"(%145, %arg102) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %149 = mhlo.bitcast %148 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %150 = "mhlo.fusion"(%147, %149) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %151 = "mhlo.fusion"(%150) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %152 = mhlo.bitcast %151 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %153 = "mhlo.fusion"(%145, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %154 = mhlo.bitcast %153 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %155 = "mhlo.fusion"(%152, %154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %156 = "mhlo.fusion"(%155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %158 = "mhlo.fusion"(%157, %arg107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %159 = "mhlo.fusion"(%142, %arg108, %158) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %160 = mhlo.bitcast %159 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %161 = "mhlo.fusion"(%160, %arg115, %arg116, %arg111) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %162 = "mhlo.fusion"(%161, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %163 = mhlo.bitcast %162 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %164 = "mhlo.fusion"(%163, %arg113) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %165 = "mhlo.fusion"(%164, %arg114, %142, %arg108, %158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %166 = "mhlo.fusion"(%165) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %167 = mhlo.bitcast %166 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %168 = "mhlo.fusion"(%167, %arg125, %arg126, %arg121, %arg117, %arg119) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %169 = "mhlo.fusion"(%168, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %170 = mhlo.bitcast %169 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %171 = "mhlo.fusion"(%168, %arg118) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %172 = mhlo.bitcast %171 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %173 = "mhlo.fusion"(%170, %172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %174 = "mhlo.fusion"(%173) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %175 = mhlo.bitcast %174 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %176 = "mhlo.fusion"(%168, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %177 = mhlo.bitcast %176 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %178 = "mhlo.fusion"(%175, %177) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %179 = "mhlo.fusion"(%178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %180 = mhlo.bitcast %179 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %181 = "mhlo.fusion"(%180, %arg123) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %182 = "mhlo.fusion"(%165, %arg124, %181) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %183 = mhlo.bitcast %182 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %184 = "mhlo.fusion"(%183, %arg131, %arg132, %arg127) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %185 = "mhlo.fusion"(%184, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %187 = "mhlo.fusion"(%186, %arg129) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %188 = "mhlo.fusion"(%187, %arg130, %165, %arg124, %181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %189 = "mhlo.fusion"(%188) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %190 = mhlo.bitcast %189 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %191 = "mhlo.fusion"(%190, %arg141, %arg142, %arg137, %arg133, %arg135) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %192 = "mhlo.fusion"(%191, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %193 = mhlo.bitcast %192 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %194 = "mhlo.fusion"(%191, %arg134) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %195 = mhlo.bitcast %194 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %196 = "mhlo.fusion"(%193, %195) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %197 = "mhlo.fusion"(%196) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %198 = mhlo.bitcast %197 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %199 = "mhlo.fusion"(%191, %arg136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %200 = mhlo.bitcast %199 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %201 = "mhlo.fusion"(%198, %200) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %202 = "mhlo.fusion"(%201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %203 = mhlo.bitcast %202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %204 = "mhlo.fusion"(%203, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %205 = "mhlo.fusion"(%188, %arg140, %204) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %206 = mhlo.bitcast %205 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %207 = "mhlo.fusion"(%206, %arg147, %arg148, %arg143) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %208 = "mhlo.fusion"(%207, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %209 = mhlo.bitcast %208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %210 = "mhlo.fusion"(%209, %arg145) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %211 = "mhlo.fusion"(%210, %arg146, %188, %arg140, %204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %212 = "mhlo.fusion"(%211) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %213 = mhlo.bitcast %212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %214 = "mhlo.fusion"(%213, %arg157, %arg158, %arg153, %arg149, %arg151) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %215 = "mhlo.fusion"(%214, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %216 = mhlo.bitcast %215 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %217 = "mhlo.fusion"(%214, %arg150) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %219 = "mhlo.fusion"(%216, %218) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %221 = mhlo.bitcast %220 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %222 = "mhlo.fusion"(%214, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %223 = mhlo.bitcast %222 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %224 = "mhlo.fusion"(%221, %223) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %225 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %226 = mhlo.bitcast %225 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %227 = "mhlo.fusion"(%226, %arg155) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %228 = "mhlo.fusion"(%211, %arg156, %227) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %230 = "mhlo.fusion"(%229, %arg163, %arg164, %arg159) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %231 = "mhlo.fusion"(%230, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %232 = mhlo.bitcast %231 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %233 = "mhlo.fusion"(%232, %arg161) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %234 = "mhlo.fusion"(%233, %arg162, %211, %arg156, %227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %235 = "mhlo.fusion"(%234) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %236 = mhlo.bitcast %235 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %237 = "mhlo.fusion"(%236, %arg173, %arg174, %arg169, %arg165, %arg167) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %238 = "mhlo.fusion"(%237, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %239 = mhlo.bitcast %238 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %240 = "mhlo.fusion"(%237, %arg166) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %241 = mhlo.bitcast %240 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %242 = "mhlo.fusion"(%239, %241) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %243 = "mhlo.fusion"(%242) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %244 = mhlo.bitcast %243 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %245 = "mhlo.fusion"(%237, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %246 = mhlo.bitcast %245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %247 = "mhlo.fusion"(%244, %246) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %248 = "mhlo.fusion"(%247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %249 = mhlo.bitcast %248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %250 = "mhlo.fusion"(%249, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %251 = "mhlo.fusion"(%234, %arg172, %250) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %252 = mhlo.bitcast %251 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %253 = "mhlo.fusion"(%252, %arg179, %arg180, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %254 = "mhlo.fusion"(%253, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %255 = mhlo.bitcast %254 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %256 = "mhlo.fusion"(%255, %arg177) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %257 = "mhlo.fusion"(%256, %arg178, %234, %arg172, %250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      mhlo.return %290 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %258 = "mhlo.fusion"(%257) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<64x50xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<64x50x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<64x50x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %292 = stablehlo.add %290, %291 : tensor<64x50xf32>
      %293 = stablehlo.rsqrt %292 : tensor<64x50xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<64x50x768xf32>
      mhlo.return %295 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>) -> tensor<64x50x768xf32>
    %259 = mhlo.bitcast %258 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %260 = "mhlo.fusion"(%259, %arg189, %arg190, %arg185, %arg181, %arg183) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<768x2304xf32>) -> tensor<3200x2304xf32>
      mhlo.return %289 : tensor<3200x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3200x2304xf32>
    %261 = "mhlo.fusion"(%260, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 0:768] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %289 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %262 = mhlo.bitcast %261 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %263 = "mhlo.fusion"(%260, %arg182) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 768:1536] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3200x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3200x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<64x50x12x64xf32>) -> tensor<64x12x64x50xf32>
      mhlo.return %289 : tensor<64x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x64x50xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x64x50xf32>) -> tensor<768x64x50xf32>
    %265 = "mhlo.fusion"(%262, %264) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x64xf32>, %arg202: tensor<768x64x50xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
      mhlo.return %283 : tensor<768x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x64xf32>, tensor<768x64x50xf32>) -> tensor<768x50x50xf32>
    %266 = "mhlo.fusion"(%265) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<64x12x50x50xf32>, tensor<64x12x50x50xf32>) -> tensor<64x12x50x50xi1>
      %286 = stablehlo.not %285 : tensor<64x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<64x12x50x50xi1>, tensor<i1>) -> tensor<64x12x50xi1>
      %288 = stablehlo.not %287 : tensor<64x12x50xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<64x12x50xi1>) -> tensor<64x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x12x50x50xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<768x50x50xf32>, tensor<f32>) -> tensor<768x50xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<64x12x50xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<64x12x50x50xf32>
      %295 = stablehlo.exponential %294 : tensor<64x12x50x50xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<64x12x50x50xf32>, tensor<f32>) -> tensor<64x12x50xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<64x12x50xf32>) -> tensor<64x12x50x50xf32>
      %298 = stablehlo.divide %295, %297 : tensor<64x12x50x50xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<64x12x50x50xi1>, tensor<64x12x50x50xf32>
      mhlo.return %299 : tensor<64x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<768x50x50xf32>) -> tensor<64x12x50x50xf32>
    %267 = mhlo.bitcast %266 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x50xf32>) -> tensor<768x50x50xf32>
    %268 = "mhlo.fusion"(%260, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.slice %arg201 [0:3200, 1536:2304] : (tensor<3200x2304xf32>) -> tensor<3200x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3200x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<64x50x12x64xf32>) -> tensor<64x12x50x64xf32>
      mhlo.return %287 : tensor<64x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x2304xf32>, tensor<768xf32>) -> tensor<64x12x50x64xf32>
    %269 = mhlo.bitcast %268 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x12x50x64xf32>) -> tensor<768x50x64xf32>
    %270 = "mhlo.fusion"(%267, %269) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768x50x50xf32>, %arg202: tensor<768x50x64xf32>):
      %283 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x64xf32>, tensor<768x50x50xf32>) -> tensor<768x64x50xf32>
      mhlo.return %283 : tensor<768x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[768,64,50]{1,2,0}"} : (tensor<768x50x50xf32>, tensor<768x50x64xf32>) -> tensor<768x64x50xf32>
    %271 = "mhlo.fusion"(%270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x64x50xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<768x64x50xf32>) -> tensor<64x12x50x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<64x12x50x64xf32>) -> tensor<64x50x12x64xf32>
      mhlo.return %284 : tensor<64x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<768x64x50xf32>) -> tensor<64x50x12x64xf32>
    %272 = mhlo.bitcast %271 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x50x12x64xf32>) -> tensor<64x50x768xf32>
    %273 = "mhlo.fusion"(%272, %arg187) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<64x768x768xf32>
      %285 = stablehlo.dot_general %284, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x768x768xf32>, tensor<64x50x768xf32>) -> tensor<64x768x50xf32>
      mhlo.return %285 : tensor<64x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,768,50]{1,2,0}"} : (tensor<64x50x768xf32>, tensor<768x768xf32>) -> tensor<64x768x50xf32>
    %274 = "mhlo.fusion"(%257, %arg188, %273) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg201, %285 : tensor<64x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%286 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<64x50xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %291 = stablehlo.subtract %286, %290 : tensor<64x50x768xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<64x50x768xf32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x50x768xf32>, tensor<f32>) -> tensor<64x50xf32>
      %294 = stablehlo.multiply %293, %288 : tensor<64x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x50xf32>
      %296 = stablehlo.add %294, %295 : tensor<64x50xf32>
      %297 = stablehlo.rsqrt %296 : tensor<64x50xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1] : (tensor<64x50xf32>) -> tensor<64x50x768xf32>
      %299 = stablehlo.multiply %291, %298 : tensor<64x50x768xf32>
      mhlo.return %299 : tensor<64x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
    %275 = mhlo.bitcast %274 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x768xf32>) -> tensor<3200x768xf32>
    %276 = "mhlo.fusion"(%275, %arg195, %arg196, %arg191) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3200x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3200x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
      mhlo.return %287 : tensor<3200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3200x3072xf32>
    %277 = "mhlo.fusion"(%276, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<3072xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3200x3072xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3200x3072xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x3072xf32>) -> tensor<64x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x50x3072xf32>
      %288 = stablehlo.multiply %285, %287 : tensor<64x50x3072xf32>
      %289 = stablehlo.negate %288 : tensor<64x50x3072xf32>
      %290 = stablehlo.exponential %289 : tensor<64x50x3072xf32>
      %291 = stablehlo.add %286, %290 : tensor<64x50x3072xf32>
      %292 = stablehlo.divide %286, %291 : tensor<64x50x3072xf32>
      %293 = stablehlo.multiply %285, %292 : tensor<64x50x3072xf32>
      mhlo.return %293 : tensor<64x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<3072xf32>) -> tensor<64x50x3072xf32>
    %278 = mhlo.bitcast %277 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x50x3072xf32>) -> tensor<3200x3072xf32>
    %279 = "mhlo.fusion"(%278, %arg193) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
      mhlo.return %283 : tensor<3200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x3072xf32>, tensor<768x3072xf32>) -> tensor<3200x768xf32>
    %280 = "mhlo.fusion"(%279, %arg194, %257, %arg188, %273) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<64x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<64x768x50xf32>):
      %283 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<64x768x50xf32>) -> tensor<64x50x768xf32>
      %284 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<64x50x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<64x50x768xf32>
      %286 = stablehlo.add %arg203, %285 : tensor<64x50x768xf32>
      %287 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3200x768xf32>
      %288 = stablehlo.add %287, %arg201 : tensor<3200x768xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3200x768xf32>) -> tensor<64x50x768xf32>
      %290 = stablehlo.add %286, %289 : tensor<64x50x768xf32>
      %291 = stablehlo.slice %290 [0:64, 1:50, 0:768] : (tensor<64x50x768xf32>) -> tensor<64x49x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x49x768xf32>, tensor<f32>) -> tensor<64x768xf32>
      mhlo.return %292 : tensor<64x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3200x768xf32>, tensor<768xf32>, tensor<64x50x768xf32>, tensor<768xf32>, tensor<64x768x50xf32>) -> tensor<64x768xf32>
    %281 = "mhlo.fusion"(%280, %arg197) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<64x768xf32>, %arg202: tensor<2x768xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<64x768xf32>
      %285 = stablehlo.dot_general %284, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x768xf32>, tensor<2x768xf32>) -> tensor<64x2xf32>
      mhlo.return %285 : tensor<64x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x768xf32>, tensor<2x768xf32>) -> tensor<64x2xf32>
    %282 = "mhlo.fusion"(%281, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<64x2xf32>, %arg202: tensor<2xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<2xf32>) -> tensor<64x2xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<64x2xf32>
      mhlo.return %284 : tensor<64x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2xf32>, tensor<2xf32>) -> tensor<64x2xf32>
    return %282 : tensor<64x2xf32>
  }
}
