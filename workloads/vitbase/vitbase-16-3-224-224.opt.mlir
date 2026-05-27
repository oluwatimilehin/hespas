module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.101_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.102_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.3_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.109_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.110_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.5_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.116_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.117_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.118_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.7_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.124_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.125_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.9_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.136_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.11_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.140_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.141_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.142_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.143_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.144_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.13_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.148_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.149_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.150_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.151_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.152_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.15_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.156_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.157_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.158_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.159_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.160_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.17_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.164_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.165_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.166_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.167_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.168_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.19_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.172_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.173_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.174_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.175_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.176_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.21_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.180_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.181_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.182_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.183_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.184_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @gemm_fusion_dot.23_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<3152x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
    return %6 : tensor<3152x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.188_computation(%arg0: tensor<192x197x64xf32>, %arg1: tensor<192x64x197xf32>) -> tensor<192x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    return %0 : tensor<192x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.189_computation(%arg0: tensor<192x197x197xf32>, %arg1: tensor<192x197x64xf32>) -> tensor<192x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    return %0 : tensor<192x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.190_computation(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
    return %2 : tensor<16x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.191_computation(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3152x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<3152x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<3152x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    return %4 : tensor<3152x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.192_computation(%arg0: tensor<3152x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<3152x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    return %0 : tensor<3152x768xf32>
  }
  func.func private @region_0.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.64(%arg0: tensor<16x197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<3152x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<16x197x768xf32>, %arg5: tensor<16x197x768xf32>, %arg6: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %3 = stablehlo.add %2, %arg2 : tensor<3152x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %5 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %6 = stablehlo.add %arg5, %5 : tensor<16x197x768xf32>
    %7 = stablehlo.add %6, %arg4 : tensor<16x197x768xf32>
    %8 = stablehlo.add %4, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.add %1, %8 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %14 = stablehlo.subtract %9, %13 : tensor<16x197x768xf32>
    %15 = stablehlo.multiply %14, %14 : tensor<16x197x768xf32>
    %16 = stablehlo.reduce(%15 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %17 = stablehlo.multiply %16, %11 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197xf32>
    %20 = stablehlo.rsqrt %19 : tensor<16x197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %22 = stablehlo.multiply %14, %21 : tensor<16x197x768xf32>
    return %22 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.67(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<16x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<16x197x768xf32>
    %13 = stablehlo.reduce(%12 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %14 = stablehlo.multiply %13, %8 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197xf32>
    %17 = stablehlo.rsqrt %16 : tensor<16x197xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %19 = stablehlo.multiply %11, %18 : tensor<16x197x768xf32>
    return %19 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.69(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.72(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.74(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.77(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.79(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.82(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.84(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.87(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.89(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.92(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.94(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.97(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.99(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.102(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.104(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.107(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.109(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.112(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.114(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.117(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.119(%arg0: tensor<16x197x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<16x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<16x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<16x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<16x197x768xf32>
    return %15 : tensor<16x197x768xf32>
  }
  func.func private @region_0.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.122(%arg0: tensor<16x197x768xf32>) -> tensor<16x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<16x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<16x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<16x197x768xf32>
    return %12 : tensor<16x197x768xf32>
  }
  func.func private @fused_add(%arg0: tensor<1000x16xf32>, %arg1: tensor<1000xf32>) -> tensor<16x1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1000xf32>) -> tensor<16x1000xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x16xf32>) -> tensor<16x1000xf32>
    %2 = stablehlo.add %0, %1 : tensor<16x1000xf32>
    return %2 : tensor<16x1000xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.6(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.7(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.8(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.9(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.10(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_transpose.11(%arg0: tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
    return %1 : tensor<16x197x12x64xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.12(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.13(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.14(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.15(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.16(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.17(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.18(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.19(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.20(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.21(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.22(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.23(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.24(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.25(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.26(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.27(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.28(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.29(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.30(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.31(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.32(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.33(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.34(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.35(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.36(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.37(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.38(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.39(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.40(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.41(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.42(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.43(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.44(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<3152x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x197x768xf32>, %arg3: tensor<16x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<16x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<16x197x768xf32>
    return %6 : tensor<16x197x768xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<3152x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<16x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3152x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3152x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<3152x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3152x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<16x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<16x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<16x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<16x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<16x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<16x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<16x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<16x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<16x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<16x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<16x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<16x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<16x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<16x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<16x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<16x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<16x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<16x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<16x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<16x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<16x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<16x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<16x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<16x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<16x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<16x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<16x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<16x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<16x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<16x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<16x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<16x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<16x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<16x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<16x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<16x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<16x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<16x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<16x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<16x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<16x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<16x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<16x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<16x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<16x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<16x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<16x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<16x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<16x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<16x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<16x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<16x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<16x197x3072xf32>
    return %98 : tensor<16x197x3072xf32>
  }
  func.func private @fused_transpose.45(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %4 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.46(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
    return %6 : tensor<16x12x64x197xf32>
  }
  func.func private @fused_transpose.47(%arg0: tensor<3152x2304xf32>, %arg1: tensor<768xf32>) -> tensor<16x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.slice %arg0 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<3152x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<3152x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
    return %6 : tensor<16x12x197x64xf32>
  }
  func.func private @fused_transpose.48(%arg0: tensor<16x768x14x14xf32>) -> tensor<16x196x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x768x14x14xf32>) -> tensor<16x768x196xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<16x768x196xf32>) -> tensor<16x196x768xf32>
    return %1 : tensor<16x196x768xf32>
  }
  func.func private @region_0.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.568(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.569(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.1(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.570(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.2(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.571(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.3(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.572(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.4(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.573(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.5(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.574(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.6(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.575(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.7(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.576(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.8(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.577(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.9(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.578(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_3.6.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.8.clone.10(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @fused_computation.579(%arg0: tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<16x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
    %5 = stablehlo.not %4 : tensor<16x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<16x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<16x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<16x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
    return %16 : tensor<16x12x197x197xf32>
  }
  func.func private @region_0.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.904(%arg0: tensor<3152x768xf32>, %arg1: tensor<16x197x768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<3152x768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<16x197x768xf32>, %arg9: tensor<16x197x768xf32>, %arg10: tensor<768xf32>) -> tensor<16x1x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<3152x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %4 = stablehlo.add %arg1, %3 : tensor<16x197x768xf32>
    %5 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
    %6 = stablehlo.add %5, %arg6 : tensor<3152x768xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %9 = stablehlo.add %arg9, %8 : tensor<16x197x768xf32>
    %10 = stablehlo.add %9, %arg8 : tensor<16x197x768xf32>
    %11 = stablehlo.add %7, %10 : tensor<16x197x768xf32>
    %12 = stablehlo.add %4, %11 : tensor<16x197x768xf32>
    %13 = stablehlo.add %2, %12 : tensor<16x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<16x197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %18 = stablehlo.subtract %13, %17 : tensor<16x197x768xf32>
    %19 = stablehlo.multiply %18, %18 : tensor<16x197x768xf32>
    %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
    %21 = stablehlo.multiply %20, %15 : tensor<16x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
    %23 = stablehlo.add %21, %22 : tensor<16x197xf32>
    %24 = stablehlo.rsqrt %23 : tensor<16x197xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
    %26 = stablehlo.multiply %18, %25 : tensor<16x197x768xf32>
    %27 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %28 = stablehlo.multiply %26, %27 : tensor<16x197x768xf32>
    %29 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
    %30 = stablehlo.add %28, %29 : tensor<16x197x768xf32>
    %31 = stablehlo.slice %30 [0:16, 0:1, 0:768] : (tensor<16x197x768xf32>) -> tensor<16x1x768xf32>
    return %31 : tensor<16x1x768xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<16x196x768xf32>, %arg1: tensor<1x197x768xf32>, %arg2: tensor<1x1x768xf32>) -> tensor<16x197x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<768xf32>) -> tensor<16x1x768xf32>
    %2 = stablehlo.concatenate %1, %arg0, dim = 1 : (tensor<16x1x768xf32>, tensor<16x196x768xf32>) -> tensor<16x197x768xf32>
    %3 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<16x197x768xf32>
    %5 = stablehlo.add %2, %4 : tensor<16x197x768xf32>
    return %5 : tensor<16x197x768xf32>
  }
  func.func @main(%arg0: tensor<1x1x768xf32>, %arg1: tensor<1x197x768xf32>, %arg2: tensor<768x3x16x16xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<768x768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<768x768xf32>, %arg9: tensor<768xf32>, %arg10: tensor<768x768xf32>, %arg11: tensor<768xf32>, %arg12: tensor<3072x768xf32>, %arg13: tensor<3072xf32>, %arg14: tensor<768x3072xf32>, %arg15: tensor<768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<768xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768x768xf32>, %arg21: tensor<768xf32>, %arg22: tensor<768x768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768x768xf32>, %arg25: tensor<768xf32>, %arg26: tensor<768x768xf32>, %arg27: tensor<768xf32>, %arg28: tensor<3072x768xf32>, %arg29: tensor<3072xf32>, %arg30: tensor<768x3072xf32>, %arg31: tensor<768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768x768xf32>, %arg37: tensor<768xf32>, %arg38: tensor<768x768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768x768xf32>, %arg41: tensor<768xf32>, %arg42: tensor<768x768xf32>, %arg43: tensor<768xf32>, %arg44: tensor<3072x768xf32>, %arg45: tensor<3072xf32>, %arg46: tensor<768x3072xf32>, %arg47: tensor<768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768x768xf32>, %arg53: tensor<768xf32>, %arg54: tensor<768x768xf32>, %arg55: tensor<768xf32>, %arg56: tensor<768x768xf32>, %arg57: tensor<768xf32>, %arg58: tensor<768x768xf32>, %arg59: tensor<768xf32>, %arg60: tensor<3072x768xf32>, %arg61: tensor<3072xf32>, %arg62: tensor<768x3072xf32>, %arg63: tensor<768xf32>, %arg64: tensor<768xf32>, %arg65: tensor<768xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768x768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768x768xf32>, %arg71: tensor<768xf32>, %arg72: tensor<768x768xf32>, %arg73: tensor<768xf32>, %arg74: tensor<768x768xf32>, %arg75: tensor<768xf32>, %arg76: tensor<3072x768xf32>, %arg77: tensor<3072xf32>, %arg78: tensor<768x3072xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768x768xf32>, %arg87: tensor<768xf32>, %arg88: tensor<768x768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768x768xf32>, %arg91: tensor<768xf32>, %arg92: tensor<3072x768xf32>, %arg93: tensor<3072xf32>, %arg94: tensor<768x3072xf32>, %arg95: tensor<768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768x768xf32>, %arg101: tensor<768xf32>, %arg102: tensor<768x768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768x768xf32>, %arg105: tensor<768xf32>, %arg106: tensor<768x768xf32>, %arg107: tensor<768xf32>, %arg108: tensor<3072x768xf32>, %arg109: tensor<3072xf32>, %arg110: tensor<768x3072xf32>, %arg111: tensor<768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768x768xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768x768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768x768xf32>, %arg121: tensor<768xf32>, %arg122: tensor<768x768xf32>, %arg123: tensor<768xf32>, %arg124: tensor<3072x768xf32>, %arg125: tensor<3072xf32>, %arg126: tensor<768x3072xf32>, %arg127: tensor<768xf32>, %arg128: tensor<768xf32>, %arg129: tensor<768xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768x768xf32>, %arg133: tensor<768xf32>, %arg134: tensor<768x768xf32>, %arg135: tensor<768xf32>, %arg136: tensor<768x768xf32>, %arg137: tensor<768xf32>, %arg138: tensor<768x768xf32>, %arg139: tensor<768xf32>, %arg140: tensor<3072x768xf32>, %arg141: tensor<3072xf32>, %arg142: tensor<768x3072xf32>, %arg143: tensor<768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<768xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768x768xf32>, %arg149: tensor<768xf32>, %arg150: tensor<768x768xf32>, %arg151: tensor<768xf32>, %arg152: tensor<768x768xf32>, %arg153: tensor<768xf32>, %arg154: tensor<768x768xf32>, %arg155: tensor<768xf32>, %arg156: tensor<3072x768xf32>, %arg157: tensor<3072xf32>, %arg158: tensor<768x3072xf32>, %arg159: tensor<768xf32>, %arg160: tensor<768xf32>, %arg161: tensor<768xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768x768xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768x768xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768x768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<768x768xf32>, %arg171: tensor<768xf32>, %arg172: tensor<3072x768xf32>, %arg173: tensor<3072xf32>, %arg174: tensor<768x3072xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768x768xf32>, %arg181: tensor<768xf32>, %arg182: tensor<768x768xf32>, %arg183: tensor<768xf32>, %arg184: tensor<768x768xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768x768xf32>, %arg187: tensor<768xf32>, %arg188: tensor<3072x768xf32>, %arg189: tensor<3072xf32>, %arg190: tensor<768x3072xf32>, %arg191: tensor<768xf32>, %arg192: tensor<768xf32>, %arg193: tensor<768xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<1000x768xf32>, %arg199: tensor<1000xf32>, %arg200: tensor<16x3x224x224xf32>) -> tensor<16x1000xf32> {
    %0 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%arg200, %arg2, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,768,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<16x3x224x224xf32>, tensor<768x3x16x16xf32>, tensor<768xf32>) -> tuple<tensor<16x768x14x14xf32>, tensor<0xui8>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<16x768x14x14xf32>, tensor<0xui8>>) -> tensor<16x768x14x14xf32>
    %2 = "mhlo.fusion"(%1) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<16x768x14x14xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x768x14x14xf32>) -> tensor<16x768x196xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1] : (tensor<16x768x196xf32>) -> tensor<16x196x768xf32>
      mhlo.return %284 : tensor<16x196x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x768x14x14xf32>) -> tensor<16x196x768xf32>
    %3 = "mhlo.fusion"(%2, %arg1, %arg0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<16x196x768xf32>, %arg202: tensor<1x197x768xf32>, %arg203: tensor<1x1x768xf32>):
      %283 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [2] : (tensor<768xf32>) -> tensor<16x1x768xf32>
      %285 = stablehlo.concatenate %284, %arg201, dim = 1 : (tensor<16x1x768xf32>, tensor<16x196x768xf32>) -> tensor<16x197x768xf32>
      %286 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %287 = stablehlo.broadcast_in_dim %286, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<16x197x768xf32>
      %288 = stablehlo.add %285, %287 : tensor<16x197x768xf32>
      mhlo.return %288 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x196x768xf32>, tensor<1x197x768xf32>, tensor<1x1x768xf32>) -> tensor<16x197x768xf32>
    %4 = "mhlo.fusion"(%3) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %6 = "mhlo.fusion"(%5, %arg16, %arg17, %arg4, %arg6, %arg8) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %7 = "mhlo.fusion"(%6, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %9 = "mhlo.fusion"(%6, %arg7) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %11 = "mhlo.fusion"(%8, %10) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %12 = "mhlo.fusion"(%11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %14 = "mhlo.fusion"(%6, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %16 = "mhlo.fusion"(%13, %15) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %17 = "mhlo.fusion"(%16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %18 = mhlo.bitcast %17 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %19 = "mhlo.fusion"(%18, %arg10) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %20 = "mhlo.fusion"(%3, %19, %arg11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %22 = "mhlo.fusion"(%21, %arg18, %arg19, %arg12) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %23 = "mhlo.fusion"(%22, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %24 = mhlo.bitcast %23 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %25 = "mhlo.fusion"(%24, %arg14) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %26 = "mhlo.fusion"(%25, %arg15, %3, %19, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %27 = "mhlo.fusion"(%26) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %28 = mhlo.bitcast %27 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %29 = "mhlo.fusion"(%28, %arg32, %arg33, %arg20, %arg22, %arg24) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %30 = "mhlo.fusion"(%29, %arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %31 = mhlo.bitcast %30 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %32 = "mhlo.fusion"(%29, %arg23) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %33 = mhlo.bitcast %32 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %34 = "mhlo.fusion"(%31, %33) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %35 = "mhlo.fusion"(%34) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %37 = "mhlo.fusion"(%29, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %38 = mhlo.bitcast %37 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %39 = "mhlo.fusion"(%36, %38) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %40 = "mhlo.fusion"(%39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %41 = mhlo.bitcast %40 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %42 = "mhlo.fusion"(%41, %arg26) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %43 = "mhlo.fusion"(%26, %42, %arg27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %44 = mhlo.bitcast %43 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %45 = "mhlo.fusion"(%44, %arg34, %arg35, %arg28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %46 = "mhlo.fusion"(%45, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %47 = mhlo.bitcast %46 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %48 = "mhlo.fusion"(%47, %arg30) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %49 = "mhlo.fusion"(%48, %arg31, %26, %42, %arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %50 = "mhlo.fusion"(%49) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %51 = mhlo.bitcast %50 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %52 = "mhlo.fusion"(%51, %arg48, %arg49, %arg36, %arg38, %arg40) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %53 = "mhlo.fusion"(%52, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %54 = mhlo.bitcast %53 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %55 = "mhlo.fusion"(%52, %arg39) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %56 = mhlo.bitcast %55 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %57 = "mhlo.fusion"(%54, %56) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %58 = "mhlo.fusion"(%57) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %60 = "mhlo.fusion"(%52, %arg41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %62 = "mhlo.fusion"(%59, %61) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %63 = "mhlo.fusion"(%62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %64 = mhlo.bitcast %63 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %65 = "mhlo.fusion"(%64, %arg42) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %66 = "mhlo.fusion"(%49, %65, %arg43) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %67 = mhlo.bitcast %66 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %68 = "mhlo.fusion"(%67, %arg50, %arg51, %arg44) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %69 = "mhlo.fusion"(%68, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %70 = mhlo.bitcast %69 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %71 = "mhlo.fusion"(%70, %arg46) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %72 = "mhlo.fusion"(%71, %arg47, %49, %65, %arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %73 = "mhlo.fusion"(%72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %74 = mhlo.bitcast %73 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %75 = "mhlo.fusion"(%74, %arg64, %arg65, %arg52, %arg54, %arg56) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %76 = "mhlo.fusion"(%75, %arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %77 = mhlo.bitcast %76 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %78 = "mhlo.fusion"(%75, %arg55) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %79 = mhlo.bitcast %78 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %80 = "mhlo.fusion"(%77, %79) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %81 = "mhlo.fusion"(%80) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %83 = "mhlo.fusion"(%75, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %84 = mhlo.bitcast %83 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %85 = "mhlo.fusion"(%82, %84) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %86 = "mhlo.fusion"(%85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %87 = mhlo.bitcast %86 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %88 = "mhlo.fusion"(%87, %arg58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %89 = "mhlo.fusion"(%72, %88, %arg59) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %90 = mhlo.bitcast %89 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %91 = "mhlo.fusion"(%90, %arg66, %arg67, %arg60) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %92 = "mhlo.fusion"(%91, %arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %93 = mhlo.bitcast %92 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %94 = "mhlo.fusion"(%93, %arg62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %95 = "mhlo.fusion"(%94, %arg63, %72, %88, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %96 = "mhlo.fusion"(%95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %98 = "mhlo.fusion"(%97, %arg80, %arg81, %arg68, %arg70, %arg72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %99 = "mhlo.fusion"(%98, %arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %100 = mhlo.bitcast %99 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %101 = "mhlo.fusion"(%98, %arg71) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %102 = mhlo.bitcast %101 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %103 = "mhlo.fusion"(%100, %102) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %106 = "mhlo.fusion"(%98, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %108 = "mhlo.fusion"(%105, %107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %110 = mhlo.bitcast %109 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %111 = "mhlo.fusion"(%110, %arg74) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %112 = "mhlo.fusion"(%95, %111, %arg75) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %113 = mhlo.bitcast %112 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %114 = "mhlo.fusion"(%113, %arg82, %arg83, %arg76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %115 = "mhlo.fusion"(%114, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %116 = mhlo.bitcast %115 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %117 = "mhlo.fusion"(%116, %arg78) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %118 = "mhlo.fusion"(%117, %arg79, %95, %111, %arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %119 = "mhlo.fusion"(%118) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %120 = mhlo.bitcast %119 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %121 = "mhlo.fusion"(%120, %arg96, %arg97, %arg84, %arg86, %arg88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %122 = "mhlo.fusion"(%121, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %124 = "mhlo.fusion"(%121, %arg87) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %125 = mhlo.bitcast %124 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %126 = "mhlo.fusion"(%123, %125) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %127 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %128 = mhlo.bitcast %127 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %129 = "mhlo.fusion"(%121, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %131 = "mhlo.fusion"(%128, %130) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %132 = "mhlo.fusion"(%131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %133 = mhlo.bitcast %132 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %134 = "mhlo.fusion"(%133, %arg90) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %135 = "mhlo.fusion"(%118, %134, %arg91) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %137 = "mhlo.fusion"(%136, %arg98, %arg99, %arg92) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %138 = "mhlo.fusion"(%137, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %139 = mhlo.bitcast %138 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %140 = "mhlo.fusion"(%139, %arg94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %141 = "mhlo.fusion"(%140, %arg95, %118, %134, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %142 = "mhlo.fusion"(%141) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %143 = mhlo.bitcast %142 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %144 = "mhlo.fusion"(%143, %arg112, %arg113, %arg100, %arg102, %arg104) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %145 = "mhlo.fusion"(%144, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %146 = mhlo.bitcast %145 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %147 = "mhlo.fusion"(%144, %arg103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %148 = mhlo.bitcast %147 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %149 = "mhlo.fusion"(%146, %148) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %150 = "mhlo.fusion"(%149) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %152 = "mhlo.fusion"(%144, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %153 = mhlo.bitcast %152 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %154 = "mhlo.fusion"(%151, %153) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %155 = "mhlo.fusion"(%154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %156 = mhlo.bitcast %155 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %157 = "mhlo.fusion"(%156, %arg106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %158 = "mhlo.fusion"(%141, %157, %arg107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %159 = mhlo.bitcast %158 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %160 = "mhlo.fusion"(%159, %arg114, %arg115, %arg108) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %161 = "mhlo.fusion"(%160, %arg109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %162 = mhlo.bitcast %161 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %163 = "mhlo.fusion"(%162, %arg110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %164 = "mhlo.fusion"(%163, %arg111, %141, %157, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %165 = "mhlo.fusion"(%164) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %166 = mhlo.bitcast %165 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %167 = "mhlo.fusion"(%166, %arg128, %arg129, %arg116, %arg118, %arg120) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %168 = "mhlo.fusion"(%167, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %169 = mhlo.bitcast %168 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %170 = "mhlo.fusion"(%167, %arg119) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %171 = mhlo.bitcast %170 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %172 = "mhlo.fusion"(%169, %171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %173 = "mhlo.fusion"(%172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %174 = mhlo.bitcast %173 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %175 = "mhlo.fusion"(%167, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %176 = mhlo.bitcast %175 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %177 = "mhlo.fusion"(%174, %176) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %178 = "mhlo.fusion"(%177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %179 = mhlo.bitcast %178 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %180 = "mhlo.fusion"(%179, %arg122) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %181 = "mhlo.fusion"(%164, %180, %arg123) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %182 = mhlo.bitcast %181 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %183 = "mhlo.fusion"(%182, %arg130, %arg131, %arg124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %184 = "mhlo.fusion"(%183, %arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %185 = mhlo.bitcast %184 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %186 = "mhlo.fusion"(%185, %arg126) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %187 = "mhlo.fusion"(%186, %arg127, %164, %180, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %188 = "mhlo.fusion"(%187) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %189 = mhlo.bitcast %188 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %190 = "mhlo.fusion"(%189, %arg144, %arg145, %arg132, %arg134, %arg136) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %191 = "mhlo.fusion"(%190, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %192 = mhlo.bitcast %191 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %193 = "mhlo.fusion"(%190, %arg135) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %194 = mhlo.bitcast %193 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %195 = "mhlo.fusion"(%192, %194) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %197 = mhlo.bitcast %196 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %198 = "mhlo.fusion"(%190, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %200 = "mhlo.fusion"(%197, %199) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %201 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %202 = mhlo.bitcast %201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %203 = "mhlo.fusion"(%202, %arg138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %204 = "mhlo.fusion"(%187, %203, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %205 = mhlo.bitcast %204 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %206 = "mhlo.fusion"(%205, %arg146, %arg147, %arg140) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %207 = "mhlo.fusion"(%206, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %208 = mhlo.bitcast %207 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %209 = "mhlo.fusion"(%208, %arg142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %210 = "mhlo.fusion"(%209, %arg143, %187, %203, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %212 = mhlo.bitcast %211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %213 = "mhlo.fusion"(%212, %arg160, %arg161, %arg148, %arg150, %arg152) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %214 = "mhlo.fusion"(%213, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %215 = mhlo.bitcast %214 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %216 = "mhlo.fusion"(%213, %arg151) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %217 = mhlo.bitcast %216 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %218 = "mhlo.fusion"(%215, %217) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %219 = "mhlo.fusion"(%218) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %220 = mhlo.bitcast %219 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %221 = "mhlo.fusion"(%213, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %222 = mhlo.bitcast %221 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %223 = "mhlo.fusion"(%220, %222) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %224 = "mhlo.fusion"(%223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %225 = mhlo.bitcast %224 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %226 = "mhlo.fusion"(%225, %arg154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %227 = "mhlo.fusion"(%210, %226, %arg155) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %228 = mhlo.bitcast %227 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %229 = "mhlo.fusion"(%228, %arg162, %arg163, %arg156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %230 = "mhlo.fusion"(%229, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %231 = mhlo.bitcast %230 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %232 = "mhlo.fusion"(%231, %arg158) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %233 = "mhlo.fusion"(%232, %arg159, %210, %226, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      mhlo.return %289 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %234 = "mhlo.fusion"(%233) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<16x197xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.subtract %arg201, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.multiply %287, %287 : tensor<16x197x768xf32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %290 = stablehlo.multiply %289, %284 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.add %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.rsqrt %292 : tensor<16x197xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %295 = stablehlo.multiply %287, %294 : tensor<16x197x768xf32>
      mhlo.return %295 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>) -> tensor<16x197x768xf32>
    %235 = mhlo.bitcast %234 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %236 = "mhlo.fusion"(%235, %arg176, %arg177, %arg164, %arg166, %arg168) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %237 = "mhlo.fusion"(%236, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %238 = mhlo.bitcast %237 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %239 = "mhlo.fusion"(%236, %arg167) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %240 = mhlo.bitcast %239 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %241 = "mhlo.fusion"(%238, %240) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %242 = "mhlo.fusion"(%241) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %243 = mhlo.bitcast %242 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %244 = "mhlo.fusion"(%236, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %246 = "mhlo.fusion"(%243, %245) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %247 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %249 = "mhlo.fusion"(%248, %arg170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %250 = "mhlo.fusion"(%233, %249, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg202, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.reduce(%285 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %288 = stablehlo.multiply %286, %287 : tensor<16x197xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %290 = stablehlo.subtract %285, %289 : tensor<16x197x768xf32>
      %291 = stablehlo.multiply %290, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.reduce(%291 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %293 = stablehlo.multiply %292, %287 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.add %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.rsqrt %295 : tensor<16x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %298 = stablehlo.multiply %290, %297 : tensor<16x197x768xf32>
      mhlo.return %298 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %251 = mhlo.bitcast %250 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %252 = "mhlo.fusion"(%251, %arg178, %arg179, %arg172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %253 = "mhlo.fusion"(%252, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %254 = mhlo.bitcast %253 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %255 = "mhlo.fusion"(%254, %arg174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %256 = "mhlo.fusion"(%255, %arg175, %233, %249, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x197x768xf32>, %arg204: tensor<16x197x768xf32>, %arg205: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg204, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.add %287, %arg203 : tensor<16x197x768xf32>
      %289 = stablehlo.add %285, %288 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.reduce(%289 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %294 = stablehlo.subtract %289, %293 : tensor<16x197x768xf32>
      %295 = stablehlo.multiply %294, %294 : tensor<16x197x768xf32>
      %296 = stablehlo.reduce(%295 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %297 = stablehlo.multiply %296, %291 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197xf32>
      %300 = stablehlo.rsqrt %299 : tensor<16x197xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %302 = stablehlo.multiply %294, %301 : tensor<16x197x768xf32>
      mhlo.return %302 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %257 = mhlo.bitcast %256 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %258 = "mhlo.fusion"(%257, %arg192, %arg193, %arg180, %arg182, %arg184) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %289 = stablehlo.dot %286, %288, precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<768x2304xf32>) -> tensor<3152x2304xf32>
      mhlo.return %289 : tensor<3152x2304xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<3152x2304xf32>
    %259 = "mhlo.fusion"(%258, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 0:768] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %289 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %260 = mhlo.bitcast %259 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %261 = "mhlo.fusion"(%258, %arg183) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 768:1536] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x768xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<3152x768xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %289 = stablehlo.transpose %288, dims = [0, 2, 3, 1] : (tensor<16x197x12x64xf32>) -> tensor<16x12x64x197xf32>
      mhlo.return %289 : tensor<16x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x64x197xf32>
    %262 = mhlo.bitcast %261 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x197xf32>) -> tensor<192x64x197xf32>
    %263 = "mhlo.fusion"(%260, %262) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>, %arg202: tensor<192x64x197xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
      mhlo.return %283 : tensor<192x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>, tensor<192x64x197xf32>) -> tensor<192x197x197xf32>
    %264 = "mhlo.fusion"(%263) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %285 = stablehlo.compare EQ, %283, %284 : (tensor<16x12x197x197xf32>, tensor<16x12x197x197xf32>) -> tensor<16x12x197x197xi1>
      %286 = stablehlo.not %285 : tensor<16x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %287 = stablehlo.reduce(%286 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<16x12x197x197xi1>, tensor<i1>) -> tensor<16x12x197xi1>
      %288 = stablehlo.not %287 : tensor<16x12x197xi1>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2] : (tensor<16x12x197xi1>) -> tensor<16x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x12x197x197xf32>
      %291 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<192x197x197xf32>, tensor<f32>) -> tensor<192x197xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x197xf32>) -> tensor<16x12x197xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %294 = stablehlo.subtract %283, %293 : tensor<16x12x197x197xf32>
      %295 = stablehlo.exponential %294 : tensor<16x12x197x197xf32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<16x12x197x197xf32>, tensor<f32>) -> tensor<16x12x197xf32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2] : (tensor<16x12x197xf32>) -> tensor<16x12x197x197xf32>
      %298 = stablehlo.divide %295, %297 : tensor<16x12x197x197xf32>
      %299 = stablehlo.select %289, %290, %298 : tensor<16x12x197x197xi1>, tensor<16x12x197x197xf32>
      mhlo.return %299 : tensor<16x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>) -> tensor<16x12x197x197xf32>
    %265 = mhlo.bitcast %264 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x197xf32>) -> tensor<192x197x197xf32>
    %266 = "mhlo.fusion"(%258, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x2304xf32>, %arg202: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.slice %arg201 [0:3152, 1536:2304] : (tensor<3152x2304xf32>) -> tensor<3152x768xf32>
      %285 = stablehlo.add %283, %284 : tensor<3152x768xf32>
      %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x12x64xf32>
      %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<16x197x12x64xf32>) -> tensor<16x12x197x64xf32>
      mhlo.return %287 : tensor<16x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x2304xf32>, tensor<768xf32>) -> tensor<16x12x197x64xf32>
    %267 = mhlo.bitcast %266 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x197x64xf32>) -> tensor<192x197x64xf32>
    %268 = "mhlo.fusion"(%265, %267) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<192x197x197xf32>, %arg202: tensor<192x197x64xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
      mhlo.return %283 : tensor<192x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x197xf32>, tensor<192x197x64xf32>) -> tensor<192x197x64xf32>
    %269 = "mhlo.fusion"(%268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x197x64xf32>):
      %283 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x197x64xf32>) -> tensor<16x12x197x64xf32>
      %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<16x12x197x64xf32>) -> tensor<16x197x12x64xf32>
      mhlo.return %284 : tensor<16x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192x197x64xf32>) -> tensor<16x197x12x64xf32>
    %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x197x12x64xf32>) -> tensor<16x197x768xf32>
    %271 = "mhlo.fusion"(%270, %arg186) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768x768xf32>):
      %283 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      %285 = stablehlo.dot_general %arg201, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x197x768xf32>, tensor<16x768x768xf32>) -> tensor<16x197x768xf32>
      mhlo.return %285 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768x768xf32>) -> tensor<16x197x768xf32>
    %272 = "mhlo.fusion"(%271, %arg187, %255, %arg175, %233, %249, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<3152x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<16x197x768xf32>, %arg206: tensor<16x197x768xf32>, %arg207: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %284 = stablehlo.add %arg201, %283 : tensor<16x197x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg204, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %285, %arg203 : tensor<3152x768xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %288 = stablehlo.broadcast_in_dim %arg207, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %289 = stablehlo.add %arg206, %288 : tensor<16x197x768xf32>
      %290 = stablehlo.add %289, %arg205 : tensor<16x197x768xf32>
      %291 = stablehlo.add %287, %290 : tensor<16x197x768xf32>
      %292 = stablehlo.add %284, %291 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<16x197xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %297 = stablehlo.subtract %292, %296 : tensor<16x197x768xf32>
      %298 = stablehlo.multiply %297, %297 : tensor<16x197x768xf32>
      %299 = stablehlo.reduce(%298 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %300 = stablehlo.multiply %299, %294 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197xf32>
      %303 = stablehlo.rsqrt %302 : tensor<16x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %305 = stablehlo.multiply %297, %304 : tensor<16x197x768xf32>
      mhlo.return %305 : tensor<16x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x197x768xf32>, tensor<768xf32>, tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x197x768xf32>
    %273 = mhlo.bitcast %272 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x768xf32>) -> tensor<3152x768xf32>
    %274 = "mhlo.fusion"(%273, %arg194, %arg195, %arg188) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.multiply %arg201, %283 : tensor<3152x768xf32>
      %285 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %286 = stablehlo.add %284, %285 : tensor<3152x768xf32>
      %287 = stablehlo.dot_general %286, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
      mhlo.return %287 : tensor<3152x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3152x3072xf32>
    %275 = "mhlo.fusion"(%274, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %284 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<3152x3072xf32>
      %285 = stablehlo.add %284, %arg201 : tensor<3152x3072xf32>
      %286 = stablehlo.multiply %283, %285 : tensor<3152x3072xf32>
      %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %288 = stablehlo.negate %285 : tensor<3152x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3152x3072xf32>
      %290 = stablehlo.multiply %288, %289 : tensor<3152x3072xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x3072xf32>) -> tensor<16x197x3072xf32>
      %292 = stablehlo.abs %291 : tensor<16x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %294 = stablehlo.compare LT, %292, %293 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %296 = stablehlo.multiply %291, %291 : tensor<16x197x3072xf32>
      %297 = stablehlo.multiply %295, %296 : tensor<16x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %299 = stablehlo.add %297, %298 : tensor<16x197x3072xf32>
      %300 = stablehlo.multiply %299, %296 : tensor<16x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %302 = stablehlo.add %300, %301 : tensor<16x197x3072xf32>
      %303 = stablehlo.multiply %302, %296 : tensor<16x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %305 = stablehlo.add %303, %304 : tensor<16x197x3072xf32>
      %306 = stablehlo.multiply %305, %296 : tensor<16x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<16x197x3072xf32>
      %309 = stablehlo.multiply %308, %296 : tensor<16x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<16x197x3072xf32>
      %312 = stablehlo.multiply %311, %296 : tensor<16x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<16x197x3072xf32>
      %315 = stablehlo.multiply %291, %314 : tensor<16x197x3072xf32>
      %316 = stablehlo.subtract %293, %315 : tensor<16x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %318 = stablehlo.compare LT, %291, %317 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %320 = stablehlo.negate %296 : tensor<16x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %322 = stablehlo.compare LT, %320, %321 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %323 = stablehlo.exponential %320 : tensor<16x197x3072xf32>
      %324 = stablehlo.divide %293, %292 : tensor<16x197x3072xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x197x3072xf32>
      %326 = stablehlo.compare LT, %292, %319 : (tensor<16x197x3072xf32>, tensor<16x197x3072xf32>) -> tensor<16x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %328 = stablehlo.divide %293, %296 : tensor<16x197x3072xf32>
      %329 = stablehlo.multiply %327, %328 : tensor<16x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<16x197x3072xf32>
      %332 = stablehlo.multiply %331, %328 : tensor<16x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<16x197x3072xf32>
      %335 = stablehlo.multiply %334, %328 : tensor<16x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<16x197x3072xf32>
      %338 = stablehlo.multiply %337, %328 : tensor<16x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<16x197x3072xf32>
      %341 = stablehlo.multiply %340, %328 : tensor<16x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<16x197x3072xf32>
      %344 = stablehlo.multiply %343, %328 : tensor<16x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<16x197x3072xf32>
      %347 = stablehlo.multiply %346, %328 : tensor<16x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<16x197x3072xf32>
      %350 = stablehlo.multiply %349, %328 : tensor<16x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<16x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %354 = stablehlo.multiply %353, %328 : tensor<16x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<16x197x3072xf32>
      %357 = stablehlo.multiply %356, %328 : tensor<16x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<16x197x3072xf32>
      %360 = stablehlo.multiply %359, %328 : tensor<16x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<16x197x3072xf32>
      %363 = stablehlo.multiply %362, %328 : tensor<16x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<16x197x3072xf32>
      %366 = stablehlo.multiply %365, %328 : tensor<16x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<16x197x3072xf32>
      %369 = stablehlo.multiply %368, %328 : tensor<16x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<16x197x3072xf32>
      %372 = stablehlo.multiply %371, %328 : tensor<16x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<16x197x3072xf32>
      %375 = stablehlo.select %326, %352, %374 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %376 = stablehlo.multiply %325, %375 : tensor<16x197x3072xf32>
      %377 = stablehlo.select %322, %317, %376 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %378 = stablehlo.subtract %319, %377 : tensor<16x197x3072xf32>
      %379 = stablehlo.select %318, %378, %377 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %380 = stablehlo.select %294, %316, %379 : tensor<16x197x3072xi1>, tensor<16x197x3072xf32>
      %381 = stablehlo.multiply %287, %380 : tensor<16x197x3072xf32>
      mhlo.return %381 : tensor<16x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<3072xf32>) -> tensor<16x197x3072xf32>
    %276 = mhlo.bitcast %275 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x197x3072xf32>) -> tensor<3152x3072xf32>
    %277 = "mhlo.fusion"(%276, %arg190) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x3072xf32>, %arg202: tensor<768x3072xf32>):
      %283 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
      mhlo.return %283 : tensor<3152x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x3072xf32>, tensor<768x3072xf32>) -> tensor<3152x768xf32>
    %278 = "mhlo.fusion"(%277, %271, %arg187, %arg197, %arg196, %arg191, %255, %arg175, %233, %249, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3152x768xf32>, %arg202: tensor<16x197x768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<3152x768xf32>, %arg208: tensor<768xf32>, %arg209: tensor<16x197x768xf32>, %arg210: tensor<16x197x768xf32>, %arg211: tensor<768xf32>):
      %283 = stablehlo.broadcast_in_dim %arg206, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %284 = stablehlo.add %283, %arg201 : tensor<3152x768xf32>
      %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %286 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %287 = stablehlo.add %arg202, %286 : tensor<16x197x768xf32>
      %288 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<768xf32>) -> tensor<3152x768xf32>
      %289 = stablehlo.add %288, %arg207 : tensor<3152x768xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3152x768xf32>) -> tensor<16x197x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %292 = stablehlo.add %arg210, %291 : tensor<16x197x768xf32>
      %293 = stablehlo.add %292, %arg209 : tensor<16x197x768xf32>
      %294 = stablehlo.add %290, %293 : tensor<16x197x768xf32>
      %295 = stablehlo.add %287, %294 : tensor<16x197x768xf32>
      %296 = stablehlo.add %285, %295 : tensor<16x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<16x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %301 = stablehlo.subtract %296, %300 : tensor<16x197x768xf32>
      %302 = stablehlo.multiply %301, %301 : tensor<16x197x768xf32>
      %303 = stablehlo.reduce(%302 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x197x768xf32>, tensor<f32>) -> tensor<16x197xf32>
      %304 = stablehlo.multiply %303, %298 : tensor<16x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<16x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<16x197xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<16x197xf32>) -> tensor<16x197x768xf32>
      %309 = stablehlo.multiply %301, %308 : tensor<16x197x768xf32>
      %310 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<16x197x768xf32>
      %312 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<16x197x768xf32>
      %313 = stablehlo.add %311, %312 : tensor<16x197x768xf32>
      %314 = stablehlo.slice %313 [0:16, 0:1, 0:768] : (tensor<16x197x768xf32>) -> tensor<16x1x768xf32>
      mhlo.return %314 : tensor<16x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<3152x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3152x768xf32>, tensor<768xf32>, tensor<16x197x768xf32>, tensor<16x197x768xf32>, tensor<768xf32>) -> tensor<16x1x768xf32>
    %279 = mhlo.bitcast %278 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x768xf32>) -> tensor<16x768xf32>
    %280 = stablehlo.custom_call @__cublas$gemm(%arg198, %279) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22gemm_backend_config\22:{\22selected_algorithm\22:\22-1\22,\22alpha_real\22:1,\22beta\22:0,\22dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\221\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[],\22rhs_batch_dimensions\22:[]},\22alpha_imag\22:0,\22precision_config\22:{\22operand_precision\22:[\22DEFAULT\22,\22DEFAULT\22],\22algorithm\22:\22ALG_UNSET\22},\22epilogue\22:\22DEFAULT\22,\22lhs_stride\22:\22768000\22,\22rhs_stride\22:\2212288\22,\22grad_x\22:false,\22grad_y\22:false,\22damax_output\22:false,\22autotune_workspace_size\22:\220\22,\22scale_mode\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], xla_shape = "(f32[1000,16]{0,1}, s8[3121152]{0})"} : (tensor<1000x768xf32>, tensor<16x768xf32>) -> tuple<tensor<1000x16xf32>, tensor<3121152xi8>>
    %281 = stablehlo.get_tuple_element %280[0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1000,16]{0,1}"} : (tuple<tensor<1000x16xf32>, tensor<3121152xi8>>) -> tensor<1000x16xf32>
    %282 = "mhlo.fusion"(%281, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1000x16xf32>, %arg202: tensor<1000xf32>):
      %283 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1000xf32>) -> tensor<16x1000xf32>
      %284 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x16xf32>) -> tensor<16x1000xf32>
      %285 = stablehlo.add %283, %284 : tensor<16x1000xf32>
      mhlo.return %285 : tensor<16x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x16xf32>, tensor<1000xf32>) -> tensor<16x1000xf32>
    return %282 : tensor<16x1000xf32>
  }
}
