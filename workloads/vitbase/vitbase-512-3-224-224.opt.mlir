module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.99_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.97_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.98_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.101_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.102_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.107_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.105_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.106_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.109_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.110_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.115_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.113_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.116_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.114_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.117_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.118_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.123_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.121_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.124_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.122_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.125_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.131_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.129_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.130_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.136_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.139_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.137_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.140_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.138_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.141_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.142_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.143_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.144_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.147_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.145_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.148_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.146_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.149_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.150_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.151_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.152_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.155_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.153_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.156_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.154_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.157_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.158_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.159_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.160_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.163_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.161_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.164_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.162_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.165_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.166_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.167_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.168_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.171_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.169_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.172_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.170_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.173_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.174_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.175_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.176_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.179_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.177_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.180_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.178_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.181_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.182_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.183_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.184_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.187_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.185_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.188_computation(%arg0: tensor<6144x197x64xf32>, %arg1: tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    return %0 : tensor<6144x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.186_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    return %4 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.189_computation(%arg0: tensor<6144x197x197xf32>, %arg1: tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    return %0 : tensor<6144x197x64xf32>
  }
  func.func private @gemm_fusion_dot_general.190_computation(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
    return %2 : tensor<512x197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.191_computation(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<100864x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<100864x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<100864x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    return %4 : tensor<100864x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.192_computation(%arg0: tensor<100864x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<100864x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    return %0 : tensor<100864x768xf32>
  }
  func.func private @gemm_fusion_dot_general.193_computation(%arg0: tensor<512x768xf32>, %arg1: tensor<1000x768xf32>) -> tensor<512x1000xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x768xf32>, tensor<1000x768xf32>) -> tensor<512x1000xf32>
    return %0 : tensor<512x1000xf32>
  }
  func.func private @region_0.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.64(%arg0: tensor<512x197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<100864x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<512x197x768xf32>, %arg5: tensor<512x197x768xf32>, %arg6: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %3 = stablehlo.add %2, %arg2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %5 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %6 = stablehlo.add %arg5, %5 : tensor<512x197x768xf32>
    %7 = stablehlo.add %6, %arg4 : tensor<512x197x768xf32>
    %8 = stablehlo.add %4, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.add %1, %8 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %14 = stablehlo.subtract %9, %13 : tensor<512x197x768xf32>
    %15 = stablehlo.multiply %14, %14 : tensor<512x197x768xf32>
    %16 = stablehlo.reduce(%15 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %17 = stablehlo.multiply %16, %11 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197xf32>
    %20 = stablehlo.rsqrt %19 : tensor<512x197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %22 = stablehlo.multiply %14, %21 : tensor<512x197x768xf32>
    return %22 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.67(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<512x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<512x197x768xf32>
    %13 = stablehlo.reduce(%12 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %14 = stablehlo.multiply %13, %8 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197xf32>
    %17 = stablehlo.rsqrt %16 : tensor<512x197xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %19 = stablehlo.multiply %11, %18 : tensor<512x197x768xf32>
    return %19 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.69(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.72(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.74(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.77(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.79(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.82(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.84(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.87(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.89(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.92(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.94(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.97(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.99(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.102(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.104(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.107(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.109(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.112(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.114(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.117(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.119(%arg0: tensor<512x197x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x197x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<512x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x197x768xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %10 = stablehlo.multiply %9, %4 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %12 = stablehlo.add %10, %11 : tensor<512x197xf32>
    %13 = stablehlo.rsqrt %12 : tensor<512x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<512x197x768xf32>
    return %15 : tensor<512x197x768xf32>
  }
  func.func private @region_0.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.122(%arg0: tensor<512x197x768xf32>) -> tensor<512x197x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512x197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x197x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x197xf32>
    %10 = stablehlo.rsqrt %9 : tensor<512x197xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<512x197x768xf32>
    return %12 : tensor<512x197x768xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.6(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.7(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.8(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.9(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.10(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_transpose.11(%arg0: tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
    return %1 : tensor<512x197x12x64xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.12(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.13(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.14(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.15(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.16(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.17(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.18(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.19(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.20(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.21(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.22(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.23(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.24(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.25(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.26(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.27(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.28(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.29(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.30(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.31(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.32(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.33(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.34(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.35(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.36(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.37(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.38(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.39(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.40(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.41(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.42(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.43(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.44(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x197x768xf32>, %arg3: tensor<512x197x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg3, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.add %4, %arg2 : tensor<512x197x768xf32>
    %6 = stablehlo.add %2, %5 : tensor<512x197x768xf32>
    return %6 : tensor<512x197x768xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<100864x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100864x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100864x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<100864x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100864x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x197x3072xf32>
    return %98 : tensor<512x197x3072xf32>
  }
  func.func private @fused_transpose.45(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %3 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.46(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
    return %5 : tensor<512x12x64x197xf32>
  }
  func.func private @fused_transpose.47(%arg0: tensor<100864x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<100864x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
    return %5 : tensor<512x12x197x64xf32>
  }
  func.func private @fused_transpose.48(%arg0: tensor<512x768x14x14xf32>) -> tensor<512x196x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x768x14x14xf32>) -> tensor<512x768x196xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<512x768x196xf32>) -> tensor<512x196x768xf32>
    return %1 : tensor<512x196x768xf32>
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
  func.func private @fused_computation.567(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.568(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.569(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.570(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.571(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.572(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.573(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.574(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.575(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.576(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.577(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
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
  func.func private @fused_computation.578(%arg0: tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<512x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
    %5 = stablehlo.not %4 : tensor<512x12x197xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<512x12x197x197xf32>
    %12 = stablehlo.exponential %11 : tensor<512x12x197x197xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
    %15 = stablehlo.divide %12, %14 : tensor<512x12x197x197xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
    return %16 : tensor<512x12x197x197xf32>
  }
  func.func private @region_0.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.891(%arg0: tensor<100864x768xf32>, %arg1: tensor<512x197x768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<100864x768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<512x197x768xf32>, %arg9: tensor<512x197x768xf32>, %arg10: tensor<768xf32>) -> tensor<512x1x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<100864x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %4 = stablehlo.add %arg1, %3 : tensor<512x197x768xf32>
    %5 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
    %6 = stablehlo.add %5, %arg6 : tensor<100864x768xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %9 = stablehlo.add %arg9, %8 : tensor<512x197x768xf32>
    %10 = stablehlo.add %9, %arg8 : tensor<512x197x768xf32>
    %11 = stablehlo.add %7, %10 : tensor<512x197x768xf32>
    %12 = stablehlo.add %4, %11 : tensor<512x197x768xf32>
    %13 = stablehlo.add %2, %12 : tensor<512x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<512x197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %18 = stablehlo.subtract %13, %17 : tensor<512x197x768xf32>
    %19 = stablehlo.multiply %18, %18 : tensor<512x197x768xf32>
    %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
    %21 = stablehlo.multiply %20, %15 : tensor<512x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
    %23 = stablehlo.add %21, %22 : tensor<512x197xf32>
    %24 = stablehlo.rsqrt %23 : tensor<512x197xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
    %26 = stablehlo.multiply %18, %25 : tensor<512x197x768xf32>
    %27 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %28 = stablehlo.multiply %26, %27 : tensor<512x197x768xf32>
    %29 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
    %30 = stablehlo.add %28, %29 : tensor<512x197x768xf32>
    %31 = stablehlo.slice %30 [0:512, 0:1, 0:768] : (tensor<512x197x768xf32>) -> tensor<512x1x768xf32>
    return %31 : tensor<512x1x768xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<512x196x768xf32>, %arg1: tensor<1x197x768xf32>, %arg2: tensor<1x1x768xf32>) -> tensor<512x197x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<768xf32>) -> tensor<512x1x768xf32>
    %2 = stablehlo.concatenate %1, %arg0, dim = 1 : (tensor<512x1x768xf32>, tensor<512x196x768xf32>) -> tensor<512x197x768xf32>
    %3 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<512x197x768xf32>
    %5 = stablehlo.add %2, %4 : tensor<512x197x768xf32>
    return %5 : tensor<512x197x768xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<512x1000xf32>, %arg1: tensor<1000xf32>) -> tensor<512x1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1000xf32>) -> tensor<512x1000xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<512x1000xf32>
    return %1 : tensor<512x1000xf32>
  }
  func.func @main(%arg0: tensor<1x1x768xf32>, %arg1: tensor<1x197x768xf32>, %arg2: tensor<768x3x16x16xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<768x768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<768x768xf32>, %arg9: tensor<768xf32>, %arg10: tensor<768x768xf32>, %arg11: tensor<768xf32>, %arg12: tensor<3072x768xf32>, %arg13: tensor<3072xf32>, %arg14: tensor<768x3072xf32>, %arg15: tensor<768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<768xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768x768xf32>, %arg21: tensor<768xf32>, %arg22: tensor<768x768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768x768xf32>, %arg25: tensor<768xf32>, %arg26: tensor<768x768xf32>, %arg27: tensor<768xf32>, %arg28: tensor<3072x768xf32>, %arg29: tensor<3072xf32>, %arg30: tensor<768x3072xf32>, %arg31: tensor<768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768x768xf32>, %arg37: tensor<768xf32>, %arg38: tensor<768x768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768x768xf32>, %arg41: tensor<768xf32>, %arg42: tensor<768x768xf32>, %arg43: tensor<768xf32>, %arg44: tensor<3072x768xf32>, %arg45: tensor<3072xf32>, %arg46: tensor<768x3072xf32>, %arg47: tensor<768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768x768xf32>, %arg53: tensor<768xf32>, %arg54: tensor<768x768xf32>, %arg55: tensor<768xf32>, %arg56: tensor<768x768xf32>, %arg57: tensor<768xf32>, %arg58: tensor<768x768xf32>, %arg59: tensor<768xf32>, %arg60: tensor<3072x768xf32>, %arg61: tensor<3072xf32>, %arg62: tensor<768x3072xf32>, %arg63: tensor<768xf32>, %arg64: tensor<768xf32>, %arg65: tensor<768xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768x768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768x768xf32>, %arg71: tensor<768xf32>, %arg72: tensor<768x768xf32>, %arg73: tensor<768xf32>, %arg74: tensor<768x768xf32>, %arg75: tensor<768xf32>, %arg76: tensor<3072x768xf32>, %arg77: tensor<3072xf32>, %arg78: tensor<768x3072xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768x768xf32>, %arg87: tensor<768xf32>, %arg88: tensor<768x768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768x768xf32>, %arg91: tensor<768xf32>, %arg92: tensor<3072x768xf32>, %arg93: tensor<3072xf32>, %arg94: tensor<768x3072xf32>, %arg95: tensor<768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768x768xf32>, %arg101: tensor<768xf32>, %arg102: tensor<768x768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768x768xf32>, %arg105: tensor<768xf32>, %arg106: tensor<768x768xf32>, %arg107: tensor<768xf32>, %arg108: tensor<3072x768xf32>, %arg109: tensor<3072xf32>, %arg110: tensor<768x3072xf32>, %arg111: tensor<768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768x768xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768x768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768x768xf32>, %arg121: tensor<768xf32>, %arg122: tensor<768x768xf32>, %arg123: tensor<768xf32>, %arg124: tensor<3072x768xf32>, %arg125: tensor<3072xf32>, %arg126: tensor<768x3072xf32>, %arg127: tensor<768xf32>, %arg128: tensor<768xf32>, %arg129: tensor<768xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768x768xf32>, %arg133: tensor<768xf32>, %arg134: tensor<768x768xf32>, %arg135: tensor<768xf32>, %arg136: tensor<768x768xf32>, %arg137: tensor<768xf32>, %arg138: tensor<768x768xf32>, %arg139: tensor<768xf32>, %arg140: tensor<3072x768xf32>, %arg141: tensor<3072xf32>, %arg142: tensor<768x3072xf32>, %arg143: tensor<768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<768xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768x768xf32>, %arg149: tensor<768xf32>, %arg150: tensor<768x768xf32>, %arg151: tensor<768xf32>, %arg152: tensor<768x768xf32>, %arg153: tensor<768xf32>, %arg154: tensor<768x768xf32>, %arg155: tensor<768xf32>, %arg156: tensor<3072x768xf32>, %arg157: tensor<3072xf32>, %arg158: tensor<768x3072xf32>, %arg159: tensor<768xf32>, %arg160: tensor<768xf32>, %arg161: tensor<768xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768x768xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768x768xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768x768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<768x768xf32>, %arg171: tensor<768xf32>, %arg172: tensor<3072x768xf32>, %arg173: tensor<3072xf32>, %arg174: tensor<768x3072xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768x768xf32>, %arg181: tensor<768xf32>, %arg182: tensor<768x768xf32>, %arg183: tensor<768xf32>, %arg184: tensor<768x768xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768x768xf32>, %arg187: tensor<768xf32>, %arg188: tensor<3072x768xf32>, %arg189: tensor<3072xf32>, %arg190: tensor<768x3072xf32>, %arg191: tensor<768xf32>, %arg192: tensor<768xf32>, %arg193: tensor<768xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<1000x768xf32>, %arg199: tensor<1000xf32>, %arg200: tensor<512x3x224x224xf32>) -> tensor<512x1000xf32> {
    %0 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%arg200, %arg2, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[512,768,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<512x3x224x224xf32>, tensor<768x3x16x16xf32>, tensor<768xf32>) -> tuple<tensor<512x768x14x14xf32>, tensor<0xui8>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<512x768x14x14xf32>, tensor<0xui8>>) -> tensor<512x768x14x14xf32>
    %2 = "mhlo.fusion"(%1) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<512x768x14x14xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x768x14x14xf32>) -> tensor<512x768x196xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1] : (tensor<512x768x196xf32>) -> tensor<512x196x768xf32>
      mhlo.return %307 : tensor<512x196x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x768x14x14xf32>) -> tensor<512x196x768xf32>
    %3 = "mhlo.fusion"(%2, %arg1, %arg0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<512x196x768xf32>, %arg202: tensor<1x197x768xf32>, %arg203: tensor<1x1x768xf32>):
      %306 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [2] : (tensor<768xf32>) -> tensor<512x1x768xf32>
      %308 = stablehlo.concatenate %307, %arg201, dim = 1 : (tensor<512x1x768xf32>, tensor<512x196x768xf32>) -> tensor<512x197x768xf32>
      %309 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<512x197x768xf32>
      %311 = stablehlo.add %308, %310 : tensor<512x197x768xf32>
      mhlo.return %311 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x196x768xf32>, tensor<1x197x768xf32>, tensor<1x1x768xf32>) -> tensor<512x197x768xf32>
    %4 = "mhlo.fusion"(%3) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %6 = "mhlo.fusion"(%5, %arg16, %arg17, %arg4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %7 = "mhlo.fusion"(%6, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %9 = "mhlo.fusion"(%5, %arg16, %arg17, %arg6) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %10 = "mhlo.fusion"(%9, %arg7) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %12 = "mhlo.fusion"(%8, %11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %13 = "mhlo.fusion"(%12) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %15 = "mhlo.fusion"(%5, %arg16, %arg17, %arg8) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %16 = "mhlo.fusion"(%15, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %18 = "mhlo.fusion"(%14, %17) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %19 = "mhlo.fusion"(%18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %21 = "mhlo.fusion"(%20, %arg10) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %22 = "mhlo.fusion"(%3, %21, %arg11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %24 = "mhlo.fusion"(%23, %arg18, %arg19, %arg12) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %25 = "mhlo.fusion"(%24, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %27 = "mhlo.fusion"(%26, %arg14) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %28 = "mhlo.fusion"(%27, %arg15, %3, %21, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %29 = "mhlo.fusion"(%28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %30 = mhlo.bitcast %29 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %31 = "mhlo.fusion"(%30, %arg32, %arg33, %arg20) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %32 = "mhlo.fusion"(%31, %arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %33 = mhlo.bitcast %32 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %34 = "mhlo.fusion"(%30, %arg32, %arg33, %arg22) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %35 = "mhlo.fusion"(%34, %arg23) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %37 = "mhlo.fusion"(%33, %36) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %38 = "mhlo.fusion"(%37) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %39 = mhlo.bitcast %38 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %40 = "mhlo.fusion"(%30, %arg32, %arg33, %arg24) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %41 = "mhlo.fusion"(%40, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %43 = "mhlo.fusion"(%39, %42) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %44 = "mhlo.fusion"(%43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %45 = mhlo.bitcast %44 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %46 = "mhlo.fusion"(%45, %arg26) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %47 = "mhlo.fusion"(%28, %46, %arg27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %48 = mhlo.bitcast %47 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %49 = "mhlo.fusion"(%48, %arg34, %arg35, %arg28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %50 = "mhlo.fusion"(%49, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %51 = mhlo.bitcast %50 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %52 = "mhlo.fusion"(%51, %arg30) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %53 = "mhlo.fusion"(%52, %arg31, %28, %46, %arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %54 = "mhlo.fusion"(%53) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %55 = mhlo.bitcast %54 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %56 = "mhlo.fusion"(%55, %arg48, %arg49, %arg36) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %57 = "mhlo.fusion"(%56, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %58 = mhlo.bitcast %57 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %59 = "mhlo.fusion"(%55, %arg48, %arg49, %arg38) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %60 = "mhlo.fusion"(%59, %arg39) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %62 = "mhlo.fusion"(%58, %61) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %63 = "mhlo.fusion"(%62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %64 = mhlo.bitcast %63 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %65 = "mhlo.fusion"(%55, %arg48, %arg49, %arg40) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %66 = "mhlo.fusion"(%65, %arg41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %67 = mhlo.bitcast %66 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %68 = "mhlo.fusion"(%64, %67) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %69 = "mhlo.fusion"(%68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %70 = mhlo.bitcast %69 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %71 = "mhlo.fusion"(%70, %arg42) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %72 = "mhlo.fusion"(%53, %71, %arg43) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %73 = mhlo.bitcast %72 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %74 = "mhlo.fusion"(%73, %arg50, %arg51, %arg44) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %75 = "mhlo.fusion"(%74, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %76 = mhlo.bitcast %75 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %77 = "mhlo.fusion"(%76, %arg46) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %78 = "mhlo.fusion"(%77, %arg47, %53, %71, %arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %79 = "mhlo.fusion"(%78) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %81 = "mhlo.fusion"(%80, %arg64, %arg65, %arg52) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %82 = "mhlo.fusion"(%81, %arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %83 = mhlo.bitcast %82 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %84 = "mhlo.fusion"(%80, %arg64, %arg65, %arg54) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %85 = "mhlo.fusion"(%84, %arg55) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %86 = mhlo.bitcast %85 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %87 = "mhlo.fusion"(%83, %86) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %88 = "mhlo.fusion"(%87) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %89 = mhlo.bitcast %88 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %90 = "mhlo.fusion"(%80, %arg64, %arg65, %arg56) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %91 = "mhlo.fusion"(%90, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %92 = mhlo.bitcast %91 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %93 = "mhlo.fusion"(%89, %92) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %94 = "mhlo.fusion"(%93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %95 = mhlo.bitcast %94 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %96 = "mhlo.fusion"(%95, %arg58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %97 = "mhlo.fusion"(%78, %96, %arg59) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %98 = mhlo.bitcast %97 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %99 = "mhlo.fusion"(%98, %arg66, %arg67, %arg60) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %100 = "mhlo.fusion"(%99, %arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %101 = mhlo.bitcast %100 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %102 = "mhlo.fusion"(%101, %arg62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %103 = "mhlo.fusion"(%102, %arg63, %78, %96, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %106 = "mhlo.fusion"(%105, %arg80, %arg81, %arg68) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %107 = "mhlo.fusion"(%106, %arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %108 = mhlo.bitcast %107 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %109 = "mhlo.fusion"(%105, %arg80, %arg81, %arg70) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %110 = "mhlo.fusion"(%109, %arg71) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %111 = mhlo.bitcast %110 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %112 = "mhlo.fusion"(%108, %111) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %113 = "mhlo.fusion"(%112) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %114 = mhlo.bitcast %113 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %115 = "mhlo.fusion"(%105, %arg80, %arg81, %arg72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %116 = "mhlo.fusion"(%115, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %117 = mhlo.bitcast %116 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %118 = "mhlo.fusion"(%114, %117) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %119 = "mhlo.fusion"(%118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %120 = mhlo.bitcast %119 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %121 = "mhlo.fusion"(%120, %arg74) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %122 = "mhlo.fusion"(%103, %121, %arg75) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %124 = "mhlo.fusion"(%123, %arg82, %arg83, %arg76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %125 = "mhlo.fusion"(%124, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %126 = mhlo.bitcast %125 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %127 = "mhlo.fusion"(%126, %arg78) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %128 = "mhlo.fusion"(%127, %arg79, %103, %121, %arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %129 = "mhlo.fusion"(%128) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %131 = "mhlo.fusion"(%130, %arg96, %arg97, %arg84) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %132 = "mhlo.fusion"(%131, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %133 = mhlo.bitcast %132 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %134 = "mhlo.fusion"(%130, %arg96, %arg97, %arg86) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %135 = "mhlo.fusion"(%134, %arg87) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %137 = "mhlo.fusion"(%133, %136) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %138 = "mhlo.fusion"(%137) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %139 = mhlo.bitcast %138 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %140 = "mhlo.fusion"(%130, %arg96, %arg97, %arg88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %141 = "mhlo.fusion"(%140, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %142 = mhlo.bitcast %141 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %143 = "mhlo.fusion"(%139, %142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %144 = "mhlo.fusion"(%143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %145 = mhlo.bitcast %144 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %146 = "mhlo.fusion"(%145, %arg90) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %147 = "mhlo.fusion"(%128, %146, %arg91) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %148 = mhlo.bitcast %147 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %149 = "mhlo.fusion"(%148, %arg98, %arg99, %arg92) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %150 = "mhlo.fusion"(%149, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %152 = "mhlo.fusion"(%151, %arg94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %153 = "mhlo.fusion"(%152, %arg95, %128, %146, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %154 = "mhlo.fusion"(%153) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %155 = mhlo.bitcast %154 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %156 = "mhlo.fusion"(%155, %arg112, %arg113, %arg100) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %157 = "mhlo.fusion"(%156, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %158 = mhlo.bitcast %157 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %159 = "mhlo.fusion"(%155, %arg112, %arg113, %arg102) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %160 = "mhlo.fusion"(%159, %arg103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %161 = mhlo.bitcast %160 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %162 = "mhlo.fusion"(%158, %161) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %163 = "mhlo.fusion"(%162) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %164 = mhlo.bitcast %163 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %165 = "mhlo.fusion"(%155, %arg112, %arg113, %arg104) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %166 = "mhlo.fusion"(%165, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %167 = mhlo.bitcast %166 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %168 = "mhlo.fusion"(%164, %167) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %169 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %170 = mhlo.bitcast %169 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %171 = "mhlo.fusion"(%170, %arg106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %172 = "mhlo.fusion"(%153, %171, %arg107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %173 = mhlo.bitcast %172 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %174 = "mhlo.fusion"(%173, %arg114, %arg115, %arg108) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %175 = "mhlo.fusion"(%174, %arg109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %176 = mhlo.bitcast %175 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %177 = "mhlo.fusion"(%176, %arg110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %178 = "mhlo.fusion"(%177, %arg111, %153, %171, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %179 = "mhlo.fusion"(%178) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %180 = mhlo.bitcast %179 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %181 = "mhlo.fusion"(%180, %arg128, %arg129, %arg116) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %182 = "mhlo.fusion"(%181, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %183 = mhlo.bitcast %182 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %184 = "mhlo.fusion"(%180, %arg128, %arg129, %arg118) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %185 = "mhlo.fusion"(%184, %arg119) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %187 = "mhlo.fusion"(%183, %186) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %188 = "mhlo.fusion"(%187) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %189 = mhlo.bitcast %188 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %190 = "mhlo.fusion"(%180, %arg128, %arg129, %arg120) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %191 = "mhlo.fusion"(%190, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %192 = mhlo.bitcast %191 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %193 = "mhlo.fusion"(%189, %192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %194 = "mhlo.fusion"(%193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %195 = mhlo.bitcast %194 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %196 = "mhlo.fusion"(%195, %arg122) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %197 = "mhlo.fusion"(%178, %196, %arg123) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %198 = mhlo.bitcast %197 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %199 = "mhlo.fusion"(%198, %arg130, %arg131, %arg124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %200 = "mhlo.fusion"(%199, %arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %202 = "mhlo.fusion"(%201, %arg126) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %203 = "mhlo.fusion"(%202, %arg127, %178, %196, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %204 = "mhlo.fusion"(%203) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %205 = mhlo.bitcast %204 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %206 = "mhlo.fusion"(%205, %arg144, %arg145, %arg132) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %207 = "mhlo.fusion"(%206, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %208 = mhlo.bitcast %207 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %209 = "mhlo.fusion"(%205, %arg144, %arg145, %arg134) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %210 = "mhlo.fusion"(%209, %arg135) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %211 = mhlo.bitcast %210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %212 = "mhlo.fusion"(%208, %211) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %213 = "mhlo.fusion"(%212) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %214 = mhlo.bitcast %213 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %215 = "mhlo.fusion"(%205, %arg144, %arg145, %arg136) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %216 = "mhlo.fusion"(%215, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %217 = mhlo.bitcast %216 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %218 = "mhlo.fusion"(%214, %217) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %219 = "mhlo.fusion"(%218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %220 = mhlo.bitcast %219 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %221 = "mhlo.fusion"(%220, %arg138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %222 = "mhlo.fusion"(%203, %221, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %223 = mhlo.bitcast %222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %224 = "mhlo.fusion"(%223, %arg146, %arg147, %arg140) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %225 = "mhlo.fusion"(%224, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %226 = mhlo.bitcast %225 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %227 = "mhlo.fusion"(%226, %arg142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %228 = "mhlo.fusion"(%227, %arg143, %203, %221, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %229 = "mhlo.fusion"(%228) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %230 = mhlo.bitcast %229 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %231 = "mhlo.fusion"(%230, %arg160, %arg161, %arg148) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %232 = "mhlo.fusion"(%231, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %233 = mhlo.bitcast %232 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %234 = "mhlo.fusion"(%230, %arg160, %arg161, %arg150) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %235 = "mhlo.fusion"(%234, %arg151) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %236 = mhlo.bitcast %235 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %237 = "mhlo.fusion"(%233, %236) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %238 = "mhlo.fusion"(%237) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %239 = mhlo.bitcast %238 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %240 = "mhlo.fusion"(%230, %arg160, %arg161, %arg152) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %241 = "mhlo.fusion"(%240, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %242 = mhlo.bitcast %241 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %243 = "mhlo.fusion"(%239, %242) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %244 = "mhlo.fusion"(%243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %246 = "mhlo.fusion"(%245, %arg154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %247 = "mhlo.fusion"(%228, %246, %arg155) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %249 = "mhlo.fusion"(%248, %arg162, %arg163, %arg156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %250 = "mhlo.fusion"(%249, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %251 = mhlo.bitcast %250 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %252 = "mhlo.fusion"(%251, %arg158) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %253 = "mhlo.fusion"(%252, %arg159, %228, %246, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      mhlo.return %312 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %254 = "mhlo.fusion"(%253) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<512x197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.subtract %arg201, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.multiply %310, %310 : tensor<512x197x768xf32>
      %312 = stablehlo.reduce(%311 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %313 = stablehlo.multiply %312, %307 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.add %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.rsqrt %315 : tensor<512x197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %318 = stablehlo.multiply %310, %317 : tensor<512x197x768xf32>
      mhlo.return %318 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>) -> tensor<512x197x768xf32>
    %255 = mhlo.bitcast %254 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %256 = "mhlo.fusion"(%255, %arg176, %arg177, %arg164) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %257 = "mhlo.fusion"(%256, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %258 = mhlo.bitcast %257 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %259 = "mhlo.fusion"(%255, %arg176, %arg177, %arg166) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %260 = "mhlo.fusion"(%259, %arg167) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %261 = mhlo.bitcast %260 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %262 = "mhlo.fusion"(%258, %261) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %263 = "mhlo.fusion"(%262) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %265 = "mhlo.fusion"(%255, %arg176, %arg177, %arg168) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %266 = "mhlo.fusion"(%265, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %267 = mhlo.bitcast %266 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %268 = "mhlo.fusion"(%264, %267) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %269 = "mhlo.fusion"(%268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %271 = "mhlo.fusion"(%270, %arg170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %272 = "mhlo.fusion"(%253, %271, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg202, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<512x197xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<512x197x768xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<512x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<512x197x768xf32>
      mhlo.return %321 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %273 = mhlo.bitcast %272 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %274 = "mhlo.fusion"(%273, %arg178, %arg179, %arg172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %275 = "mhlo.fusion"(%274, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %276 = mhlo.bitcast %275 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %277 = "mhlo.fusion"(%276, %arg174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %278 = "mhlo.fusion"(%277, %arg175, %253, %271, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x197x768xf32>, %arg204: tensor<512x197x768xf32>, %arg205: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg204, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.add %310, %arg203 : tensor<512x197x768xf32>
      %312 = stablehlo.add %308, %311 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<512x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %317 = stablehlo.subtract %312, %316 : tensor<512x197x768xf32>
      %318 = stablehlo.multiply %317, %317 : tensor<512x197x768xf32>
      %319 = stablehlo.reduce(%318 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %320 = stablehlo.multiply %319, %314 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197xf32>
      %323 = stablehlo.rsqrt %322 : tensor<512x197xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %325 = stablehlo.multiply %317, %324 : tensor<512x197x768xf32>
      mhlo.return %325 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %279 = mhlo.bitcast %278 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %280 = "mhlo.fusion"(%279, %arg192, %arg193, %arg180) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %281 = "mhlo.fusion"(%280, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %311 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %282 = mhlo.bitcast %281 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %283 = "mhlo.fusion"(%279, %arg192, %arg193, %arg182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %284 = "mhlo.fusion"(%283, %arg183) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x768xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %311 = stablehlo.transpose %310, dims = [0, 2, 3, 1] : (tensor<512x197x12x64xf32>) -> tensor<512x12x64x197xf32>
      mhlo.return %311 : tensor<512x12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x64x197xf32>
    %285 = mhlo.bitcast %284 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x64x197xf32>) -> tensor<6144x64x197xf32>
    %286 = "mhlo.fusion"(%282, %285) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>, %arg202: tensor<6144x64x197xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
      mhlo.return %306 : tensor<6144x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>, tensor<6144x64x197xf32>) -> tensor<6144x197x197xf32>
    %287 = "mhlo.fusion"(%286) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %308 = stablehlo.compare EQ, %306, %307 : (tensor<512x12x197x197xf32>, tensor<512x12x197x197xf32>) -> tensor<512x12x197x197xi1>
      %309 = stablehlo.not %308 : tensor<512x12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %310 = stablehlo.reduce(%309 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<512x12x197x197xi1>, tensor<i1>) -> tensor<512x12x197xi1>
      %311 = stablehlo.not %310 : tensor<512x12x197xi1>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<512x12x197xi1>) -> tensor<512x12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x12x197x197xf32>
      %314 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<6144x197x197xf32>, tensor<f32>) -> tensor<6144x197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6144x197xf32>) -> tensor<512x12x197xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %317 = stablehlo.subtract %306, %316 : tensor<512x12x197x197xf32>
      %318 = stablehlo.exponential %317 : tensor<512x12x197x197xf32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<512x12x197x197xf32>, tensor<f32>) -> tensor<512x12x197xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<512x12x197xf32>) -> tensor<512x12x197x197xf32>
      %321 = stablehlo.divide %318, %320 : tensor<512x12x197x197xf32>
      %322 = stablehlo.select %312, %313, %321 : tensor<512x12x197x197xi1>, tensor<512x12x197x197xf32>
      mhlo.return %322 : tensor<512x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>) -> tensor<512x12x197x197xf32>
    %288 = mhlo.bitcast %287 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x197xf32>) -> tensor<6144x197x197xf32>
    %289 = "mhlo.fusion"(%279, %arg192, %arg193, %arg184) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
      mhlo.return %310 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<100864x768xf32>
    %290 = "mhlo.fusion"(%289, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x12x64xf32>
      %309 = stablehlo.transpose %308, dims = [0, 2, 1, 3] : (tensor<512x197x12x64xf32>) -> tensor<512x12x197x64xf32>
      mhlo.return %309 : tensor<512x12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>) -> tensor<512x12x197x64xf32>
    %291 = mhlo.bitcast %290 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x12x197x64xf32>) -> tensor<6144x197x64xf32>
    %292 = "mhlo.fusion"(%288, %291) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<6144x197x197xf32>, %arg202: tensor<6144x197x64xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
      mhlo.return %306 : tensor<6144x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x197xf32>, tensor<6144x197x64xf32>) -> tensor<6144x197x64xf32>
    %293 = "mhlo.fusion"(%292) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6144x197x64xf32>):
      %306 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6144x197x64xf32>) -> tensor<512x12x197x64xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 1, 3] : (tensor<512x12x197x64xf32>) -> tensor<512x197x12x64xf32>
      mhlo.return %307 : tensor<512x197x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6144x197x64xf32>) -> tensor<512x197x12x64xf32>
    %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x197x12x64xf32>) -> tensor<512x197x768xf32>
    %295 = "mhlo.fusion"(%294, %arg186) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768x768xf32>):
      %306 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<512x768x768xf32>
      %308 = stablehlo.dot_general %arg201, %307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x197x768xf32>, tensor<512x768x768xf32>) -> tensor<512x197x768xf32>
      mhlo.return %308 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768x768xf32>) -> tensor<512x197x768xf32>
    %296 = "mhlo.fusion"(%295, %arg187, %277, %arg175, %253, %271, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<100864x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<512x197x768xf32>, %arg206: tensor<512x197x768xf32>, %arg207: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %307 = stablehlo.add %arg201, %306 : tensor<512x197x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %308, %arg203 : tensor<100864x768xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg207, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %312 = stablehlo.add %arg206, %311 : tensor<512x197x768xf32>
      %313 = stablehlo.add %312, %arg205 : tensor<512x197x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<512x197x768xf32>
      %315 = stablehlo.add %307, %314 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<512x197xf32>
      %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %320 = stablehlo.subtract %315, %319 : tensor<512x197x768xf32>
      %321 = stablehlo.multiply %320, %320 : tensor<512x197x768xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %323 = stablehlo.multiply %322, %317 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197xf32>
      %326 = stablehlo.rsqrt %325 : tensor<512x197xf32>
      %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %328 = stablehlo.multiply %320, %327 : tensor<512x197x768xf32>
      mhlo.return %328 : tensor<512x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x197x768xf32>, tensor<768xf32>, tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x197x768xf32>
    %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x768xf32>) -> tensor<100864x768xf32>
    %298 = "mhlo.fusion"(%297, %arg194, %arg195, %arg188) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.multiply %arg201, %306 : tensor<100864x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<100864x768xf32>
      %310 = stablehlo.dot_general %309, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
      mhlo.return %310 : tensor<100864x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<100864x3072xf32>
    %299 = "mhlo.fusion"(%298, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<100864x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<100864x3072xf32>
      %309 = stablehlo.multiply %306, %308 : tensor<100864x3072xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %311 = stablehlo.negate %308 : tensor<100864x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100864x3072xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<100864x3072xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x3072xf32>) -> tensor<512x197x3072xf32>
      %315 = stablehlo.abs %314 : tensor<512x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %317 = stablehlo.compare LT, %315, %316 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %319 = stablehlo.multiply %314, %314 : tensor<512x197x3072xf32>
      %320 = stablehlo.multiply %318, %319 : tensor<512x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %322 = stablehlo.add %320, %321 : tensor<512x197x3072xf32>
      %323 = stablehlo.multiply %322, %319 : tensor<512x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %325 = stablehlo.add %323, %324 : tensor<512x197x3072xf32>
      %326 = stablehlo.multiply %325, %319 : tensor<512x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %328 = stablehlo.add %326, %327 : tensor<512x197x3072xf32>
      %329 = stablehlo.multiply %328, %319 : tensor<512x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %331 = stablehlo.add %329, %330 : tensor<512x197x3072xf32>
      %332 = stablehlo.multiply %331, %319 : tensor<512x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %334 = stablehlo.add %332, %333 : tensor<512x197x3072xf32>
      %335 = stablehlo.multiply %334, %319 : tensor<512x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %337 = stablehlo.add %335, %336 : tensor<512x197x3072xf32>
      %338 = stablehlo.multiply %314, %337 : tensor<512x197x3072xf32>
      %339 = stablehlo.subtract %316, %338 : tensor<512x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %341 = stablehlo.compare LT, %314, %340 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %343 = stablehlo.negate %319 : tensor<512x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %345 = stablehlo.compare LT, %343, %344 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %346 = stablehlo.exponential %343 : tensor<512x197x3072xf32>
      %347 = stablehlo.divide %316, %315 : tensor<512x197x3072xf32>
      %348 = stablehlo.multiply %346, %347 : tensor<512x197x3072xf32>
      %349 = stablehlo.compare LT, %315, %342 : (tensor<512x197x3072xf32>, tensor<512x197x3072xf32>) -> tensor<512x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %351 = stablehlo.divide %316, %319 : tensor<512x197x3072xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<512x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %353 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %354 = stablehlo.add %352, %353 : tensor<512x197x3072xf32>
      %355 = stablehlo.multiply %354, %351 : tensor<512x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %356 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %357 = stablehlo.add %355, %356 : tensor<512x197x3072xf32>
      %358 = stablehlo.multiply %357, %351 : tensor<512x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %360 = stablehlo.add %358, %359 : tensor<512x197x3072xf32>
      %361 = stablehlo.multiply %360, %351 : tensor<512x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %363 = stablehlo.add %361, %362 : tensor<512x197x3072xf32>
      %364 = stablehlo.multiply %363, %351 : tensor<512x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<512x197x3072xf32>
      %367 = stablehlo.multiply %366, %351 : tensor<512x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<512x197x3072xf32>
      %370 = stablehlo.multiply %369, %351 : tensor<512x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<512x197x3072xf32>
      %373 = stablehlo.multiply %372, %351 : tensor<512x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<512x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %377 = stablehlo.multiply %376, %351 : tensor<512x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %379 = stablehlo.add %377, %378 : tensor<512x197x3072xf32>
      %380 = stablehlo.multiply %379, %351 : tensor<512x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %382 = stablehlo.add %380, %381 : tensor<512x197x3072xf32>
      %383 = stablehlo.multiply %382, %351 : tensor<512x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %385 = stablehlo.add %383, %384 : tensor<512x197x3072xf32>
      %386 = stablehlo.multiply %385, %351 : tensor<512x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %388 = stablehlo.add %386, %387 : tensor<512x197x3072xf32>
      %389 = stablehlo.multiply %388, %351 : tensor<512x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %391 = stablehlo.add %389, %390 : tensor<512x197x3072xf32>
      %392 = stablehlo.multiply %391, %351 : tensor<512x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %394 = stablehlo.add %392, %393 : tensor<512x197x3072xf32>
      %395 = stablehlo.multiply %394, %351 : tensor<512x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x197x3072xf32>
      %397 = stablehlo.add %395, %396 : tensor<512x197x3072xf32>
      %398 = stablehlo.select %349, %375, %397 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %399 = stablehlo.multiply %348, %398 : tensor<512x197x3072xf32>
      %400 = stablehlo.select %345, %340, %399 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %401 = stablehlo.subtract %342, %400 : tensor<512x197x3072xf32>
      %402 = stablehlo.select %341, %401, %400 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %403 = stablehlo.select %317, %339, %402 : tensor<512x197x3072xi1>, tensor<512x197x3072xf32>
      %404 = stablehlo.multiply %310, %403 : tensor<512x197x3072xf32>
      mhlo.return %404 : tensor<512x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<3072xf32>) -> tensor<512x197x3072xf32>
    %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x197x3072xf32>) -> tensor<100864x3072xf32>
    %301 = "mhlo.fusion"(%300, %arg190) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x3072xf32>, %arg202: tensor<768x3072xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
      mhlo.return %306 : tensor<100864x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x3072xf32>, tensor<768x3072xf32>) -> tensor<100864x768xf32>
    %302 = "mhlo.fusion"(%301, %295, %arg187, %arg197, %arg196, %arg191, %277, %arg175, %253, %271, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<100864x768xf32>, %arg202: tensor<512x197x768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<100864x768xf32>, %arg208: tensor<768xf32>, %arg209: tensor<512x197x768xf32>, %arg210: tensor<512x197x768xf32>, %arg211: tensor<768xf32>):
      %306 = stablehlo.broadcast_in_dim %arg206, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<100864x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %310 = stablehlo.add %arg202, %309 : tensor<512x197x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<768xf32>) -> tensor<100864x768xf32>
      %312 = stablehlo.add %311, %arg207 : tensor<100864x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100864x768xf32>) -> tensor<512x197x768xf32>
      %314 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %315 = stablehlo.add %arg210, %314 : tensor<512x197x768xf32>
      %316 = stablehlo.add %315, %arg209 : tensor<512x197x768xf32>
      %317 = stablehlo.add %313, %316 : tensor<512x197x768xf32>
      %318 = stablehlo.add %310, %317 : tensor<512x197x768xf32>
      %319 = stablehlo.add %308, %318 : tensor<512x197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %320 = stablehlo.reduce(%319 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<512x197xf32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %324 = stablehlo.subtract %319, %323 : tensor<512x197x768xf32>
      %325 = stablehlo.multiply %324, %324 : tensor<512x197x768xf32>
      %326 = stablehlo.reduce(%325 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<512x197x768xf32>, tensor<f32>) -> tensor<512x197xf32>
      %327 = stablehlo.multiply %326, %321 : tensor<512x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x197xf32>
      %329 = stablehlo.add %327, %328 : tensor<512x197xf32>
      %330 = stablehlo.rsqrt %329 : tensor<512x197xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<512x197xf32>) -> tensor<512x197x768xf32>
      %332 = stablehlo.multiply %324, %331 : tensor<512x197x768xf32>
      %333 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<512x197x768xf32>
      %335 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<512x197x768xf32>
      %336 = stablehlo.add %334, %335 : tensor<512x197x768xf32>
      %337 = stablehlo.slice %336 [0:512, 0:1, 0:768] : (tensor<512x197x768xf32>) -> tensor<512x1x768xf32>
      mhlo.return %337 : tensor<512x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<100864x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<100864x768xf32>, tensor<768xf32>, tensor<512x197x768xf32>, tensor<512x197x768xf32>, tensor<768xf32>) -> tensor<512x1x768xf32>
    %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x768xf32>) -> tensor<512x768xf32>
    %304 = "mhlo.fusion"(%303, %arg198) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x768xf32>, %arg202: tensor<1000x768xf32>):
      %306 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x768xf32>, tensor<1000x768xf32>) -> tensor<512x1000xf32>
      mhlo.return %306 : tensor<512x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<512x768xf32>, tensor<1000x768xf32>) -> tensor<512x1000xf32>
    %305 = "mhlo.fusion"(%304, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1000xf32>, %arg202: tensor<1000xf32>):
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1000xf32>) -> tensor<512x1000xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<512x1000xf32>
      mhlo.return %307 : tensor<512x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1000xf32>, tensor<1000xf32>) -> tensor<512x1000xf32>
    return %305 : tensor<512x1000xf32>
  }
}
