module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.97_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.98_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.99_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.101_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.102_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.105_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.106_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.107_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.109_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.110_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.113_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.114_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.116_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.115_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.117_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.118_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.121_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.122_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.124_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.123_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.125_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.129_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.130_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.131_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.136_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.137_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.138_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.140_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.139_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.141_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.142_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.143_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.144_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.145_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.146_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.148_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.147_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.149_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.150_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.151_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.152_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.153_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.154_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.156_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.155_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.157_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.158_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.159_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.160_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.161_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.162_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.164_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.163_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.165_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.166_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.167_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.168_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.169_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.170_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.172_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.171_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.173_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.174_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.175_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.176_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.177_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.178_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.180_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.179_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.181_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.182_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.183_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.184_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.185_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.186_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.188_computation(%arg0: tensor<12288x50x64xf32>, %arg1: tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    return %0 : tensor<12288x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.187_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    return %4 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.189_computation(%arg0: tensor<12288x50x50xf32>, %arg1: tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
    return %0 : tensor<12288x64x50xf32>
  }
  func.func private @gemm_fusion_dot_general.190_computation(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<1024x768x50xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
    return %2 : tensor<1024x768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.191_computation(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<51200x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<51200x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<51200x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    return %4 : tensor<51200x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.192_computation(%arg0: tensor<51200x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<51200x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    return %0 : tensor<51200x768xf32>
  }
  func.func private @gemm_fusion_dot_general.193_computation(%arg0: tensor<1024x768xf32>, %arg1: tensor<2x768xf32>) -> tensor<1024x2xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024x768xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x768xf32>, tensor<2x768xf32>) -> tensor<1024x2xf32>
    return %2 : tensor<1024x2xf32>
  }
  func.func private @region_1.4.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.62(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.65(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.67(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.70(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.72(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.75(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.77(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.80(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.82(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.85(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.87(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.90(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.92(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.95(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.97(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.100(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.102(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.105(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.107(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.110(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.112(%arg0: tensor<1024x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<1024x50x768xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<1024x50x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %13 = stablehlo.add %11, %12 : tensor<1024x50xf32>
    %14 = stablehlo.rsqrt %13 : tensor<1024x50xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<1024x50x768xf32>
    return %16 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.115(%arg0: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    return %12 : tensor<1024x50x768xf32>
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
  func.func private @triton_softmax_computation.117(%arg0: tensor<768xf32>, %arg1: tensor<1024x768x50xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg4, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    %13 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1024x50x768xf32>
    %15 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<1024x50x768xf32>
    %17 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %18 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<1024x50x768xf32>
    %20 = stablehlo.add %16, %19 : tensor<1024x50x768xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %22 = stablehlo.multiply %21, %1 : tensor<1024x50xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<1024x50x768xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<1024x50x768xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %27 = stablehlo.multiply %26, %1 : tensor<1024x50xf32>
    %28 = stablehlo.add %27, %8 : tensor<1024x50xf32>
    %29 = stablehlo.rsqrt %28 : tensor<1024x50xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<1024x50x768xf32>
    return %31 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.120(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1024x50x768xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<1024x50x768xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %18 = stablehlo.multiply %17, %1 : tensor<1024x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<1024x50x768xf32>
    %21 = stablehlo.multiply %20, %20 : tensor<1024x50x768xf32>
    %22 = stablehlo.reduce(%21 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %23 = stablehlo.multiply %22, %1 : tensor<1024x50xf32>
    %24 = stablehlo.add %23, %8 : tensor<1024x50xf32>
    %25 = stablehlo.rsqrt %24 : tensor<1024x50xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %27 = stablehlo.multiply %20, %26 : tensor<1024x50x768xf32>
    return %27 : tensor<1024x50x768xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.6(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.7(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.8(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.9(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.10(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
  }
  func.func private @fused_transpose.11(%arg0: tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
    return %1 : tensor<1024x50x12x64xf32>
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
  func.func private @fused_multiply(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.12(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.13(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.14(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.15(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.16(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.17(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.18(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.19(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.20(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.21(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.22(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.23(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.24(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.25(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.26(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.27(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.28(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.29(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.30(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.31(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.32(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.33(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.34(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.35(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.36(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.37(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.38(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.39(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.40(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.41(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    return %7 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.42(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.43(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.44(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<1024x768x50xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg6 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.subtract %arg6, %3 : tensor<1024x50x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1024x50x768xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1024x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
    %9 = stablehlo.add %7, %8 : tensor<1024x50xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1024x50xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<1024x50x768xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1024x50x768xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<1024x50x768xf32>
    %17 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %18 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<1024x50x768xf32>
    %20 = stablehlo.add %16, %19 : tensor<1024x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %22 = stablehlo.add %21, %arg0 : tensor<51200x768xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %24 = stablehlo.add %20, %23 : tensor<1024x50x768xf32>
    return %24 : tensor<1024x50x768xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<51200x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1024x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<1024x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<1024x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<1024x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<1024x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<1024x50x3072xf32>
    return %10 : tensor<1024x50x3072xf32>
  }
  func.func private @fused_transpose.45(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %3 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.46(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
    return %5 : tensor<1024x12x64x50xf32>
  }
  func.func private @fused_transpose.47(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>) -> tensor<1024x12x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<51200x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<51200x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
    return %5 : tensor<1024x12x50x64xf32>
  }
  func.func private @fused_transpose.48(%arg0: tensor<1024x768x7x7xf32>) -> tensor<1024x49x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x768x7x7xf32>) -> tensor<1024x768x49xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<1024x768x49xf32>) -> tensor<1024x49x768xf32>
    return %1 : tensor<1024x49x768xf32>
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
  func.func private @fused_computation.185(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.186(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.187(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.188(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.189(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.190(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.191(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.192(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.193(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.194(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.195(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
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
  func.func private @fused_computation.196(%arg0: tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %5 = stablehlo.not %4 : tensor<1024x12x50xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
    %8 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %11 = stablehlo.subtract %0, %10 : tensor<1024x12x50x50xf32>
    %12 = stablehlo.exponential %11 : tensor<1024x12x50x50xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = stablehlo.divide %12, %14 : tensor<1024x12x50x50xf32>
    %16 = stablehlo.select %6, %7, %15 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
    return %16 : tensor<1024x12x50x50xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<1024x49x768xf32>, %arg1: tensor<1x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<1024x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<1024x1x768xf32>
    %1 = stablehlo.concatenate %0, %arg0, dim = 1 : (tensor<1024x1x768xf32>, tensor<1024x49x768xf32>) -> tensor<1024x50x768xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<1024x50x768xf32>
    %4 = stablehlo.add %1, %3 : tensor<1024x50x768xf32>
    return %4 : tensor<1024x50x768xf32>
  }
  func.func private @region_1.4.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<51200x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1024x50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<1024x768x50xf32>) -> tensor<1024x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x50x768xf32>
    %3 = stablehlo.add %arg2, %2 : tensor<1024x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
    %5 = stablehlo.add %4, %arg0 : tensor<51200x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
    %7 = stablehlo.add %3, %6 : tensor<1024x50x768xf32>
    %8 = stablehlo.slice %7 [0:1024, 1:50, 0:768] : (tensor<1024x50x768xf32>) -> tensor<1024x49x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x49x768xf32>, tensor<f32>) -> tensor<1024x768xf32>
    return %9 : tensor<1024x768xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<1024x2xf32>, %arg1: tensor<2xf32>) -> tensor<1024x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<1024x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1024x2xf32>
    return %1 : tensor<1024x2xf32>
  }
  func.func @main(%arg0: tensor<768xf32>, %arg1: tensor<768x3x32x32xf32>, %arg2: tensor<50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768x768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<3072x768xf32>, %arg16: tensor<3072xf32>, %arg17: tensor<768x3072xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768x768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<3072x768xf32>, %arg32: tensor<3072xf32>, %arg33: tensor<768x3072xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<3072x768xf32>, %arg48: tensor<3072xf32>, %arg49: tensor<768x3072xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768x768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<3072x768xf32>, %arg64: tensor<3072xf32>, %arg65: tensor<768x3072xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768x768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<3072x768xf32>, %arg80: tensor<3072xf32>, %arg81: tensor<768x3072xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<3072x768xf32>, %arg96: tensor<3072xf32>, %arg97: tensor<768x3072xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768x768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<3072x768xf32>, %arg112: tensor<3072xf32>, %arg113: tensor<768x3072xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768x768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<3072x768xf32>, %arg128: tensor<3072xf32>, %arg129: tensor<768x3072xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768xf32>, %arg133: tensor<768x768xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<3072x768xf32>, %arg144: tensor<3072xf32>, %arg145: tensor<768x3072xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768xf32>, %arg149: tensor<768x768xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768x768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<3072x768xf32>, %arg160: tensor<3072xf32>, %arg161: tensor<768x3072xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768xf32>, %arg165: tensor<768x768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<3072x768xf32>, %arg176: tensor<3072xf32>, %arg177: tensor<768x3072xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768xf32>, %arg181: tensor<768x768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768x768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<3072x768xf32>, %arg192: tensor<3072xf32>, %arg193: tensor<768x3072xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<2x768xf32>, %arg198: tensor<2xf32>, %arg199: tensor<1x50xi32>, %arg200: tensor<1024x3x224x224xf32>) -> tensor<1024x2xf32> {
    %0 = stablehlo.custom_call @__cudnn$convForward(%arg200, %arg1) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,768,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x3x224x224xf32>, tensor<768x3x32x32xf32>) -> tuple<tensor<1024x768x7x7xf32>, tensor<0xui8>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<1024x768x7x7xf32>, tensor<0xui8>>) -> tensor<1024x768x7x7xf32>
    %2 = "mhlo.fusion"(%1) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1024x768x7x7xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x768x7x7xf32>) -> tensor<1024x768x49xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1] : (tensor<1024x768x49xf32>) -> tensor<1024x49x768xf32>
      mhlo.return %308 : tensor<1024x49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x768x7x7xf32>) -> tensor<1024x49x768xf32>
    %3 = "mhlo.fusion"(%arg2, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<1x50xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %307 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %308 = stablehlo.compare LT, %arg202, %307 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_0 = stablehlo.constant dense<50> : tensor<i32>
      %309 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %310 = stablehlo.add %arg202, %309 : tensor<1x50xi32>
      %311 = stablehlo.select %308, %310, %arg202 : tensor<1x50xi1>, tensor<1x50xi32>
      %312 = stablehlo.compare GE, %311, %307 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %313 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %314 = stablehlo.compare LE, %311, %313 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %315 = stablehlo.and %312, %314 : tensor<1x50xi1>
      %316 = mhlo.bitcast %315 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %317 = stablehlo.broadcast_in_dim %316, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %318 = mhlo.bitcast %311 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
      %319 = "stablehlo.gather"(%arg201, %318) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[50,1,768]{2,0,1}"} : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
      %320 = mhlo.bitcast %319 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 0, 1]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %322 = stablehlo.select %317, %320, %321 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      mhlo.return %322 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<1x50xi32>) -> tensor<1x50x768xf32>
    %4 = "mhlo.fusion"(%2, %3, %arg0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1024x49x768xf32>, %arg202: tensor<1x50x768xf32>, %arg203: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x1x768xf32>
      %308 = stablehlo.concatenate %307, %arg201, dim = 1 : (tensor<1024x1x768xf32>, tensor<1024x49x768xf32>) -> tensor<1024x50x768xf32>
      %309 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.add %308, %310 : tensor<1024x50x768xf32>
      mhlo.return %311 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x49x768xf32>, tensor<1x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %5 = "mhlo.fusion"(%arg4, %arg3, %4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg203 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg203, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      %320 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<1024x50x768xf32>
      %322 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.add %321, %322 : tensor<1024x50x768xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %325 = stablehlo.multiply %324, %308 : tensor<1024x50xf32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %327 = stablehlo.subtract %323, %326 : tensor<1024x50x768xf32>
      %328 = stablehlo.multiply %327, %327 : tensor<1024x50x768xf32>
      %329 = stablehlo.reduce(%328 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %330 = stablehlo.multiply %329, %308 : tensor<1024x50xf32>
      %331 = stablehlo.add %330, %315 : tensor<1024x50xf32>
      %332 = stablehlo.rsqrt %331 : tensor<1024x50xf32>
      %333 = stablehlo.broadcast_in_dim %332, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %334 = stablehlo.multiply %327, %333 : tensor<1024x50x768xf32>
      mhlo.return %334 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %7 = "mhlo.fusion"(%6, %arg13, %arg14, %arg9) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %8 = "mhlo.fusion"(%7, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %10 = "mhlo.fusion"(%6, %arg13, %arg14, %arg5) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %11 = "mhlo.fusion"(%10, %arg6) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %13 = "mhlo.fusion"(%9, %12) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %14 = "mhlo.fusion"(%13) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %16 = "mhlo.fusion"(%6, %arg13, %arg14, %arg7) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %17 = "mhlo.fusion"(%16, %arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %18 = mhlo.bitcast %17 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %19 = "mhlo.fusion"(%15, %18) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %20 = "mhlo.fusion"(%19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %22 = "mhlo.fusion"(%21, %arg11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %23 = "mhlo.fusion"(%arg12, %22, %arg4, %arg3, %4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<1024x768x50xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg205 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg205, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      %320 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<1024x50x768xf32>
      %322 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.add %321, %322 : tensor<1024x50x768xf32>
      %324 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %325 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %326 = stablehlo.add %324, %325 : tensor<1024x50x768xf32>
      %327 = stablehlo.add %323, %326 : tensor<1024x50x768xf32>
      %328 = stablehlo.reduce(%327 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %329 = stablehlo.multiply %328, %308 : tensor<1024x50xf32>
      %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %331 = stablehlo.subtract %327, %330 : tensor<1024x50x768xf32>
      %332 = stablehlo.multiply %331, %331 : tensor<1024x50x768xf32>
      %333 = stablehlo.reduce(%332 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %334 = stablehlo.multiply %333, %308 : tensor<1024x50xf32>
      %335 = stablehlo.add %334, %315 : tensor<1024x50xf32>
      %336 = stablehlo.rsqrt %335 : tensor<1024x50xf32>
      %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %338 = stablehlo.multiply %331, %337 : tensor<1024x50x768xf32>
      mhlo.return %338 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<1024x768x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %24 = mhlo.bitcast %23 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %25 = "mhlo.fusion"(%24, %arg19, %arg20, %arg15) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %26 = "mhlo.fusion"(%25, %arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %27 = mhlo.bitcast %26 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %28 = "mhlo.fusion"(%27, %arg17) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %29 = "mhlo.fusion"(%28, %arg18, %arg12, %22, %arg4, %arg3, %4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<1024x768x50xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg207 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg207, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      %320 = stablehlo.broadcast_in_dim %arg206, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<1024x50x768xf32>
      %322 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.add %321, %322 : tensor<1024x50x768xf32>
      %324 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %325 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %326 = stablehlo.add %324, %325 : tensor<1024x50x768xf32>
      %327 = stablehlo.add %323, %326 : tensor<1024x50x768xf32>
      %328 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %329 = stablehlo.add %328, %arg201 : tensor<51200x768xf32>
      %330 = mhlo.bitcast %329 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %331 = stablehlo.add %327, %330 : tensor<1024x50x768xf32>
      mhlo.return %331 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %30 = "mhlo.fusion"(%29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %31 = mhlo.bitcast %30 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %32 = "mhlo.fusion"(%31, %arg29, %arg30, %arg25) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %33 = "mhlo.fusion"(%32, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %34 = mhlo.bitcast %33 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %35 = "mhlo.fusion"(%31, %arg29, %arg30, %arg21) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %36 = "mhlo.fusion"(%35, %arg22) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %37 = mhlo.bitcast %36 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %38 = "mhlo.fusion"(%34, %37) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %39 = "mhlo.fusion"(%38) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %40 = mhlo.bitcast %39 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %41 = "mhlo.fusion"(%31, %arg29, %arg30, %arg23) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %42 = "mhlo.fusion"(%41, %arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %43 = mhlo.bitcast %42 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %44 = "mhlo.fusion"(%40, %43) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %45 = "mhlo.fusion"(%44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %46 = mhlo.bitcast %45 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %47 = "mhlo.fusion"(%46, %arg27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %48 = "mhlo.fusion"(%29, %arg28, %47) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %49 = mhlo.bitcast %48 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %50 = "mhlo.fusion"(%49, %arg35, %arg36, %arg31) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %51 = "mhlo.fusion"(%50, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %52 = mhlo.bitcast %51 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %53 = "mhlo.fusion"(%52, %arg33) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %54 = "mhlo.fusion"(%53, %arg34, %29, %arg28, %47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %55 = "mhlo.fusion"(%54) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %56 = mhlo.bitcast %55 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %57 = "mhlo.fusion"(%56, %arg45, %arg46, %arg41) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %58 = "mhlo.fusion"(%57, %arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %60 = "mhlo.fusion"(%56, %arg45, %arg46, %arg37) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %61 = "mhlo.fusion"(%60, %arg38) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %62 = mhlo.bitcast %61 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %63 = "mhlo.fusion"(%59, %62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %64 = "mhlo.fusion"(%63) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %66 = "mhlo.fusion"(%56, %arg45, %arg46, %arg39) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %67 = "mhlo.fusion"(%66, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %68 = mhlo.bitcast %67 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %69 = "mhlo.fusion"(%65, %68) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %70 = "mhlo.fusion"(%69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %71 = mhlo.bitcast %70 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %72 = "mhlo.fusion"(%71, %arg43) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %73 = "mhlo.fusion"(%54, %arg44, %72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %74 = mhlo.bitcast %73 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %75 = "mhlo.fusion"(%74, %arg51, %arg52, %arg47) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %76 = "mhlo.fusion"(%75, %arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %77 = mhlo.bitcast %76 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %78 = "mhlo.fusion"(%77, %arg49) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %79 = "mhlo.fusion"(%78, %arg50, %54, %arg44, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %80 = "mhlo.fusion"(%79) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %81 = mhlo.bitcast %80 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %82 = "mhlo.fusion"(%81, %arg61, %arg62, %arg57) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %83 = "mhlo.fusion"(%82, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %84 = mhlo.bitcast %83 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %85 = "mhlo.fusion"(%81, %arg61, %arg62, %arg53) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %86 = "mhlo.fusion"(%85, %arg54) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %87 = mhlo.bitcast %86 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %88 = "mhlo.fusion"(%84, %87) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %89 = "mhlo.fusion"(%88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %90 = mhlo.bitcast %89 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %91 = "mhlo.fusion"(%81, %arg61, %arg62, %arg55) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %92 = "mhlo.fusion"(%91, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %93 = mhlo.bitcast %92 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %94 = "mhlo.fusion"(%90, %93) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %95 = "mhlo.fusion"(%94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %96 = mhlo.bitcast %95 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %97 = "mhlo.fusion"(%96, %arg59) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %98 = "mhlo.fusion"(%79, %arg60, %97) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %100 = "mhlo.fusion"(%99, %arg67, %arg68, %arg63) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %101 = "mhlo.fusion"(%100, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %102 = mhlo.bitcast %101 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %103 = "mhlo.fusion"(%102, %arg65) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %104 = "mhlo.fusion"(%103, %arg66, %79, %arg60, %97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %105 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %106 = mhlo.bitcast %105 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %107 = "mhlo.fusion"(%106, %arg77, %arg78, %arg73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %108 = "mhlo.fusion"(%107, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %109 = mhlo.bitcast %108 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %110 = "mhlo.fusion"(%106, %arg77, %arg78, %arg69) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %111 = "mhlo.fusion"(%110, %arg70) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %112 = mhlo.bitcast %111 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %113 = "mhlo.fusion"(%109, %112) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %114 = "mhlo.fusion"(%113) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %116 = "mhlo.fusion"(%106, %arg77, %arg78, %arg71) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %117 = "mhlo.fusion"(%116, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %118 = mhlo.bitcast %117 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %119 = "mhlo.fusion"(%115, %118) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %120 = "mhlo.fusion"(%119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %121 = mhlo.bitcast %120 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %122 = "mhlo.fusion"(%121, %arg75) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %123 = "mhlo.fusion"(%104, %arg76, %122) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %124 = mhlo.bitcast %123 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %125 = "mhlo.fusion"(%124, %arg83, %arg84, %arg79) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %126 = "mhlo.fusion"(%125, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %127 = mhlo.bitcast %126 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %128 = "mhlo.fusion"(%127, %arg81) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %129 = "mhlo.fusion"(%128, %arg82, %104, %arg76, %122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %130 = "mhlo.fusion"(%129) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %131 = mhlo.bitcast %130 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %132 = "mhlo.fusion"(%131, %arg93, %arg94, %arg89) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %133 = "mhlo.fusion"(%132, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %134 = mhlo.bitcast %133 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %135 = "mhlo.fusion"(%131, %arg93, %arg94, %arg85) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %136 = "mhlo.fusion"(%135, %arg86) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %137 = mhlo.bitcast %136 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %138 = "mhlo.fusion"(%134, %137) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %139 = "mhlo.fusion"(%138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %140 = mhlo.bitcast %139 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %141 = "mhlo.fusion"(%131, %arg93, %arg94, %arg87) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %142 = "mhlo.fusion"(%141, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %143 = mhlo.bitcast %142 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %144 = "mhlo.fusion"(%140, %143) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %145 = "mhlo.fusion"(%144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %146 = mhlo.bitcast %145 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %147 = "mhlo.fusion"(%146, %arg91) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %148 = "mhlo.fusion"(%129, %arg92, %147) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %149 = mhlo.bitcast %148 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %150 = "mhlo.fusion"(%149, %arg99, %arg100, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %151 = "mhlo.fusion"(%150, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %152 = mhlo.bitcast %151 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %153 = "mhlo.fusion"(%152, %arg97) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %154 = "mhlo.fusion"(%153, %arg98, %129, %arg92, %147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %155 = "mhlo.fusion"(%154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %156 = mhlo.bitcast %155 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %157 = "mhlo.fusion"(%156, %arg109, %arg110, %arg105) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %158 = "mhlo.fusion"(%157, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %159 = mhlo.bitcast %158 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %160 = "mhlo.fusion"(%156, %arg109, %arg110, %arg101) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %161 = "mhlo.fusion"(%160, %arg102) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %162 = mhlo.bitcast %161 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %163 = "mhlo.fusion"(%159, %162) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %165 = mhlo.bitcast %164 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %166 = "mhlo.fusion"(%156, %arg109, %arg110, %arg103) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %167 = "mhlo.fusion"(%166, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %168 = mhlo.bitcast %167 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %169 = "mhlo.fusion"(%165, %168) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %170 = "mhlo.fusion"(%169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %171 = mhlo.bitcast %170 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %172 = "mhlo.fusion"(%171, %arg107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %173 = "mhlo.fusion"(%154, %arg108, %172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %174 = mhlo.bitcast %173 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %175 = "mhlo.fusion"(%174, %arg115, %arg116, %arg111) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %176 = "mhlo.fusion"(%175, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %177 = mhlo.bitcast %176 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %178 = "mhlo.fusion"(%177, %arg113) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %179 = "mhlo.fusion"(%178, %arg114, %154, %arg108, %172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %180 = "mhlo.fusion"(%179) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %181 = mhlo.bitcast %180 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %182 = "mhlo.fusion"(%181, %arg125, %arg126, %arg121) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %183 = "mhlo.fusion"(%182, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %185 = "mhlo.fusion"(%181, %arg125, %arg126, %arg117) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %186 = "mhlo.fusion"(%185, %arg118) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %187 = mhlo.bitcast %186 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %188 = "mhlo.fusion"(%184, %187) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %189 = "mhlo.fusion"(%188) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %190 = mhlo.bitcast %189 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %191 = "mhlo.fusion"(%181, %arg125, %arg126, %arg119) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %192 = "mhlo.fusion"(%191, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %193 = mhlo.bitcast %192 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %194 = "mhlo.fusion"(%190, %193) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %195 = "mhlo.fusion"(%194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %196 = mhlo.bitcast %195 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %197 = "mhlo.fusion"(%196, %arg123) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %198 = "mhlo.fusion"(%179, %arg124, %197) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %200 = "mhlo.fusion"(%199, %arg131, %arg132, %arg127) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %201 = "mhlo.fusion"(%200, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %202 = mhlo.bitcast %201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %203 = "mhlo.fusion"(%202, %arg129) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %204 = "mhlo.fusion"(%203, %arg130, %179, %arg124, %197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %205 = "mhlo.fusion"(%204) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %206 = mhlo.bitcast %205 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %207 = "mhlo.fusion"(%206, %arg141, %arg142, %arg137) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %208 = "mhlo.fusion"(%207, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %209 = mhlo.bitcast %208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %210 = "mhlo.fusion"(%206, %arg141, %arg142, %arg133) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %211 = "mhlo.fusion"(%210, %arg134) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %212 = mhlo.bitcast %211 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %213 = "mhlo.fusion"(%209, %212) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %214 = "mhlo.fusion"(%213) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %215 = mhlo.bitcast %214 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %216 = "mhlo.fusion"(%206, %arg141, %arg142, %arg135) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %217 = "mhlo.fusion"(%216, %arg136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %219 = "mhlo.fusion"(%215, %218) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %221 = mhlo.bitcast %220 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %222 = "mhlo.fusion"(%221, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %223 = "mhlo.fusion"(%204, %arg140, %222) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %224 = mhlo.bitcast %223 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %225 = "mhlo.fusion"(%224, %arg147, %arg148, %arg143) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %226 = "mhlo.fusion"(%225, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %227 = mhlo.bitcast %226 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %228 = "mhlo.fusion"(%227, %arg145) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %229 = "mhlo.fusion"(%228, %arg146, %204, %arg140, %222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %230 = "mhlo.fusion"(%229) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %231 = mhlo.bitcast %230 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %232 = "mhlo.fusion"(%231, %arg157, %arg158, %arg153) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %233 = "mhlo.fusion"(%232, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %234 = mhlo.bitcast %233 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %235 = "mhlo.fusion"(%231, %arg157, %arg158, %arg149) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %236 = "mhlo.fusion"(%235, %arg150) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %237 = mhlo.bitcast %236 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %238 = "mhlo.fusion"(%234, %237) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %239 = "mhlo.fusion"(%238) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %240 = mhlo.bitcast %239 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %241 = "mhlo.fusion"(%231, %arg157, %arg158, %arg151) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %242 = "mhlo.fusion"(%241, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %243 = mhlo.bitcast %242 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %244 = "mhlo.fusion"(%240, %243) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %245 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %246 = mhlo.bitcast %245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %247 = "mhlo.fusion"(%246, %arg155) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %248 = "mhlo.fusion"(%229, %arg156, %247) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %249 = mhlo.bitcast %248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %250 = "mhlo.fusion"(%249, %arg163, %arg164, %arg159) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %251 = "mhlo.fusion"(%250, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %252 = mhlo.bitcast %251 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %253 = "mhlo.fusion"(%252, %arg161) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %254 = "mhlo.fusion"(%253, %arg162, %229, %arg156, %247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %255 = "mhlo.fusion"(%254) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %256 = mhlo.bitcast %255 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %257 = "mhlo.fusion"(%256, %arg173, %arg174, %arg169) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %258 = "mhlo.fusion"(%257, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %259 = mhlo.bitcast %258 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %260 = "mhlo.fusion"(%256, %arg173, %arg174, %arg165) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %261 = "mhlo.fusion"(%260, %arg166) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %262 = mhlo.bitcast %261 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %263 = "mhlo.fusion"(%259, %262) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %264 = "mhlo.fusion"(%263) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %265 = mhlo.bitcast %264 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %266 = "mhlo.fusion"(%256, %arg173, %arg174, %arg167) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %267 = "mhlo.fusion"(%266, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %268 = mhlo.bitcast %267 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %269 = "mhlo.fusion"(%265, %268) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %270 = "mhlo.fusion"(%269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %271 = mhlo.bitcast %270 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %272 = "mhlo.fusion"(%271, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %273 = "mhlo.fusion"(%254, %arg172, %272) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %274 = mhlo.bitcast %273 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %275 = "mhlo.fusion"(%274, %arg179, %arg180, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %276 = "mhlo.fusion"(%275, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %277 = mhlo.bitcast %276 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %278 = "mhlo.fusion"(%277, %arg177) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %279 = "mhlo.fusion"(%278, %arg178, %254, %arg172, %272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      mhlo.return %314 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %280 = "mhlo.fusion"(%279) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %309 = stablehlo.multiply %307, %308 : tensor<1024x50xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %311 = stablehlo.subtract %arg201, %310 : tensor<1024x50x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1024x50x768xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %314 = stablehlo.multiply %313, %308 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %316 = stablehlo.add %314, %315 : tensor<1024x50xf32>
      %317 = stablehlo.rsqrt %316 : tensor<1024x50xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %319 = stablehlo.multiply %311, %318 : tensor<1024x50x768xf32>
      mhlo.return %319 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>) -> tensor<1024x50x768xf32>
    %281 = mhlo.bitcast %280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %282 = "mhlo.fusion"(%281, %arg189, %arg190, %arg185) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %283 = "mhlo.fusion"(%282, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %312 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %284 = mhlo.bitcast %283 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %285 = "mhlo.fusion"(%281, %arg189, %arg190, %arg181) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %286 = "mhlo.fusion"(%285, %arg182) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %310 = stablehlo.multiply %308, %309 : tensor<51200x768xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %312 = stablehlo.transpose %311, dims = [0, 2, 3, 1] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      mhlo.return %312 : tensor<1024x12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x64x50xf32>
    %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
    %288 = "mhlo.fusion"(%284, %287) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>, %arg202: tensor<12288x64x50xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %307 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %289 = "mhlo.fusion"(%288) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %309 = stablehlo.compare EQ, %307, %308 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %310 = stablehlo.not %309 : tensor<1024x12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %311 = stablehlo.reduce(%310 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      %312 = stablehlo.not %311 : tensor<1024x12x50xi1>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %315 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12288x50x50xf32>, tensor<f32>) -> tensor<12288x50xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12288x50xf32>) -> tensor<1024x12x50xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %318 = stablehlo.subtract %307, %317 : tensor<1024x12x50x50xf32>
      %319 = stablehlo.exponential %318 : tensor<1024x12x50x50xf32>
      %320 = stablehlo.reduce(%319 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %322 = stablehlo.divide %319, %321 : tensor<1024x12x50x50xf32>
      %323 = stablehlo.select %313, %314, %322 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      mhlo.return %323 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %290 = mhlo.bitcast %289 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
    %291 = "mhlo.fusion"(%281, %arg189, %arg190, %arg183) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %311 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %292 = "mhlo.fusion"(%291, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %310 = stablehlo.transpose %309, dims = [0, 2, 1, 3] : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      mhlo.return %310 : tensor<1024x12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x12x50x64xf32>
    %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
    %294 = "mhlo.fusion"(%290, %293) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>, %arg202: tensor<12288x50x64xf32>):
      %307 = stablehlo.dot_general %arg202, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x64xf32>, tensor<12288x50x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %307 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12288,64,50]{1,2,0}"} : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x64x50xf32>
    %295 = "mhlo.fusion"(%294) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x64x50xf32>):
      %307 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<12288x64x50xf32>) -> tensor<1024x12x50x64xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1, 3] : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      mhlo.return %308 : tensor<1024x50x12x64xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x64x50xf32>) -> tensor<1024x50x12x64xf32>
    %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
    %297 = "mhlo.fusion"(%296, %arg187) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768x768xf32>):
      %307 = stablehlo.transpose %arg202, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      %309 = stablehlo.dot_general %308, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x768x768xf32>, tensor<1024x50x768xf32>) -> tensor<1024x768x50xf32>
      mhlo.return %309 : tensor<1024x768x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,768,50]{1,2,0}"} : (tensor<1024x50x768xf32>, tensor<768x768xf32>) -> tensor<1024x768x50xf32>
    %298 = "mhlo.fusion"(%279, %arg188, %297) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg201, %309 : tensor<1024x50x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %312 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %313 = stablehlo.multiply %311, %312 : tensor<1024x50xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %315 = stablehlo.subtract %310, %314 : tensor<1024x50x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1024x50x768xf32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      %318 = stablehlo.multiply %317, %312 : tensor<1024x50xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %320 = stablehlo.add %318, %319 : tensor<1024x50xf32>
      %321 = stablehlo.rsqrt %320 : tensor<1024x50xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %323 = stablehlo.multiply %315, %322 : tensor<1024x50x768xf32>
      mhlo.return %323 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
    %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
    %300 = "mhlo.fusion"(%299, %arg195, %arg196, %arg191) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<51200x768xf32>
      %309 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<51200x768xf32>
      %311 = stablehlo.dot_general %310, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
      mhlo.return %311 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %301 = "mhlo.fusion"(%300, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<51200x3072xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %312 = stablehlo.multiply %309, %311 : tensor<1024x50x3072xf32>
      %313 = stablehlo.negate %312 : tensor<1024x50x3072xf32>
      %314 = stablehlo.exponential %313 : tensor<1024x50x3072xf32>
      %315 = stablehlo.add %310, %314 : tensor<1024x50x3072xf32>
      %316 = stablehlo.divide %310, %315 : tensor<1024x50x3072xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<1024x50x3072xf32>
      mhlo.return %317 : tensor<1024x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<1024x50x3072xf32>
    %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
    %303 = "mhlo.fusion"(%302, %arg193) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<768x3072xf32>):
      %307 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
      mhlo.return %307 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %304 = "mhlo.fusion"(%303, %arg194, %279, %arg188, %297) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50x768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x768x50xf32>):
      %307 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<1024x768x50xf32>) -> tensor<1024x50x768xf32>
      %308 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %309 = stablehlo.add %307, %308 : tensor<1024x50x768xf32>
      %310 = stablehlo.add %arg203, %309 : tensor<1024x50x768xf32>
      %311 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %312 = stablehlo.add %311, %arg201 : tensor<51200x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %314 = stablehlo.add %310, %313 : tensor<1024x50x768xf32>
      %315 = stablehlo.slice %314 [0:1024, 1:50, 0:768] : (tensor<1024x50x768xf32>) -> tensor<1024x49x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x49x768xf32>, tensor<f32>) -> tensor<1024x768xf32>
      mhlo.return %316 : tensor<1024x768xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>, tensor<1024x768x50xf32>) -> tensor<1024x768xf32>
    %305 = "mhlo.fusion"(%304, %arg197) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x768xf32>, %arg202: tensor<2x768xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x768xf32>
      %308 = stablehlo.multiply %arg201, %307 : tensor<1024x768xf32>
      %309 = stablehlo.dot_general %308, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x768xf32>, tensor<2x768xf32>) -> tensor<1024x2xf32>
      mhlo.return %309 : tensor<1024x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x768xf32>, tensor<2x768xf32>) -> tensor<1024x2xf32>
    %306 = "mhlo.fusion"(%305, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x2xf32>, %arg202: tensor<2xf32>):
      %307 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<2xf32>) -> tensor<1024x2xf32>
      %308 = stablehlo.add %307, %arg201 : tensor<1024x2xf32>
      mhlo.return %308 : tensor<1024x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x2xf32>, tensor<2xf32>) -> tensor<1024x2xf32>
    return %306 : tensor<1024x2xf32>
  }
}
