module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot.2_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.101_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.102_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.4_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.109_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.110_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.6_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.116_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.117_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.118_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.8_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.124_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.125_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.10_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.136_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.12_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.140_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.141_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.142_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.143_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.144_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.14_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.148_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.149_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.150_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.151_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.152_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.16_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.156_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.157_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.158_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.159_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.160_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.18_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.164_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.165_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.166_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.167_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.168_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.20_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.172_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.173_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.174_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.175_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.176_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.22_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.180_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.181_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.182_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.183_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.184_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot.24_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<197x2304xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
    return %6 : tensor<197x2304xf32>
  }
  func.func private @gemm_fusion_dot_general.188_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x197xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x197xf32>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x197xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x197xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x197xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @gemm_fusion_dot_general.189_computation(%arg0: tensor<12x197x197xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x197xf32>) -> tensor<197x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x197xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    return %4 : tensor<197x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.190_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @gemm_fusion_dot_general.191_computation(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<197x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<197x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %3 = stablehlo.add %1, %2 : tensor<197x768xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    return %4 : tensor<197x3072xf32>
  }
  func.func private @gemm_fusion_dot_general.192_computation(%arg0: tensor<197x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<197x768xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    return %0 : tensor<197x768xf32>
  }
  func.func private @region_0.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.76(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<197x768xf32>, %arg6: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %7 = stablehlo.add %6, %arg3 : tensor<197x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.add %8, %arg2 : tensor<1x197x768xf32>
    %10 = stablehlo.add %5, %9 : tensor<1x197x768xf32>
    %11 = stablehlo.add %2, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x197xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %19 = stablehlo.subtract %11, %18 : tensor<1x197x768xf32>
    %20 = stablehlo.multiply %19, %19 : tensor<1x197x768xf32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %22 = stablehlo.reduce(%21 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %24 = stablehlo.multiply %23, %15 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %26 = stablehlo.add %24, %25 : tensor<1x197xf32>
    %27 = stablehlo.rsqrt %26 : tensor<1x197xf32>
    %28 = mhlo.bitcast %27 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %30 = stablehlo.multiply %19, %29 : tensor<1x197x768xf32>
    return %30 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.80(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg0 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<1x197xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %15 = stablehlo.subtract %7, %14 : tensor<1x197x768xf32>
    %16 = stablehlo.multiply %15, %15 : tensor<1x197x768xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %20 = stablehlo.multiply %19, %11 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197xf32>
    %23 = stablehlo.rsqrt %22 : tensor<1x197xf32>
    %24 = mhlo.bitcast %23 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %26 = stablehlo.multiply %15, %25 : tensor<1x197x768xf32>
    return %26 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.82(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.86(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.88(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.92(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.94(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.98(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.100(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.104(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.106(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.110(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.112(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.116(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.118(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.122(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.124(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.128(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.130(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.134(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.136(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.140(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.142(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x197xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.subtract %3, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x197x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %16 = stablehlo.multiply %15, %7 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x197xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x197xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %22 = stablehlo.multiply %11, %21 : tensor<1x197x768xf32>
    return %22 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.146(%arg0: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x197xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.subtract %arg0, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x197x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %12 = stablehlo.multiply %11, %3 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x197xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %18 = stablehlo.multiply %7, %17 : tensor<1x197x768xf32>
    return %18 : tensor<1x197x768xf32>
  }
  func.func private @fused_slice(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.1(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.2(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.3(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.4(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.5(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.6(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.7(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.8(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.9(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.10(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.11(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.12(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.13(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.14(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.15(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.16(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.17(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.18(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.19(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.20(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.21(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.22(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.23(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.24(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.25(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.26(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.27(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.28(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.29(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.30(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.31(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.32(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.33(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.34(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_slice.35(%arg0: tensor<197x2304xf32>) -> tensor<768x197xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
    %1 = stablehlo.slice %0 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
    return %1 : tensor<768x197xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<197x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<1x197x768xf32>
    %7 = stablehlo.add %2, %6 : tensor<1x197x768xf32>
    return %7 : tensor<1x197x768xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x197x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<197x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<197x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %5 = stablehlo.negate %2 : tensor<197x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<197x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x197x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x197x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x197x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x197x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x197x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x197x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x197x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x197x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x197x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x197x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x197x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x197x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x197x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x197x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x197x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x197x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x197x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x197x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x197x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x197x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x197x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x197x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x197x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x197x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x197x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x197x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x197x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x197x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x197x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x197x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x197x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x197x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x197x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x197x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x197x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x197x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x197x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x197x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x197x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x197x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x197x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x197x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x197x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x197x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x197x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x197x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x197x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x197x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x197x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x197x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x197x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x197x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x197x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x197x3072xf32>
    return %98 : tensor<1x197x3072xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<1x768x14x14xf32>) -> tensor<1x196x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x14x14xf32>) -> tensor<1x768x196xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<1x768x196xf32>) -> tensor<1x196x768xf32>
    return %1 : tensor<1x196x768xf32>
  }
  func.func private @region_0.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.530(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.531(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.532(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.533(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.534(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.535(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.536(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.537(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.538(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.539(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.540(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @region_0.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.541(%arg0: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
    %2 = stablehlo.not %1 : tensor<12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
    %4 = stablehlo.not %3 : tensor<12x197xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x197x197xf32>
    %10 = stablehlo.exponential %9 : tensor<12x197x197xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x197x197xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
    return %14 : tensor<12x197x197xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x1x768xf32>, %arg2: tensor<1x196x768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.concatenate %arg1, %arg2, dim = 1 : (tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1x197x768xf32>
    return %1 : tensor<1x197x768xf32>
  }
  func.func private @region_0.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.930(%arg0: tensor<1000xf32>, %arg1: tensor<1000x768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<197x768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<197x768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<1x197x768xf32>, %arg9: tensor<197x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<197x768xf32>, %arg12: tensor<768xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg6 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<197x768xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg12, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %7 = stablehlo.add %6, %arg11 : tensor<197x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.broadcast_in_dim %arg10, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %10 = stablehlo.add %9, %arg9 : tensor<197x768xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %12 = stablehlo.add %11, %arg8 : tensor<1x197x768xf32>
    %13 = stablehlo.add %8, %12 : tensor<1x197x768xf32>
    %14 = stablehlo.add %5, %13 : tensor<1x197x768xf32>
    %15 = stablehlo.add %2, %14 : tensor<1x197x768xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %18 = mhlo.bitcast %17 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<1x197xf32>
    %21 = mhlo.bitcast %20 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %23 = stablehlo.subtract %15, %22 : tensor<1x197x768xf32>
    %24 = stablehlo.multiply %23, %23 : tensor<1x197x768xf32>
    %25 = mhlo.bitcast %24 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
    %27 = mhlo.bitcast %26 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
    %28 = stablehlo.multiply %27, %19 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %30 = stablehlo.add %28, %29 : tensor<1x197xf32>
    %31 = stablehlo.rsqrt %30 : tensor<1x197xf32>
    %32 = mhlo.bitcast %31 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %34 = stablehlo.multiply %23, %33 : tensor<1x197x768xf32>
    %35 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %36 = stablehlo.multiply %34, %35 : tensor<1x197x768xf32>
    %37 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %38 = stablehlo.add %36, %37 : tensor<1x197x768xf32>
    %39 = stablehlo.slice %38 [0:1, 0:1, 0:768] : (tensor<1x197x768xf32>) -> tensor<1x1x768xf32>
    %40 = mhlo.bitcast %39 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [1] : (tensor<768xf32>) -> tensor<1000x768xf32>
    %42 = stablehlo.multiply %41, %arg1 : tensor<1000x768xf32>
    %43 = stablehlo.reduce(%42 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1000x768xf32>, tensor<f32>) -> tensor<1000xf32>
    %44 = stablehlo.add %arg0, %43 : tensor<1000xf32>
    return %44 : tensor<1000xf32>
  }
  func.func @main(%arg0: tensor<1x1x768xf32>, %arg1: tensor<1x197x768xf32>, %arg2: tensor<768x3x16x16xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<768x768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<768x768xf32>, %arg9: tensor<768xf32>, %arg10: tensor<768x768xf32>, %arg11: tensor<768xf32>, %arg12: tensor<3072x768xf32>, %arg13: tensor<3072xf32>, %arg14: tensor<768x3072xf32>, %arg15: tensor<768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<768xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768x768xf32>, %arg21: tensor<768xf32>, %arg22: tensor<768x768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768x768xf32>, %arg25: tensor<768xf32>, %arg26: tensor<768x768xf32>, %arg27: tensor<768xf32>, %arg28: tensor<3072x768xf32>, %arg29: tensor<3072xf32>, %arg30: tensor<768x3072xf32>, %arg31: tensor<768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768x768xf32>, %arg37: tensor<768xf32>, %arg38: tensor<768x768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768x768xf32>, %arg41: tensor<768xf32>, %arg42: tensor<768x768xf32>, %arg43: tensor<768xf32>, %arg44: tensor<3072x768xf32>, %arg45: tensor<3072xf32>, %arg46: tensor<768x3072xf32>, %arg47: tensor<768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768x768xf32>, %arg53: tensor<768xf32>, %arg54: tensor<768x768xf32>, %arg55: tensor<768xf32>, %arg56: tensor<768x768xf32>, %arg57: tensor<768xf32>, %arg58: tensor<768x768xf32>, %arg59: tensor<768xf32>, %arg60: tensor<3072x768xf32>, %arg61: tensor<3072xf32>, %arg62: tensor<768x3072xf32>, %arg63: tensor<768xf32>, %arg64: tensor<768xf32>, %arg65: tensor<768xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768x768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768x768xf32>, %arg71: tensor<768xf32>, %arg72: tensor<768x768xf32>, %arg73: tensor<768xf32>, %arg74: tensor<768x768xf32>, %arg75: tensor<768xf32>, %arg76: tensor<3072x768xf32>, %arg77: tensor<3072xf32>, %arg78: tensor<768x3072xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768x768xf32>, %arg87: tensor<768xf32>, %arg88: tensor<768x768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768x768xf32>, %arg91: tensor<768xf32>, %arg92: tensor<3072x768xf32>, %arg93: tensor<3072xf32>, %arg94: tensor<768x3072xf32>, %arg95: tensor<768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768x768xf32>, %arg101: tensor<768xf32>, %arg102: tensor<768x768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768x768xf32>, %arg105: tensor<768xf32>, %arg106: tensor<768x768xf32>, %arg107: tensor<768xf32>, %arg108: tensor<3072x768xf32>, %arg109: tensor<3072xf32>, %arg110: tensor<768x3072xf32>, %arg111: tensor<768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768x768xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768x768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768x768xf32>, %arg121: tensor<768xf32>, %arg122: tensor<768x768xf32>, %arg123: tensor<768xf32>, %arg124: tensor<3072x768xf32>, %arg125: tensor<3072xf32>, %arg126: tensor<768x3072xf32>, %arg127: tensor<768xf32>, %arg128: tensor<768xf32>, %arg129: tensor<768xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768x768xf32>, %arg133: tensor<768xf32>, %arg134: tensor<768x768xf32>, %arg135: tensor<768xf32>, %arg136: tensor<768x768xf32>, %arg137: tensor<768xf32>, %arg138: tensor<768x768xf32>, %arg139: tensor<768xf32>, %arg140: tensor<3072x768xf32>, %arg141: tensor<3072xf32>, %arg142: tensor<768x3072xf32>, %arg143: tensor<768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<768xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768x768xf32>, %arg149: tensor<768xf32>, %arg150: tensor<768x768xf32>, %arg151: tensor<768xf32>, %arg152: tensor<768x768xf32>, %arg153: tensor<768xf32>, %arg154: tensor<768x768xf32>, %arg155: tensor<768xf32>, %arg156: tensor<3072x768xf32>, %arg157: tensor<3072xf32>, %arg158: tensor<768x3072xf32>, %arg159: tensor<768xf32>, %arg160: tensor<768xf32>, %arg161: tensor<768xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768x768xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768x768xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768x768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<768x768xf32>, %arg171: tensor<768xf32>, %arg172: tensor<3072x768xf32>, %arg173: tensor<3072xf32>, %arg174: tensor<768x3072xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768x768xf32>, %arg181: tensor<768xf32>, %arg182: tensor<768x768xf32>, %arg183: tensor<768xf32>, %arg184: tensor<768x768xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768x768xf32>, %arg187: tensor<768xf32>, %arg188: tensor<3072x768xf32>, %arg189: tensor<3072xf32>, %arg190: tensor<768x3072xf32>, %arg191: tensor<768xf32>, %arg192: tensor<768xf32>, %arg193: tensor<768xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<1000x768xf32>, %arg199: tensor<1000xf32>, %arg200: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = mhlo.bitcast %arg181 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %1 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%arg200, %arg2, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,768,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x3x224x224xf32>, tensor<768x3x16x16xf32>, tensor<768xf32>) -> tuple<tensor<1x768x14x14xf32>, tensor<0xui8>>
    %2 = stablehlo.get_tuple_element %1[0] : (tuple<tensor<1x768x14x14xf32>, tensor<0xui8>>) -> tensor<1x768x14x14xf32>
    %3 = "mhlo.fusion"(%2) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x14x14xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x14x14xf32>) -> tensor<1x768x196xf32>
      %293 = stablehlo.transpose %292, dims = [0, 2, 1] : (tensor<1x768x196xf32>) -> tensor<1x196x768xf32>
      mhlo.return %293 : tensor<1x196x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x14x14xf32>) -> tensor<1x196x768xf32>
    %4 = "mhlo.fusion"(%arg1, %arg0, %3) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x1x768xf32>, %arg203: tensor<1x196x768xf32>):
      %292 = stablehlo.concatenate %arg202, %arg203, dim = 1 : (tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<1x197x768xf32>
      mhlo.return %293 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %6 = "mhlo.fusion"(%4) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %8 = "mhlo.fusion"(%7, %arg16, %arg17, %arg4, %arg6, %arg8) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %9 = "mhlo.fusion"(%8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %11 = mhlo.bitcast %arg7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %12 = "mhlo.fusion"(%8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %14 = "mhlo.fusion"(%5, %10, %11, %13) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %15 = "mhlo.fusion"(%14) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %16 = mhlo.bitcast %arg9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %17 = "mhlo.fusion"(%8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %18 = mhlo.bitcast %17 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %19 = "mhlo.fusion"(%15, %16, %18) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %21 = "mhlo.fusion"(%20, %arg10) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %22 = "mhlo.fusion"(%4, %21, %arg11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %24 = "mhlo.fusion"(%23, %arg18, %arg19, %arg12) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %25 = "mhlo.fusion"(%24, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %27 = "mhlo.fusion"(%26, %arg14) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %28 = "mhlo.fusion"(%27, %arg15, %4, %21, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %29 = mhlo.bitcast %arg21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %30 = "mhlo.fusion"(%28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %31 = mhlo.bitcast %30 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %32 = "mhlo.fusion"(%31, %arg32, %arg33, %arg20, %arg22, %arg24) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %33 = "mhlo.fusion"(%32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %34 = mhlo.bitcast %33 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %35 = mhlo.bitcast %arg23 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %36 = "mhlo.fusion"(%32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %37 = mhlo.bitcast %36 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %38 = "mhlo.fusion"(%29, %34, %35, %37) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %39 = "mhlo.fusion"(%38) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %40 = mhlo.bitcast %arg25 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %41 = "mhlo.fusion"(%32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %43 = "mhlo.fusion"(%39, %40, %42) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %44 = mhlo.bitcast %43 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %45 = "mhlo.fusion"(%44, %arg26) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %46 = "mhlo.fusion"(%28, %45, %arg27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %47 = mhlo.bitcast %46 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %48 = "mhlo.fusion"(%47, %arg34, %arg35, %arg28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %49 = "mhlo.fusion"(%48, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %50 = mhlo.bitcast %49 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %51 = "mhlo.fusion"(%50, %arg30) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %52 = "mhlo.fusion"(%51, %arg31, %28, %45, %arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %53 = mhlo.bitcast %arg37 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %54 = "mhlo.fusion"(%52) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %55 = mhlo.bitcast %54 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %56 = "mhlo.fusion"(%55, %arg48, %arg49, %arg36, %arg38, %arg40) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %57 = "mhlo.fusion"(%56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %58 = mhlo.bitcast %57 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %59 = mhlo.bitcast %arg39 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %60 = "mhlo.fusion"(%56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %62 = "mhlo.fusion"(%53, %58, %59, %61) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %63 = "mhlo.fusion"(%62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %64 = mhlo.bitcast %arg41 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %65 = "mhlo.fusion"(%56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %66 = mhlo.bitcast %65 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %67 = "mhlo.fusion"(%63, %64, %66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %68 = mhlo.bitcast %67 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %69 = "mhlo.fusion"(%68, %arg42) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %70 = "mhlo.fusion"(%52, %69, %arg43) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %71 = mhlo.bitcast %70 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %72 = "mhlo.fusion"(%71, %arg50, %arg51, %arg44) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %73 = "mhlo.fusion"(%72, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %74 = mhlo.bitcast %73 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %75 = "mhlo.fusion"(%74, %arg46) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %76 = "mhlo.fusion"(%75, %arg47, %52, %69, %arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %77 = mhlo.bitcast %arg53 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %78 = "mhlo.fusion"(%76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %79 = mhlo.bitcast %78 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %80 = "mhlo.fusion"(%79, %arg64, %arg65, %arg52, %arg54, %arg56) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %81 = "mhlo.fusion"(%80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %83 = mhlo.bitcast %arg55 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %84 = "mhlo.fusion"(%80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %85 = mhlo.bitcast %84 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %86 = "mhlo.fusion"(%77, %82, %83, %85) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %87 = "mhlo.fusion"(%86) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %88 = mhlo.bitcast %arg57 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %89 = "mhlo.fusion"(%80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %90 = mhlo.bitcast %89 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %91 = "mhlo.fusion"(%87, %88, %90) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %92 = mhlo.bitcast %91 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %93 = "mhlo.fusion"(%92, %arg58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %94 = "mhlo.fusion"(%76, %93, %arg59) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %95 = mhlo.bitcast %94 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %96 = "mhlo.fusion"(%95, %arg66, %arg67, %arg60) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %97 = "mhlo.fusion"(%96, %arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %98 = mhlo.bitcast %97 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %99 = "mhlo.fusion"(%98, %arg62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %100 = "mhlo.fusion"(%99, %arg63, %76, %93, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %101 = mhlo.bitcast %arg69 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %102 = "mhlo.fusion"(%100) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %103 = mhlo.bitcast %102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %104 = "mhlo.fusion"(%103, %arg80, %arg81, %arg68, %arg70, %arg72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %105 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %106 = mhlo.bitcast %105 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %107 = mhlo.bitcast %arg71 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %108 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %109 = mhlo.bitcast %108 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %110 = "mhlo.fusion"(%101, %106, %107, %109) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %111 = "mhlo.fusion"(%110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %112 = mhlo.bitcast %arg73 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %113 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %114 = mhlo.bitcast %113 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %115 = "mhlo.fusion"(%111, %112, %114) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %116 = mhlo.bitcast %115 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %117 = "mhlo.fusion"(%116, %arg74) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %118 = "mhlo.fusion"(%100, %117, %arg75) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %119 = mhlo.bitcast %118 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %120 = "mhlo.fusion"(%119, %arg82, %arg83, %arg76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %121 = "mhlo.fusion"(%120, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %122 = mhlo.bitcast %121 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %123 = "mhlo.fusion"(%122, %arg78) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %124 = "mhlo.fusion"(%123, %arg79, %100, %117, %arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %125 = mhlo.bitcast %arg85 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %126 = "mhlo.fusion"(%124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %127 = mhlo.bitcast %126 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %128 = "mhlo.fusion"(%127, %arg96, %arg97, %arg84, %arg86, %arg88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %129 = "mhlo.fusion"(%128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %131 = mhlo.bitcast %arg87 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %132 = "mhlo.fusion"(%128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %133 = mhlo.bitcast %132 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %134 = "mhlo.fusion"(%125, %130, %131, %133) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %135 = "mhlo.fusion"(%134) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %136 = mhlo.bitcast %arg89 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %137 = "mhlo.fusion"(%128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %138 = mhlo.bitcast %137 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %139 = "mhlo.fusion"(%135, %136, %138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %140 = mhlo.bitcast %139 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %141 = "mhlo.fusion"(%140, %arg90) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %142 = "mhlo.fusion"(%124, %141, %arg91) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %143 = mhlo.bitcast %142 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %144 = "mhlo.fusion"(%143, %arg98, %arg99, %arg92) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %145 = "mhlo.fusion"(%144, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %146 = mhlo.bitcast %145 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %147 = "mhlo.fusion"(%146, %arg94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %148 = "mhlo.fusion"(%147, %arg95, %124, %141, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %149 = mhlo.bitcast %arg101 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %150 = "mhlo.fusion"(%148) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %152 = "mhlo.fusion"(%151, %arg112, %arg113, %arg100, %arg102, %arg104) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %153 = "mhlo.fusion"(%152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %154 = mhlo.bitcast %153 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %155 = mhlo.bitcast %arg103 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %156 = "mhlo.fusion"(%152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %158 = "mhlo.fusion"(%149, %154, %155, %157) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %160 = mhlo.bitcast %arg105 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %161 = "mhlo.fusion"(%152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %162 = mhlo.bitcast %161 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %163 = "mhlo.fusion"(%159, %160, %162) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %164 = mhlo.bitcast %163 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %165 = "mhlo.fusion"(%164, %arg106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %166 = "mhlo.fusion"(%148, %165, %arg107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %167 = mhlo.bitcast %166 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %168 = "mhlo.fusion"(%167, %arg114, %arg115, %arg108) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %169 = "mhlo.fusion"(%168, %arg109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %170 = mhlo.bitcast %169 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %171 = "mhlo.fusion"(%170, %arg110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %172 = "mhlo.fusion"(%171, %arg111, %148, %165, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %173 = mhlo.bitcast %arg117 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %174 = "mhlo.fusion"(%172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %175 = mhlo.bitcast %174 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %176 = "mhlo.fusion"(%175, %arg128, %arg129, %arg116, %arg118, %arg120) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %177 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %178 = mhlo.bitcast %177 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %179 = mhlo.bitcast %arg119 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %180 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %181 = mhlo.bitcast %180 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %182 = "mhlo.fusion"(%173, %178, %179, %181) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %184 = mhlo.bitcast %arg121 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %185 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %187 = "mhlo.fusion"(%183, %184, %186) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %189 = "mhlo.fusion"(%188, %arg122) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %190 = "mhlo.fusion"(%172, %189, %arg123) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %191 = mhlo.bitcast %190 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %192 = "mhlo.fusion"(%191, %arg130, %arg131, %arg124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %193 = "mhlo.fusion"(%192, %arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %194 = mhlo.bitcast %193 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %195 = "mhlo.fusion"(%194, %arg126) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %196 = "mhlo.fusion"(%195, %arg127, %172, %189, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %197 = mhlo.bitcast %arg133 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %198 = "mhlo.fusion"(%196) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %200 = "mhlo.fusion"(%199, %arg144, %arg145, %arg132, %arg134, %arg136) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %201 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %202 = mhlo.bitcast %201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %203 = mhlo.bitcast %arg135 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %204 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %205 = mhlo.bitcast %204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %206 = "mhlo.fusion"(%197, %202, %203, %205) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %207 = "mhlo.fusion"(%206) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %208 = mhlo.bitcast %arg137 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %209 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %210 = mhlo.bitcast %209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %211 = "mhlo.fusion"(%207, %208, %210) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %212 = mhlo.bitcast %211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %213 = "mhlo.fusion"(%212, %arg138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %214 = "mhlo.fusion"(%196, %213, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %215 = mhlo.bitcast %214 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %216 = "mhlo.fusion"(%215, %arg146, %arg147, %arg140) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %217 = "mhlo.fusion"(%216, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %219 = "mhlo.fusion"(%218, %arg142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %220 = "mhlo.fusion"(%219, %arg143, %196, %213, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %221 = mhlo.bitcast %arg149 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %222 = "mhlo.fusion"(%220) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %223 = mhlo.bitcast %222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %224 = "mhlo.fusion"(%223, %arg160, %arg161, %arg148, %arg150, %arg152) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %225 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %226 = mhlo.bitcast %225 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %227 = mhlo.bitcast %arg151 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %228 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %230 = "mhlo.fusion"(%221, %226, %227, %229) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %231 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %232 = mhlo.bitcast %arg153 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %233 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %234 = mhlo.bitcast %233 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %235 = "mhlo.fusion"(%231, %232, %234) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %236 = mhlo.bitcast %235 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %237 = "mhlo.fusion"(%236, %arg154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %238 = "mhlo.fusion"(%220, %237, %arg155) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %239 = mhlo.bitcast %238 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %240 = "mhlo.fusion"(%239, %arg162, %arg163, %arg156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %241 = "mhlo.fusion"(%240, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %242 = mhlo.bitcast %241 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %243 = "mhlo.fusion"(%242, %arg158) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %244 = "mhlo.fusion"(%243, %arg159, %220, %237, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg204 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg203 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      mhlo.return %299 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %245 = mhlo.bitcast %arg165 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %246 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x197xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %299 = stablehlo.subtract %arg201, %298 : tensor<1x197x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x197x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.reduce(%301 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %303, %295 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %306 = stablehlo.add %304, %305 : tensor<1x197xf32>
      %307 = stablehlo.rsqrt %306 : tensor<1x197xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %310 = stablehlo.multiply %299, %309 : tensor<1x197x768xf32>
      mhlo.return %310 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %247 = mhlo.bitcast %246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %248 = "mhlo.fusion"(%247, %arg176, %arg177, %arg164, %arg166, %arg168) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %249 = "mhlo.fusion"(%248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %250 = mhlo.bitcast %249 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %251 = mhlo.bitcast %arg167 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %252 = "mhlo.fusion"(%248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %253 = mhlo.bitcast %252 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %254 = "mhlo.fusion"(%245, %250, %251, %253) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %255 = "mhlo.fusion"(%254) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %256 = mhlo.bitcast %arg169 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %257 = "mhlo.fusion"(%248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %258 = mhlo.bitcast %257 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %259 = "mhlo.fusion"(%255, %256, %258) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %260 = mhlo.bitcast %259 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %261 = "mhlo.fusion"(%260, %arg170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %262 = "mhlo.fusion"(%244, %261, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.add %294, %arg201 : tensor<1x197x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x197xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %303 = stablehlo.subtract %295, %302 : tensor<1x197x768xf32>
      %304 = stablehlo.multiply %303, %303 : tensor<1x197x768xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %307, %299 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x197xf32>
      %311 = stablehlo.rsqrt %310 : tensor<1x197xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %314 = stablehlo.multiply %303, %313 : tensor<1x197x768xf32>
      mhlo.return %314 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %263 = mhlo.bitcast %262 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %264 = "mhlo.fusion"(%263, %arg178, %arg179, %arg172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %265 = "mhlo.fusion"(%264, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %266 = mhlo.bitcast %265 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %267 = "mhlo.fusion"(%266, %arg174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %268 = "mhlo.fusion"(%244, %261, %arg171, %267, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg204 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg202 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.add %297, %arg201 : tensor<1x197x768xf32>
      %299 = stablehlo.add %294, %298 : tensor<1x197x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%300 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<1x197xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %307 = stablehlo.subtract %299, %306 : tensor<1x197x768xf32>
      %308 = stablehlo.multiply %307, %307 : tensor<1x197x768xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %312 = stablehlo.multiply %311, %303 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197xf32>
      %315 = stablehlo.rsqrt %314 : tensor<1x197xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %318 = stablehlo.multiply %307, %317 : tensor<1x197x768xf32>
      mhlo.return %318 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %269 = mhlo.bitcast %268 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %270 = "mhlo.fusion"(%269, %arg192, %arg193, %arg180, %arg182, %arg184) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %298 = stablehlo.dot %295, %297, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768x2304xf32>) -> tensor<197x2304xf32>
      mhlo.return %298 : tensor<197x2304xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[197,2304]{0,1}"} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<197x2304xf32>
    %271 = "mhlo.fusion"(%270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [0:768, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %272 = mhlo.bitcast %271 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %273 = mhlo.bitcast %arg183 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %274 = "mhlo.fusion"(%270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [768:1536, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %275 = mhlo.bitcast %274 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %276 = "mhlo.fusion"(%0, %272, %273, %275) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x197xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<12x64x197xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x197xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %297 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %298 = stablehlo.add %297, %arg204 : tensor<12x64x197xf32>
      %299 = stablehlo.multiply %298, %294 : tensor<12x64x197xf32>
      %300 = stablehlo.dot_general %296, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %300 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %293 = stablehlo.compare EQ, %arg201, %292 : (tensor<12x197x197xf32>, tensor<12x197x197xf32>) -> tensor<12x197x197xi1>
      %294 = stablehlo.not %293 : tensor<12x197x197xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %295 = stablehlo.reduce(%294 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x197x197xi1>, tensor<i1>) -> tensor<12x197xi1>
      %296 = stablehlo.not %295 : tensor<12x197xi1>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<12x197xi1>) -> tensor<12x197x197xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x197x197xf32>
      %299 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %301 = stablehlo.subtract %arg201, %300 : tensor<12x197x197xf32>
      %302 = stablehlo.exponential %301 : tensor<12x197x197xf32>
      %303 = stablehlo.reduce(%302 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
      %305 = stablehlo.divide %302, %304 : tensor<12x197x197xf32>
      %306 = stablehlo.select %297, %298, %305 : tensor<12x197x197xi1>, tensor<12x197x197xf32>
      mhlo.return %306 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
    %278 = mhlo.bitcast %arg185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %279 = "mhlo.fusion"(%270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x2304xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<197x2304xf32>) -> tensor<2304x197xf32>
      %293 = stablehlo.slice %292 [1536:2304, 0:197] : (tensor<2304x197xf32>) -> tensor<768x197xf32>
      mhlo.return %293 : tensor<768x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x2304xf32>) -> tensor<768x197xf32>
    %280 = mhlo.bitcast %279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x197xf32>) -> tensor<12x64x197xf32>
    %281 = "mhlo.fusion"(%277, %278, %280) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x197xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x197xf32>
      %293 = stablehlo.add %292, %arg203 : tensor<12x64x197xf32>
      %294 = stablehlo.transpose %293, dims = [0, 2, 1] : (tensor<12x64x197xf32>) -> tensor<12x197x64xf32>
      %295 = stablehlo.dot_general %294, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,197]{1,2,0}"} : (tensor<12x197x64xf32>, tensor<12x197x197xf32>) -> tensor<12x64x197xf32>
      %296 = stablehlo.transpose %295, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
      mhlo.return %296 : tensor<197x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[197,64,12]{1,2,0}"} : (tensor<12x197x197xf32>, tensor<12x64xf32>, tensor<12x64x197xf32>) -> tensor<197x64x12xf32>
    %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<197x64x12xf32>) -> tensor<197x768xf32>
    %283 = "mhlo.fusion"(%282, %arg186) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %284 = "mhlo.fusion"(%283, %arg187, %244, %261, %arg171, %267, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197x768xf32>, %arg204: tensor<197x768xf32>, %arg205: tensor<768xf32>, %arg206: tensor<197x768xf32>, %arg207: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg201 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg207, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg206 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.broadcast_in_dim %arg205, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %299 = stablehlo.add %298, %arg204 : tensor<197x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %301 = stablehlo.add %300, %arg203 : tensor<1x197x768xf32>
      %302 = stablehlo.add %297, %301 : tensor<1x197x768xf32>
      %303 = stablehlo.add %294, %302 : tensor<1x197x768xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<1x197xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %311 = stablehlo.subtract %303, %310 : tensor<1x197x768xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<1x197x768xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %315 = mhlo.bitcast %314 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %316 = stablehlo.multiply %315, %307 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %318 = stablehlo.add %316, %317 : tensor<1x197xf32>
      %319 = stablehlo.rsqrt %318 : tensor<1x197xf32>
      %320 = mhlo.bitcast %319 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %322 = stablehlo.multiply %311, %321 : tensor<1x197x768xf32>
      mhlo.return %322 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %285 = mhlo.bitcast %284 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    %286 = "mhlo.fusion"(%285, %arg194, %arg195, %arg188) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.multiply %arg201, %292 : tensor<197x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %295 = stablehlo.add %293, %294 : tensor<197x768xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
      mhlo.return %296 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %287 = "mhlo.fusion"(%286, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<197x3072xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<197x3072xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %297 = stablehlo.negate %294 : tensor<197x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<197x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %301 = stablehlo.abs %300 : tensor<1x197x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x197x3072xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x197x3072xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x197x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x197x3072xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x197x3072xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x197x3072xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x197x3072xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x197x3072xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x197x3072xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %329 = stablehlo.negate %305 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %332 = stablehlo.exponential %329 : tensor<1x197x3072xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x197x3072xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x197x3072xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x197x3072xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x197x3072xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x197x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x197x3072xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x197x3072xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x197x3072xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x197x3072xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x197x3072xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x197x3072xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x197x3072xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x197x3072xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x197x3072xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x197x3072xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x197x3072xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x197x3072xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x197x3072xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x197x3072xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x197x3072xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x197x3072xf32>
      mhlo.return %390 : tensor<1x197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<1x197x3072xf32>
    %288 = mhlo.bitcast %287 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    %289 = "mhlo.fusion"(%288, %arg190) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
      mhlo.return %292 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %290 = "mhlo.fusion"(%arg199, %arg198, %arg197, %arg196, %283, %arg187, %289, %arg191, %244, %261, %arg171, %267, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1000xf32>, %arg202: tensor<1000x768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<197x768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<197x768xf32>, %arg208: tensor<768xf32>, %arg209: tensor<1x197x768xf32>, %arg210: tensor<197x768xf32>, %arg211: tensor<768xf32>, %arg212: tensor<197x768xf32>, %arg213: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %293 = stablehlo.add %292, %arg207 : tensor<197x768xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg206, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %296 = stablehlo.add %295, %arg205 : tensor<197x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %298 = stablehlo.broadcast_in_dim %arg213, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %299 = stablehlo.add %298, %arg212 : tensor<197x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %301 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %302 = stablehlo.add %301, %arg210 : tensor<197x768xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %304 = stablehlo.add %303, %arg209 : tensor<1x197x768xf32>
      %305 = stablehlo.add %300, %304 : tensor<1x197x768xf32>
      %306 = stablehlo.add %297, %305 : tensor<1x197x768xf32>
      %307 = stablehlo.add %294, %306 : tensor<1x197x768xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %312 = stablehlo.multiply %310, %311 : tensor<1x197xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %315 = stablehlo.subtract %307, %314 : tensor<1x197x768xf32>
      %316 = stablehlo.multiply %315, %315 : tensor<1x197x768xf32>
      %317 = mhlo.bitcast %316 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %318 = stablehlo.reduce(%317 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<197x768xf32>, tensor<f32>) -> tensor<197xf32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<197xf32>) -> tensor<1x197xf32>
      %320 = stablehlo.multiply %319, %311 : tensor<1x197xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %322 = stablehlo.add %320, %321 : tensor<1x197xf32>
      %323 = stablehlo.rsqrt %322 : tensor<1x197xf32>
      %324 = mhlo.bitcast %323 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %326 = stablehlo.multiply %315, %325 : tensor<1x197x768xf32>
      %327 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %328 = stablehlo.multiply %326, %327 : tensor<1x197x768xf32>
      %329 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %330 = stablehlo.add %328, %329 : tensor<1x197x768xf32>
      %331 = stablehlo.slice %330 [0:1, 0:1, 0:768] : (tensor<1x197x768xf32>) -> tensor<1x1x768xf32>
      %332 = mhlo.bitcast %331 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
      %333 = stablehlo.broadcast_in_dim %332, dims = [1] : (tensor<768xf32>) -> tensor<1000x768xf32>
      %334 = stablehlo.multiply %333, %arg202 : tensor<1000x768xf32>
      %335 = stablehlo.reduce(%334 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1000x768xf32>, tensor<f32>) -> tensor<1000xf32>
      %336 = stablehlo.add %arg201, %335 : tensor<1000xf32>
      mhlo.return %336 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<1000x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<197x768xf32>, tensor<768xf32>, tensor<197x768xf32>, tensor<768xf32>, tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1000xf32>
    %291 = mhlo.bitcast %290 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %291 : tensor<1x1000xf32>
  }
}
